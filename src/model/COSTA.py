
import wandb
from model.encoder import GCNEncoder
from model.base import BaseGSSLRunner
import torch
import torch.nn.functional as F
import GCL.augmentors as A
from GCL.eval import get_split, LREvaluator, from_predefined_split
from tqdm import tqdm
from util.helper import _similarity
from torch.optim import Adam
from util.data import get_dataset


class DualBranchContrast(torch.nn.Module):
    def __init__(self, loss, mode, intraview_negs=False, **kwargs):
        super(DualBranchContrast, self).__init__()
        self.loss = loss
        self.kwargs = kwargs

    def forward(self, h1=None, h2=None):
        l1 = self.loss(anchor=h1, sample=h2)
        l2 = self.loss(anchor=h2, sample=h1)
        return (l1 + l2) * 0.5


class InfoNCE(object):
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample):
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob.diag()
        return -loss.mean()

    def __call__(self, anchor, sample) -> torch.FloatTensor:
        loss = self.compute(anchor, sample)
        return loss


class COSTA(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim, device):
        super(COSTA, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        self.device = device
        self.hidden_dim = hidden_dim

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


class Runner(BaseGSSLRunner):
    def __init__(self, conf, **kwargs):
        super().__init__(conf, **kwargs)
    
    def load_dataset(self):
        self.dataset = get_dataset(self.config['data_dir'], self.config['dataset'])
    # dataset = Planetoid(data_dir, name=args.dataset, transform=T.NormalizeFeatures())
        self.data = self.dataset[0].to(self.device)
        
    def train(self):
        aug1 = A.Compose([A.EdgeRemoving(pe=self.config['drop_edge_rate_1']),
                        A.FeatureMasking(pf=self.config['drop_feature_rate_1'])])
        aug2 = A.Compose([A.EdgeRemoving(pe=self.config['drop_edge_rate_2']),
                        A.FeatureMasking(pf=self.config['drop_feature_rate_1'])])

        gconv = GCNEncoder(input_dim=self.dataset.num_features,
                    hidden_dim=self.config['num_hidden'], activation=torch.nn.ReLU, num_layers=self.config['num_layers']).to(self.device)

        self.model = COSTA(encoder=gconv, 
                            augmentor=(aug1, aug2),
                            hidden_dim=self.config['num_hidden'],
                            proj_dim=self.config['num_proj_hidden'],
                            device=self.device)
        self.model = self.model.to(self.device)

        contrast_model = DualBranchContrast(loss=InfoNCE(
            tau=self.config['tau']), mode='L2L', intraview_negs=True).to(self.device)

        optimizer = Adam(self.model.parameters(), lr=self.config['learning_rate'])


        with tqdm(total=self.config['num_epochs'], desc='(T)') as pbar:
            for epoch in range(1, self.config['num_epochs']+1):
                self.model.train()
                optimizer.zero_grad()

                z, z1, z2 = self.model(self.data.x, self.data.edge_index, self.data.edge_attr)
                
                k = torch.tensor(int(z.shape[0] * self.config["ratio"]))
                p = (1/torch.sqrt(k))*torch.randn(k, z.shape[0]).to(self.device)

                z1 = p @ z1
                z2 = p @ z2 
               
                h1, h2 = [self.model.project(x) for x in [z1, z2]]
     
                
                loss = contrast_model(h1, h2)
                loss.backward()
                optimizer.step()

                wandb.log({'loss': loss.item()})
                pbar.set_postfix({'loss': loss.item()})
                pbar.update()

                if  self.config['test_every_epoch'] and epoch % 10 == 0:
                    self.test( t='random')
                   

       

    def test(self, t="random"):
        self.model.eval()
        z, _, _ = self.model(self.data.x, self.data.edge_index, self.data.edge_attr)
        if t == 'random':
            split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
        if t == 'public':
            split = from_predefined_split(self.data)
        result = LREvaluator(num_epochs=self.config["lr_num_epochs"])(z, self.data.y, split)
#        wandb.log(result)
        print(f"(E): Best test F1Mi={result['micro_f1']:.4f}, F1Ma={result['macro_f1']:.4f}")

            
