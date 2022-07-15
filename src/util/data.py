from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import os.path as osp
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon


# def get_dataset(path, name):
#     assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']
#     name = 'dblp' if name == 'DBLP' else name
#     return (CitationFull if name == 'dblp' else Planetoid)(
#         path,
#         name,
#         transform=T.NormalizeFeatures())

def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy',
                    'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv', 'ogbg-code']
    name = 'dblp' if name == 'DBLP' else name
    # root_path = osp.expanduser('./')

    if name == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())

    if name == 'Coauthor-Phy':
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

    if name == 'WikiCS':
        return WikiCS(root=path, transform=T.NormalizeFeatures())

    if name == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())

    if name == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())

    if name.startswith('ogbn'):
        return PygNodePropPredDataset(root=osp.join(path, 'OGB'), name=name, transform=T.NormalizeFeatures())

    if name == 'dblp':
         return CitationFull(osp.join(path, 'Citation'), name, transform=T.NormalizeFeatures())

    if name in ['Cora', 'CiteSeer', 'PubMed']:
        return Planetoid(path, name, transform=T.NormalizeFeatures())