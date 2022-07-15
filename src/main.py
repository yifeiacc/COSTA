import argparse
import os.path as osp
from tkinter.tix import Tree
import yaml
from yaml import SafeLoader
import wandb
from util.helper import fix_random_seed

support_models = ['SFA','COSTA']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/root/workspace/gssl/')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='default.yaml')
    parser.add_argument('--model', type=str)
    args = parser.parse_args()

    config = yaml.load(open(osp.join(
        osp.join(args.root, 'config'), args.config)), Loader=SafeLoader)[args.dataset]
    config['dataset'] = args.dataset
    config['data_dir'] = osp.join(args.root, 'data')

    wandb.init(project='COSTA', entity='yifeiacc', config={'test_every_epoch': False})
    wandb.config.update(args)
    wandb.config.update(config)

    print(wandb.config)
    fix_random_seed(wandb.config['seed'])
    
    if wandb.config['model'] in support_models:
        exec('from model.{} import Runner'.format(wandb.config['model']))
        eval('Runner(conf=wandb.config).execute()')

    else:
        print("Model {} is not supported".format(wandb.config['model']))
        exit(1)
