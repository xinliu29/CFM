import argparse
import yaml
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler as DS
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from trainer import Trainer
from dataset.megadepth import Megadepth
from utils import setup_dist, torch_set_gpu, load_network, str2bool

# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
# torch.autograd.set_detect_anomaly(True)
torch.set_grad_enabled(True)

parser = argparse.ArgumentParser(description='CFM', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--network', type=str, default='cfm')
parser.add_argument('--feature', type=str, default='spp')
parser.add_argument('--filter_type', type=str, choices=['nn', 'n3net', 'sinkhorn'], default='nn')
parser.add_argument('--valid_layers', nargs='+', type=int, default=[3, 5, 8])
parser.add_argument('--iter_num', type=int, default=1)
parser.add_argument('--sigma_spat', type=float, default=0.2)
parser.add_argument('--use_prune', type=str2bool, default=False)
parser.add_argument('--layer_prune', nargs='+', type=str2bool, default=[True, True])
parser.add_argument('--threshold', nargs='+', type=float, default=[-0.975, -0.8])
parser.add_argument('--n_min_tokens', nargs='+', type=float, default=[0.95, 0.9])
parser.add_argument('--last_sinkhorn', type=str2bool, default=False)
parser.add_argument('--use_global', type=str2bool, default=True)
parser.add_argument('--config', type=str, default="configs/train_megadepth.yaml")
parser.add_argument('--base_dir', type=str, default=".")
parser.add_argument('--resume_path', type=str, default="None")


def train_distributed(rank, world_size, model, args):
    args.local_rank = rank
    torch.cuda.set_device(rank)
    train_set = Megadepth(
        base_path=args.base_path,
        scene_list_fn=args.scene_list_fn,
        pairs_per_scene=args.pairs_per_scene,
        image_size=args.image_size,
        nfeatures=args.max_keypoints,
        feature_type=args.feature,
        train=True,
        min_inliers=args.min_inliers,
        max_inliers=args.max_inliers,
        random_inliers=args.random_inliers,
    )

    device = torch.device(f'cuda:{rank}')
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    setup_dist(rank=rank, world_size=world_size)
    model = DDP(model, device_ids=[rank])
    train_sampler = DS(train_set, shuffle=False, drop_last=True)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size // world_size, num_workers=args.workers // world_size,
                              pin_memory=False, sampler=train_sampler)
    Trainer(model=model, train_loader=train_loader, eval_loader=None, args=args).train()


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config, 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(yaml.safe_load(f))
        args = parser.parse_args(namespace=t_args)
    args.base_dir = Path(args.base_dir)
    if not args.base_dir.exists():
        args.base_dir = Path("/home/u1120220257/hyb/data/mffm/")
    args.save_path = args.base_dir / 'outputs'
    args.base_path = args.base_dir / args.feature

    torch_set_gpu(gpus=args.gpu)

    net_config = args.net
    net_config['descriptor_dim'] = 256 if args.feature == 'spp' else 128
    net_config['filter_type'] = args.filter_type
    net_config['valid_layers'] = args.valid_layers
    net_config['filter']['iter_num'] = args.iter_num
    net_config['filter']['sigma_spat'] = args.sigma_spat
    net_config['filter']['use_global'] = args.use_global

    net_config['use_prune'] = args.use_prune
    net_config['layer_prune'] = args.layer_prune
    net_config['threshold'] = args.threshold
    net_config['n_min_tokens'] = args.n_min_tokens

    net_config['last_sinkhorn'] = args.last_sinkhorn

    print(net_config)
    network = load_network(args.network)(net_config)

    # load pretrained weight
    if args.resume_path != 'None':
        network.load_state_dict(torch.load(args.save_path / args.resume_path, map_location='cpu')['model'], strict=True)
        print(f'Load resume weight from {args.save_path / args.resume_path}')

    mp.spawn(train_distributed, nprocs=len(args.gpu), args=(len(args.gpu), network, args))
