import os
import torch
import torch.distributed as dist


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError("Unknown boolean type")


def load_network(name):
    # Dynamically import the module
    mod = __import__('KLM.{}'.format(name), fromlist=[''])
    # Return the class from the module
    return getattr(mod, name.upper())


def setup_dist(rank, world_size):
    """
    Setup a distributed process group
    """
    if dist.is_initialized():
        return
    backend = "gloo" if not torch.cuda.is_available() else "nccl"
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_PORT'] = "24108"
    # initialize the process group
    dist.init_process_group(backend=backend, init_method="env://")


def torch_set_gpu(gpus):
    if type(gpus) is int:
        gpus = [gpus]
    cuda = all(gpu >= 0 for gpu in gpus)
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in gpus])
        assert cuda and torch.cuda.is_available(), \
            f"{os.environ['HOSTNAME']} has GPUs {os.environ['CUDA_VISIBLE_DEVICES']} unavailable"
        torch.backends.cudnn.benchmark = True  # speed-up cudnn
        torch.backends.cudnn.fastest = True  # even more speed-up?
        print(f"Launching on GPUs {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        print('Launching on CPU')
    return cuda

