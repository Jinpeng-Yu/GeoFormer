from utils import dist_utils, misc
import argparse
import logging
import os
import numpy as np
import sys
import torch
from pprint import pprint
from config_pcn import cfg
from core.train_pcn import train_net
from core.test_pcn import test_net
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument parser of SnowflakeNet')
    parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
    parser.add_argument('--inference', dest='inference', help='Inference for benchmark', action='store_true')
    parser.add_argument('--local-rank', type=int, default=0)
    # seed
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    args.log_name = 'PCN'

    return args


def main():
    # Get args from command line
    args = get_args_from_command_line()
    print('cuda available ', torch.cuda.is_available())

    torch.backends.cudnn.benchmark = True
    dist_utils.init_dist(launcher='pytorch', backend='nccl')
    # re-set gpu_ids with distributed training mode
    _, world_size = dist_utils.get_dist_info()
    args.world_size = world_size

    
    # # logger
    # timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # log_file = os.path.join(cfg.DIR.OUT_PATH, f'{timestamp}.log')
    # logger = get_root_logger(log_file=log_file, name=args.log_name)

    # batch size
    assert cfg.TRAIN.BATCH_SIZE % world_size == 0
    cfg.TRAIN.BATCH_SIZE = cfg.TRAIN.BATCH_SIZE // world_size
    assert args.local_rank == torch.distributed.get_rank()

    # # log
    # log_args_to_file(args, 'args', logger = logger)
    # log_config_to_file(cfg, 'config', logger = logger)

    # logger.info(f'Distributed training: {args.distributed}')

    # set random seeds
    if args.seed is not None:
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation

    if args.local_rank == 0:
        # print
        print('Current device count: {}'.format(torch.cuda.device_count()))

        # Print config
        print('Use config:')
        pprint(cfg)

    if not args.test and not args.inference:
        train_net(args, cfg)
    else:
        if cfg.CONST.WEIGHTS is None:
            raise Exception('Please specify the path to checkpoint in the configuration file!')

        test_net(args, cfg)

if __name__ == '__main__':
    # Check python version
    # seed = 1
    # set_seed(seed)
    logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.DEBUG)
    main()
