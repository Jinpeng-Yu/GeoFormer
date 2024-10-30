import logging
import os
import torch
import utils.data_loaders
import utils.helpers
import argparse
from datetime import datetime
from tqdm import tqdm
from time import time
from tensorboardX import SummaryWriter
from core.test_pcn import test_net
from utils.average_meter import AverageMeter
from torch.optim.lr_scheduler import *
from utils.schedular import GradualWarmupScheduler
from utils.loss_utils import *
from models.model_utils import PCViews
from models.SVDFormer import Model
from utils import misc, dist_utils


def train_net(args, cfg):
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TRAIN), shuffle=True)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TRAIN),
                                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                                    num_workers=cfg.CONST.NUM_WORKERS,
                                                    collate_fn=utils.data_loaders.collate_fn,
                                                    pin_memory=True,
                                                    drop_last=False,
                                                    prefetch_factor=8,
                                                    persistent_workers=True,
                                                    sampler=train_sampler)
    val_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TEST), shuffle=False)
    val_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TEST),
                                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                                  num_workers=cfg.CONST.NUM_WORKERS//2,
                                                  collate_fn=utils.data_loaders.collate_fn,
                                                  pin_memory=True,
                                                  drop_last=False,
                                                  prefetch_factor=4,
                                                  persistent_workers=True,
                                                  sampler=val_sampler)

    # Set up folders for logs and checkpoints
    # Create tensorboard writers
    if args.local_rank == 0:
        output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', datetime.now().isoformat())
        cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
        cfg.DIR.LOGS = output_dir % 'logs'
        train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
        val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))
        if not os.path.exists(cfg.DIR.CHECKPOINTS):
            os.makedirs(cfg.DIR.CHECKPOINTS)
    else:
        train_writer = None
        val_writer = None



    model = Model(cfg)
    if torch.cuda.is_available():
        model.to(args.local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
    
    # Create the optimizers
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=cfg.TRAIN.LEARNING_RATE,
                                 weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                 betas=cfg.TRAIN.BETAS)

    # lr scheduler
    scheduler_steplr = MultiStepLR(optimizer,milestones=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS,
                                          after_scheduler=scheduler_steplr)

    init_epoch = 0
    best_metrics = float('inf')
    steps = 0
    BestEpoch = 0
    render = PCViews(TRANS=-cfg.NETWORK.view_distance, RESOLUTION=224)

    if 'WEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        steps = cfg.TRAIN.WARMUP_STEPS+1
        lr_scheduler = MultiStepLR(optimizer,milestones=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
        optimizer.param_groups[0]['lr']= cfg.TRAIN.LEARNING_RATE

        logging.info('Recover complete.')

    # Training/Testing the network
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):
        train_sampler.set_epoch(epoch_idx)
        epoch_start_time = time()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        model.train()

        total_cd_pc = 0
        total_cd_p1 = 0
        total_cd_p2 = 0
        total_partial = 0

        batch_end_time = time()
        n_batches = len(train_data_loader)
        if args.local_rank == 0:
            print('epoch: ', epoch_idx, 'optimizer: ', optimizer.param_groups[0]['lr'])
        with tqdm(train_data_loader, disable=args.local_rank != 0) as t:
            for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(t):
                data_time.update(time() - batch_end_time)
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)
                partial = data['partial_cloud']
                gt = data['gtcloud']

                partial_depth = render.get_CCM(partial)
                
                pcds_pred = model(partial,partial_depth)

                loss_total, losses = get_loss_HyperCD(pcds_pred, partial, gt, sqrt=True)

                optimizer.zero_grad()
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(cfg, 'GRAD_NORM_CLIP', 10), norm_type=2)
                optimizer.step()
                losses = [dist_utils.reduce_tensor(loss, args) for loss in losses]
                torch.cuda.synchronize()

                cd_pc_item = losses[0].item() * 1e3
                total_cd_pc += cd_pc_item
                cd_p1_item = losses[1].item() * 1e3
                total_cd_p1 += cd_p1_item
                cd_p2_item = losses[2].item() * 1e3
                total_cd_p2 += cd_p2_item
                partial_item = losses[3].item() * 1e3
                total_partial += partial_item
                n_itr = (epoch_idx - 1) * n_batches + batch_idx

                if train_writer is not None:
                    train_writer.add_scalar('Loss/Batch/cd_pc', cd_pc_item, n_itr)
                    train_writer.add_scalar('Loss/Batch/cd_p1', cd_p1_item, n_itr)
                    train_writer.add_scalar('Loss/Batch/cd_p2', cd_p2_item, n_itr)
                    train_writer.add_scalar('Loss/Batch/partial', partial_item, n_itr)
                    batch_time.update(time() - batch_end_time)
                    batch_end_time = time()
                    t.set_description('[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches))
                    t.set_postfix(loss='%s' % ['%.4f' % l for l in [cd_pc_item, cd_p1_item, cd_p2_item, partial_item]])

                if steps <= cfg.TRAIN.WARMUP_STEPS:
                    lr_scheduler.step()
                    steps += 1

        avg_cdc = total_cd_pc / n_batches
        avg_cd1 = total_cd_p1 / n_batches
        avg_cd2 = total_cd_p2 / n_batches
        avg_partial = total_partial / n_batches

        lr_scheduler.step()
        epoch_end_time = time()
        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/cd_pc', avg_cdc, epoch_idx)
            train_writer.add_scalar('Loss/Epoch/cd_p1', avg_cd1, epoch_idx)
            train_writer.add_scalar('Loss/Epoch/cd_p2', avg_cd2, epoch_idx)
            train_writer.add_scalar('Loss/Epoch/partial', avg_partial, epoch_idx)
            logging.info(
                '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
                (epoch_idx, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [avg_cdc, avg_cd1, avg_cd2, avg_partial]]))

        # Validate the current model
        cd_eval = test_net(args, cfg, epoch_idx, val_data_loader, val_writer, model)
        # Save checkpoints
        if args.local_rank == 0:
            if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or cd_eval < best_metrics:
                if cd_eval < best_metrics:
                    best_metrics = cd_eval
                    BestEpoch = epoch_idx
                    file_name = 'ckpt-best.pth'

                else:
                    file_name = 'ckpt-epoch-%03d.pth' % epoch_idx
                output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, output_path)

                logging.info('Saved checkpoint to %s ...' % output_path)
            logging.info('Best Performance: Epoch %d -- CD %.4f' % (BestEpoch,best_metrics))

    if train_writer is not None and val_writer is not None:
        train_writer.close()
        val_writer.close()
