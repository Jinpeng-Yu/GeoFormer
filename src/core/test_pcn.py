import logging
import torch
import utils.data_loaders
import utils.helpers
from tqdm import tqdm
from utils.average_meter import AverageMeter
from utils.loss_utils import *
from models.SVDFormer import Model
from models.model_utils import PCViews
from utils import misc, dist_utils


def test_net(args, cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, model=None):
    if test_data_loader is None:
        # Set up data loader
        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TEST), shuffle=False)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TEST),
                                                #   batch_size=1,
                                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                                  num_workers=cfg.CONST.NUM_WORKERS//2,
                                                  collate_fn=utils.data_loaders.collate_fn,
                                                  pin_memory=True,
                                                  sampler=test_sampler)


    # Setup networks and initialize networks
    if model is None:
        model = Model(cfg)
        if torch.cuda.is_available():
            model.to(args.local_rank)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)

        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model'])

    # Switch models to evaluation mode
    model.eval()

    n_samples = len(test_data_loader)
    test_losses = AverageMeter(['CD','DCD','F1'])
    test_metrics = AverageMeter(['CD','DCD','F1'])
    category_metrics = dict()
    render = PCViews(TRANS=-cfg.NETWORK.view_distance, RESOLUTION=224)

    # Testing loop
    with tqdm(test_data_loader, disable=args.local_rank != 0) as t:
        for model_idx, (taxonomy_id, model_id, data) in enumerate(t):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            
            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                partial = data['partial_cloud']
                gt = data['gtcloud']

                b, n, _ = partial.shape
                partial_depth = render.get_CCM(partial)

                pcds_pred = model(partial.contiguous(),partial_depth)
                cdl1,cdl2,f1 = calc_cd(pcds_pred[-1],gt,calc_f1=True)
                dcd,_,_ = calc_dcd(pcds_pred[-1],gt)

                cdl1 = dist_utils.reduce_tensor(cdl1, args)
                cdl2 = dist_utils.reduce_tensor(cdl2, args)
                dcd = dist_utils.reduce_tensor(dcd, args)
                f1 = dist_utils.reduce_tensor(f1, args)

                cd = cdl1.mean().item() * 1e3
                dcd = dcd.mean().item()
                f1 = f1.mean().item()

                _metrics = [cd, dcd, f1]
                test_losses.update([cd, dcd, f1])

                test_metrics.update(_metrics)
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(['CD','DCD','F1'])
                category_metrics[taxonomy_id].update(_metrics)

                t.set_description('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                             (model_idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()
                                                                                ], ['%.4f' % m for m in _metrics]))

    torch.cuda.synchronize()

    # Print testing results
    if args.local_rank == 0:
        print('============================ TEST RESULTS ============================')
        print('Taxonomy', end='\t')
        print('#Sample', end='\t')
        for metric in test_metrics.items:
            print(metric, end='\t')
        print()

        for taxonomy_id in category_metrics:
            print(taxonomy_id, end='\t')
            print(category_metrics[taxonomy_id].count(0), end='\t')
            for value in category_metrics[taxonomy_id].avg():
                print('%.4f' % value, end='\t')
            print()

        print('Overall', end='\t\t\t')
        for value in test_metrics.avg():
            print('%.4f' % value, end='\t')
        print('\n')

        print('Epoch ', epoch_idx, end='\t')
        for value in test_losses.avg():
            print('%.4f' % value, end='\t')
        print('\n')

    # Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar('Loss/Epoch/cd', test_losses.avg(0), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/dcd', test_losses.avg(1), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/f1', test_losses.avg(2), epoch_idx)
        for i, metric in enumerate(test_metrics.items):
            test_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch_idx)

    return test_losses.avg(0)
