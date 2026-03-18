# import debugpy;debugpy.connect(('22.9.35.97', 6792))
import argparse
import random
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from util.misc import collate_fn_with_mask as collate_fn
from engine import evaluate
from models import build_model

from datasets import build_dataset, test_transforms

from util.logger import get_logger
from util.config import Config


def get_args_parser():
    parser = argparse.ArgumentParser('SSP_SAM test script', add_help=False)
    # Model/test essentials
    parser.add_argument('--is_pretrain', action='store_true')
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers (used for aux loss weights)")
    parser.add_argument('--clip_pretrained', default='pretrained_checkpoints/CS/CS-ViT-L-14-336px.pt', type=str,
                        help='Path to the clip (surgery version).')
    parser.add_argument('--max_query_len', default=40, type=int,
                        help='The maximum total input sequence length after WordPiece tokenization.')

    # Loss settings needed by build_model
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--loss_loc', default='loss_masks', type=str,
                        help="The loss function for the predicted masks")
    parser.add_argument('--loss_loc_rec', default='loss_boxes', type=str,
                        help="The loss function for the predicted boxes")
    parser.add_argument('--box_xyxy', action='store_true',
                        help='Use xyxy format to encode bounding boxes')

    # * Loss coefficients
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--dice_loss_coef', default=4, type=float)
    parser.add_argument('--focal_loss_coef', default=4, type=float)
    parser.add_argument('--other_loss_coefs', default={}, type=float)

    # Dataset/test parameters
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--split_root', default='./split/data/')
    parser.add_argument('--dataset', default='referit')
    parser.add_argument('--test_split', default='testA')
    parser.add_argument('--output_dir', default='work_dirs/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--save_pred_path', default='')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--checkpoint', default='', help='resume from checkpoint')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_memory', default=True, type=boolean_string)
    parser.add_argument('--batch_size_test', default=1, type=int)
    parser.add_argument('--test_transforms', default=test_transforms)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # configure file
    parser.add_argument('--config', default='configs/SSP_SAM_CLIP_L_FT_referit.py', type=str, help='Path to the configure file.')
    return parser



def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def main(args):
    utils.init_distributed_mode(args)

    logger = get_logger("test", None, utils.get_rank())

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessor = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module


    dataset_test = build_dataset(test=True, args=args)

    logger.info(f'The size of dataset: test({len(dataset_test)})')

    if args.distributed:
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = DataLoader(dataset_test, args.batch_size_test, sampler=sampler_test,
                                 pin_memory=args.pin_memory, drop_last=True,
                                 collate_fn=collate_fn, num_workers=args.num_workers)

    output_dir = Path(args.output_dir)
    assert args.checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    

    i = 0
    ckpt_name = []
    tmp = OrderedDict()
    for name, param in checkpoint['model'].items():
        if not name.startswith('encoder'):
            i += 1
            ckpt_name.append(name)
            tmp[name] = param
    checkpoint['model'] = tmp

    model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

    i = 0
    model_name = []
    for name, param in model_without_ddp.encoder.named_parameters():
        i += 1
        model_name.append('encoder.'+name)

    if args.dataset == "phrase_cut":
        test_stats, test_acc, test_time = evaluate_phrase_cut(
            model, criterion, postprocessor, data_loader_test, device, args.save_pred_path
        )
    else:
        test_stats, test_acc, test_time = evaluate(
            model, criterion, postprocessor, data_loader_test, device, args.save_pred_path
        )
    logger.info('  '.join(['[Test accuracy]', *[f'{k}: {v:.4f}' for k, v in test_acc.items()]]))
    logger.info('  '.join(['[Test time]', *[f'{k}: {v:.6f}' for k, v in test_time.items()]]))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SSP-SAM test script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.config:
        cfg = Config(args.config)
        # Keep test parser minimal: only apply keys that exist in test args.
        for k, v in cfg._cfg_dict.items():
            if hasattr(args, k):
                setattr(args, k, v)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
