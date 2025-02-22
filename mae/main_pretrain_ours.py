# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import wandb

import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset

#import timm
#import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_pretrain_ours import train_one_epoch_ours, validate_vis_img2
from save_relevance import save_relevance_maps
from dataset_mae import MAEDataset
import models_lib
from transformers import AutoTokenizer
from cliport.models.core.clip import CLIPResTokenizer
import sys


#assert timm.__version__ == "0.3.2"  # version check
MEAN_CLIPORT = [0.48145466, 0.4578275, 0.40821073]
STD_CLIPORT = [0.26862954, 0.26130258, 0.27577711]
PATH = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5/exist_dataset_no_aug_all.hdf5'
TEST_PATH = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5/exist_dataset_no_aug_all_test.hdf5'

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    
    # Training paramters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='None', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=(224, 224), nargs=2, type=int,
                    help="Images input size as two integers, e.g., '224 224'")
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default=PATH, type=str,
                        help='dataset path')
    parser.add_argument('--test_path', default=TEST_PATH, type=str,
                        help='test dataset path')
    parser.add_argument('--output_dir', default='./debug',
                        help='path where to save, empty for no saving')
    parser.add_argument('--my_log', action='store_true',
                        help='log training process')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=False)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # ours parameters and other function
    parser.add_argument('--pretrain', default=None, type=str)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--save_ca', action='store_true', help='save cross attention maps')
    parser.add_argument('--wandb_resume', default=None, type=str)
    parser.add_argument('--save_relevance', action='store_true')
    #parser.add_argument('--stand_norm',action='store_true')  abandoned
    parser.add_argument('--transform', default='None', type=str)
    parser.add_argument('--aug', action='store_true')
    parser.add_argument('--text_model', default="openai/clip-vit-base-patch32")
    parser.add_argument('--condition_free', action='store_true')
    return parser

def get_flip_transform():
    transform_flip = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Normalize(mean=MEAN_CLIPORT, std=STD_CLIPORT)])
    return transform_flip

def get_fix_transform():
    trasform_fix = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_CLIPORT, std=STD_CLIPORT)])
    return trasform_fix

def get_fix_transform_standnorm():
    
    class StandardNormalize(object):
        def __call__(self, tensor):
            min_val = torch.min(tensor)
            max_val = torch.max(tensor)
            normalized_tensor = (tensor - min_val) / (max_val - min_val)
            scaled_tensor = normalized_tensor * 2 - 1
            return scaled_tensor
    
    trasform_fix = transforms.Compose([
        transforms.ToTensor(),
        StandardNormalize()])
    return trasform_fix

def get_aug_transform(input_size):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic # type: ignore
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_CLIPORT, std=STD_CLIPORT)])
    return transform_train

def get_voltron_transform():
    NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform_voltron = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=NORMALIZATION[0], std=NORMALIZATION[1])])
    return transform_voltron

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # enable this one if inputs are varient during training
    #cudnn.benchmark = True

    # choose augmentations
    if args.transform == 'flip':
        transform_train = get_flip_transform()
    elif args.transform == 'stand_norm':
        transform_train = get_fix_transform_standnorm()
    else:
        transform_train = get_fix_transform()
    
    # replace with voltron transform if model is voltron
    if 'voltron' in args.model:
       transform_train = get_voltron_transform()

    dataset_train = MAEDataset(transform=transform_train, data_path=args.data_path, aug=args.aug, condition_free=args.condition_free)
    dataset_vis = MAEDataset(transform=transform_train, data_path=args.test_path, aug=False)
    #dataset_train = Subset(dataset_train, range(600))

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        print("num tasks and global rank:", num_tasks, global_rank)
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.my_log:
        wandb.init(project='MAE', name=args.model, entity='cxz', id=args.wandb_resume)
        log_writer = wandb
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    data_loader_vis = torch.utils.data.DataLoader(
        dataset_vis, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=args.pin_mem
    )

    # define the model
    args.input_size = tuple(args.input_size)  # Convert list to tuple
    model = models_lib.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, text_model=args.text_model, img_size=args.input_size)
    print("Imported model: %s" % args.model)
    model.to(device)
    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))

    # momentum schedule
    if 'jepa' in args.model:
        print("Using JEPA hyperparameters")
        ema = [0.996, 1.0]
        ipe_scale = 1.0
        ipe = len(data_loader_train)
        num_epochs = args.epochs
        momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                            for i in range(int(ipe*num_epochs*ipe_scale)+1))
    else:
        momentum_scheduler = None

    # define the text processor
    if 'res' in args.model:
        text_processor = CLIPResTokenizer()
    elif 'voltron' in args.model:
        text_processor = AutoTokenizer.from_pretrained("distilbert-base-uncased", cache_dir='cache/hf-cache')
    else:
        text_processor = AutoTokenizer.from_pretrained(args.text_model)
    
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    # load pre-trained model
    if args.pretrain:
        misc.dynamic_load_pretrain(model_without_ddp, args.pretrain)
    else:
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    
    # condition free
    if args.condition_free:
        print("Enabled condition free training")
        model_without_ddp.copy_mask_tokens()

# ============== Demo mode ==============
    if args.demo:
        validate_vis_img2(model_without_ddp,
                          data_loader_vis,
                          device, 0,
                          log_writer=log_writer,
                          args=args,
                          text_processor=text_processor)
        sys.exit(0)
        

# ============== Save relevance maps ==============
    if args.save_relevance:
        
        data_loader_vis = torch.utils.data.DataLoader(
            dataset_vis, batch_size=128, num_workers=2, drop_last=False, shuffle=False, pin_memory=True
        )

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=128, num_workers=2, drop_last=False, shuffle=False, pin_memory=True
        )
        
        save_relevance_maps(model_without_ddp, data_loader_train, data_loader_vis, device, args, text_processor)
        sys.exit(0)

# ============== Training ==============
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch_ours(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args,
            text_processor=text_processor,
            momentum_scheduler=momentum_scheduler
        )

        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

            validate_vis_img2(model_without_ddp, 
                              data_loader_vis, device, 
                              (epoch + 1)*1000, 
                              log_writer=log_writer, 
                              args=args,
                              text_processor=text_processor)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
