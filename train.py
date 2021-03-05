import argparse
import os
import pickle
import random
import shutil
import sys
import time
import warnings

import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.utils import clip_grad_value_
from torch.utils.tensorboard import SummaryWriter

import utils
from data_loader import ModifiedImageFolder

best_acc1 = 0


def save_checkpoint(state, is_best, out_dir, name='', filename='checkpoint.pth.tar'):
    """Saves a model.

    Args:
        state (dict): model information
        is_best (bool): True - if current model has the best metric, False - otherwise
        out_dir (str): directory to save results
        name (str): unique name associated with current model
        filename (str): suffix for above name. Default: `checkpoint.pth.tar`
    """
    if not utils.touch(out_dir):
        print('Trying to save the model at /tmp')
        out_dir = '/tmp'
    torch.save(state, f'{out_dir}/{name}_{filename}')
    if is_best:
        shutil.copyfile(f'{out_dir}/{name}_{filename}', f'{out_dir}/{name}_model_best.pth.tar')
    print(f'Model saved at: {out_dir}/{name}_{filename}')


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs

    Args:
        optimizer ():
        epoch ():
        args ():
    """
    lr = args.lr * (0.1 ** ((epoch // 30) + (epoch // 75)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_parser():
    """"Defines the command line arguments"""
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
    parser.add_argument('--out_dir', required=True,
                        help='directory to be used to save the results')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet50)')
    parser.add_argument('--arch_name', required=True)

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')

    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    parser.add_argument('-affine', action='store_true')
    parser.add_argument('-no_augment', action='store_true')
    parser.add_argument('-resume', action='store_true')
    parser.add_argument('--add_name', default=None, type=str,
                        help='add extra name. Can be thought of as experiment name')

    parser.add_argument('--optim', default="sgd", type=str,
                        help='optimiser to use used.')
    parser.add_argument('--lr_scheduler', default=None, type=str,
                        help='lr_scheduler used.')
    parser.add_argument('--step', default=None, type=int,
                        help='Step size for lr scheduler')
    parser.add_argument('--gamma', default=None, type=float,
                        help='Step size for lr scheduler')

    parser.add_argument('--model_file', default=None, type=str,
                        help='Model file on which evaluatuion needs to be done')

    parser.add_argument('-dump_eval', action='store_true')
    parser.add_argument('-corr', action='store_true')
    return parser


def write_tb_text(writer, args):
    args_to_write = ['epochs', 'batch_size', 'optim', 'lr', 'weight_decay', 'momentum', 'lr_scheduler']
    for attr in args_to_write:
        if attr == 'lr_scheduler':
            if getattr(args, attr):
                text = 'type: {}, gamma: {} , step: {}'.format(getattr(args, attr), getattr(args, 'gamma'),
                                                               getattr(args, 'step'))
            else:
                text = 'None'
            writer.add_text(attr, text)
        else:
            writer.add_text(attr, str(getattr(args, attr)))

    return


def main():
    parser = get_parser()
    args = parser.parse_args()
    name = args.arch_name
    if args.add_name is not None:
        name += '_' + args.add_name
    if args.no_augment:
        name += '_no_augment'
    if args.affine:
        name += '_all_augment'
    if args.optim != 'sgd':
        name += '_{}_'.format(args.optim)
    if args.lr_scheduler:
        name += '_lrsche_' + args.lr_scheduler
    setattr(args, 'name', name)

    if not utils.touch(args.out_dir):
        print('Process will be terminated')
        sys.exit()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if args.dist_url == 'env://' and args.world_size == -1:
        args.world_size = int(os.environ['WORLD_SIZE'])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        try:
            writer = SummaryWriter('tf_log/{}'.format(args.name), flush_secs=10)
        except:
            pass

    if args.gpu is not None:
        print('Use GPU: {} for training'.format(args.gpu))

    if args.distributed:
        if args.dist_url == 'env://' and args.rank == -1:
            args.rank = int(os.environ['RANK'])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch_name))
        model = models.__dict__[args.arch_name](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch_name))

        if args.corr:
            if 'efficientnet' in args.arch_name:
                model = timm.create_model(args.arch_name, num_classes=413)
            else:
                model = models.__dict__[args.arch_name](num_classes=413)
        else:
            if 'efficientnet' in args.arch_name:
                model = timm.create_model(args.arch_name)
            else:
                model = models.__dict__[args.arch_name]()

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch_name.startswith('alexnet') or args.arch_name.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    if args.optim.lower() == 'sgd':

        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optim.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.lr,
                                        momentum=args.momentum)

    # optionally resume from a checkpoint
    if args.evaluate:
        resume_file = args.model_file
    else:
        resume_file = 'weights/{}_checkpoint.pth.tar'.format(args.name)
    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        if args.gpu is None:
            checkpoint = torch.load(resume_file)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(resume_file, map_location=loc)
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        best_acc5 = checkpoint['best_acc5']
        if args.gpu is not None:
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(args.gpu)
            best_acc5 = best_acc5.to(args.gpu)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume_file, checkpoint['epoch']))
    else:
        if args.evaluate:
            raise ValueError("Model file not specified correctly.")

        print("=> no checkpoint found at '{}'".format(args.name))

    cudnn.benchmark = True

    # Data loading code
    train_dir = os.path.join(args.data, 'train')
    val_dir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])  # Advied by Saketh, change later

    if args.lr_scheduler == 'exponential':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    elif args.lr_scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step, gamma=0.1)
    elif args.lr_scheduler is None:
        lr_scheduler = None
    else:
        raise NotImplementedError("Not implemented {} lr scheduler currently".format(args.lr_scheduler))

    out_dir = args.out_dir.rstrip('/')
    if args.no_augment:
        print('no_augment')
        train_dataset = ModifiedImageFolder(
            train_dir,
            out_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                normalize,
            ]), stage='train')

    elif args.affine:
        print('afffine on')
        train_dataset = ModifiedImageFolder(
            train_dir,
            out_dir,
            transforms.Compose([
                transforms.RandomAffine(degrees=(-45, 45), translate=(0.3, 0.3), scale=(0.5, 1.5)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), stage='train')
    else:
        train_dataset = ModifiedImageFolder(
            train_dir,
            out_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), stage='train')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        ModifiedImageFolder(val_dir, out_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), stage='val'),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        write_tb_text(writer, args)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.optim.lower() == 'sgd':
            adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_top1, train_top5, train_loss = train(train_loader, model, criterion, optimizer, epoch, args)
        if args.optim.lower() == 'rmsprop':
            lr_scheduler.step()

        # evaluate on validation set

        val_top1, val_top5, val_loss = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = val_top1 > best_acc1
        best_acc1 = max(val_top1, best_acc1)
        if is_best:
            best_acc5 = val_top5

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            writer.add_scalar('loss/train', train_loss, epoch)
            writer.add_scalar('loss/val', val_loss, epoch)
            writer.add_scalar('top1/train', train_top1, epoch)
            writer.add_scalar('top1/val', val_top1, epoch)
            writer.add_scalar('top5/train', train_top5, epoch)
            writer.add_scalar('top5/val', val_top5, epoch)

            save_checkpoint({
                'arch': args.arch_name,
                'best_acc1': best_acc1,
                'best_acc5': best_acc5,
                'epoch': epoch + 1,
                'model': 1,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
            }, is_best, args.out_dir, name=args.name)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    progress = utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix='Epoch: [{}]'.format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = utils.accuracy(output, target, top_k=(1, 5))
        losses.update(loss.detach().item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.optim.lower() == 'rmsprop':
            clip_grad_value_(model.parameters(), 10)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, criterion, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    progress = utils.ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode

    model.eval()
    softmax = nn.Softmax(dim=-1)
    dump_array = np.zeros((0, 413))
    dump_targets = np.zeros((0,))

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)[:, :413]
            loss = criterion(output, target)
            dump_array = np.vstack([dump_array, softmax(output).cpu().numpy()])
            dump_targets = np.hstack([dump_targets, target.cpu().numpy()])

            # measure accuracy and record loss
            acc1, acc5 = utils.accuracy(output, target, top_k=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    if args.dump_eval:
        import pdb
        pdb.set_trace()
        pickle.dump(dump_array, open('eval_dump.pkl', 'wb'))
        pickle.dump(dump_targets, open('eval_target.pkl', 'wb'))

    return top1.avg, top5.avg, losses.avg


if __name__ == '__main__':
    main()
