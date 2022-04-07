import os
import time
import torch
import socket
import argparse
import subprocess

import torch.nn as nn
import torch.distributed as dist
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from typing import Tuple
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

# from models.resnet50_classifier import Resnet_classifier
from models.resnet import ResNet
from utils import util

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]



def reduce_tensor(tensor: torch.Tensor, world_size: int):
    """Reduce tensor across all nodes."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def to_python_float(t: torch.Tensor):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


def train(train_loader: DataLoader,
          model: nn.Module,
          criterion: nn.Module,
          optimizer: Optimizer,
          epoch: int,
          world_size: int,
          is_master: bool,
          log_interval: int = 100):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # Create non_blocking tensors for distributed training
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # print(output.shape, target.shape)
        # compute gradients in a backward pass
        optimizer.zero_grad()
        loss.backward()

        # Call step of optimizer to update model params
        optimizer.step()

        if i % log_interval == 0:
            # Every log_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))

            # Average loss and accuracy across processes for logging
            reduced_loss = reduce_tensor(loss.data, world_size)
            prec1 = reduce_tensor(prec1, world_size)
            prec5 = reduce_tensor(prec5, world_size)

            # to_python_float incurs a host<->device sync
            batch_size = input[0].size(0)
            losses.update(to_python_float(reduced_loss), batch_size)
            top1.update(to_python_float(prec1), batch_size)
            top5.update(to_python_float(prec5), batch_size)

            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / log_interval)
            end = time.time()

            # Only the first node should log infos.
            if is_master:
                print(
                    f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Speed {world_size * batch_size / batch_time.val:.3f} ({world_size * batch_size / batch_time.avg:.3f})\t"
                    f"Loss {losses.val:.10f} ({losses.avg:.4f})\t"
                    f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                    f"Prec@5 {top5.val:.3f} ({top5.avg:.3f})"
                )


def adjust_learning_rate(initial_lr: float,
                         optimizer: Optimizer,
                         epoch: int):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // 75))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validate(val_loader: DataLoader,
             model: nn.Module,
             criterion: nn.Module,
             world_size: int,
             is_master: bool,
             log_freq: int = 100):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            with torch.no_grad():
                # compute output
                output = model(input)
                loss = criterion(output, target)

            # Measure accuracy
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))

            # Average loss and accuracy across processes for logging
            reduced_loss = reduce_tensor(loss.data, world_size)
            prec1 = reduce_tensor(prec1, world_size)
            prec5 = reduce_tensor(prec5, world_size)

            # to_python_float incurs a host<->device sync
            batch_size = input[0].size(0)
            losses.update(to_python_float(reduced_loss), batch_size)
            top1.update(to_python_float(prec1), batch_size)
            top5.update(to_python_float(prec5), batch_size)

            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / log_freq)
            end = time.time()

            if i % log_freq == 0 and is_master:
                # Only the first node should log infos.
                print(
                    f"Test: [{i}/{len(val_loader)}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Speed {world_size * batch_size / batch_time.val:.3f} ({world_size * batch_size / batch_time.avg:.3f})\t"
                    f"Loss {losses.val:.10f} ({losses.avg:.4f})\t"
                    f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                    f"Prec@5 {top5.val:.3f} ({top5.avg:.3f})"
                )

        if is_master:
            print(f' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}')

    return top1.avg


def run(data_dir: str,
        save_dir: str,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        log_interval: int,
        save_model: bool):
    # number of nodes / node ID
    n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
    node_id = int(os.environ['SLURM_NODEID'])

    # local rank on the current node / global rank
    local_rank = int(os.environ['SLURM_LOCALID'])
    global_rank = int(os.environ['SLURM_PROCID'])

    # number of processes / GPUs per node
    world_size = int(os.environ['SLURM_NTASKS'])
    n_gpu_per_node = world_size // n_nodes

    # define master address and master port
    hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
    master_addr = hostnames.split()[0].decode('utf-8')

    # set environment variables for 'env://'
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(29500)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(global_rank)

    # define whether this is the master process / if we are in distributed mode
    is_master = node_id == 0 and local_rank == 0
    multi_node = n_nodes > 1
    multi_gpu = world_size > 1

    # summary
    PREFIX = "%i - " % global_rank
    print(PREFIX + "Number of nodes: %i" % n_nodes)
    print(PREFIX + "Node ID        : %i" % node_id)
    print(PREFIX + "Local rank     : %i" % local_rank)
    print(PREFIX + "Global rank    : %i" % global_rank)
    print(PREFIX + "World size     : %i" % world_size)
    print(PREFIX + "GPUs per node  : %i" % n_gpu_per_node)
    print(PREFIX + "Master         : %s" % str(is_master))
    print(PREFIX + "Multi-node     : %s" % str(multi_node))
    print(PREFIX + "Multi-GPU      : %s" % str(multi_gpu))
    print(PREFIX + "Hostname       : %s" % socket.gethostname())

    # set GPU device
    torch.cuda.set_device(local_rank)

    print("Initializing PyTorch distributed ...")
    torch.distributed.init_process_group(
        init_method='env://',
        backend='nccl',
    )

    print("Initialize Model...")
    # Construct Model
    model = ResNet()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()
    # Make model DistributedDataParallel
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = SGD(model.parameters(), learning_rate, momentum=0.9, weight_decay=1e-4)

    print("Initialize Dataloaders...")
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    jittering = util.ColorJitter(brightness=0.4, contrast=0.4,
                                  saturation=0.4)
    lighting = util.Lighting(alphastd=0.1,
                              eigval=[0.2175, 0.0188, 0.0045],
                              eigvec=[[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]])


    transform_train =  transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            jittering,
            lighting,
            normalize,
        ])


    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    
    trainset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
    valset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)

    # Create DistributedSampler to handle distributing the dataset across nodes
    # This can only be called after torch.distributed.init_process_group is called
    train_sampler = DistributedSampler(trainset)
    val_sampler = DistributedSampler(valset)

    # Create the Dataloaders to feed data to the training and validation steps
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              num_workers=10,
                              sampler=train_sampler,
                              pin_memory=True)
    val_loader = DataLoader(valset,
                            batch_size=batch_size,
                            num_workers=10,
                            sampler=val_sampler,
                            pin_memory=True)

    best_prec1 = 0

    
    for epoch in range(epochs):
        # Set epoch count for DistributedSampler.
        # We don't need to set_epoch for the validation sampler as we don't want
        # to shuffle for validation.
        train_sampler.set_epoch(epoch)

        # Adjust learning rate according to schedule
        adjust_learning_rate(learning_rate, optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, world_size, is_master, log_interval)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, world_size, is_master)

        # remember best prec@1 and save checkpoint if desired
        if prec1 > best_prec1:
            best_prec1 = prec1
            # if is_master and save_model:
            #     torch.save(model.state_dict(), save_dir + "checkpoint.pt")

        if is_master:
            print("Epoch Summary: ")
            print(f"\tEpoch Accuracy: {prec1}")
            print(f"\tBest Accuracy: {best_prec1}")


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Imagenet')
    parser.add_argument('--data_dir', type = str, default = '',
                        help='file where results are to be written')
    parser.add_argument('--save_dir', type = str, default = '',
                        help='folder where results are to be stored')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: .1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    run(data_dir = args.data_dir,
        save_dir = args.save_dir,
        batch_size = args.batch_size,
        epochs = args.epochs,
        learning_rate = args.lr,
        log_interval = args.log_interval,
        save_model = args.save_model)
