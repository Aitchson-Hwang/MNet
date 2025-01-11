from __future__ import print_function, absolute_import

import argparse
import random

import numpy as np
import torch,time,os

torch.backends.cudnn.benchmark = True

from scripts.utils.misc import save_checkpoint, adjust_learning_rate

import scripts.datasets as datasets
import scripts.machines as machines
from options import Options
import math

# the following are using for wv
import scripts.models as archs
from scripts.models.SplitNet import *
import os
from PIL import Image
def get_sorted_image_files(image_path, extension='.png'):
    """获取指定路径下所有指定扩展名的文件，并按字典顺序排序"""
    files = [f for f in os.listdir(image_path) if f.endswith(extension)]
    files.sort()
    return files

def read_image_at_index(image_path, t, extension='.png'):
    """读取指定路径下的第t个图像文件"""
    sorted_files = get_sorted_image_files(image_path, extension)
    if t-1 < 0 or t-1 >= len(sorted_files):
        raise ValueError("Index t is out of range.")
    file_path = os.path.join(image_path, sorted_files[t-1])
    return file_path
def cosine_annealing_learning_rate(epoch, total_epochs, initial_lr, final_lr):
    """
    计算通过余弦退火方式逐渐降低学习率的函数
    Args:
        epoch: 当前轮次
        total_epochs: 总轮次
        initial_lr: 初始学习率
        final_lr: 最终学习率
    Returns:
        当前轮次的学习率
    """
    cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
    lr = (initial_lr - final_lr) * cosine_decay + final_lr
    return lr

def main(args):
    seed = 500  # 627  99(10.26) 500(2024/1/03)
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if args.data == '10kgray':
        dataset_func = datasets.COCO
    elif args.data == '10kmid':
        dataset_func = datasets.COCO
    elif args.data == '10khigh':
        dataset_func = datasets.COCO
    train_loader = torch.utils.data.DataLoader(dataset_func('train',args),batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(dataset_func('val',args),batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    print(torch.cuda.is_available())
    lr = args.lr
    data_loaders = (train_loader,val_loader)

    # watermark protection
    if args.iswv == 1:
        Machine = machines.__dict__[args.machine](datasets=data_loaders, args=args)
        print('============================ Initization Finish && Training Start =============================================')
        Machine.train()
    # watermark attack
    else:
        initial_lr = args.inlr  # 初始学习率  训练CLWD时更改为1e-4，为了防止梯度爆炸
        final_lr = args.fnlr  # 最终学习率

        Machine = machines.__dict__[args.machine](datasets=data_loaders, args=args)
        print('============================ Initization Finish && Training Start =============================================')

        for epoch in range(Machine.args.start_epoch, Machine.args.epochs):
            if args.arch == 'mnet':
                # 采用余弦退火降学习率
                lr = cosine_annealing_learning_rate(epoch, 150, initial_lr, final_lr)
                print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))
                Machine.record('lr', lr, epoch)
                Machine.train(epoch,lr)
            elif args.machine == 'vx':
                print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))
                lr = adjust_learning_rate(data_loaders, Machine.optimizer, epoch, lr, args)
                Machine.record('lr',lr, epoch)
                Machine.train(epoch)
            elif args.machine == 'denet':
                print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))
                lr = adjust_learning_rate(data_loaders, Machine.optimizer, epoch, lr, args)
                Machine.record('lr',lr, epoch)
                Machine.train(epoch)
            else:
                print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))
                lr = adjust_learning_rate(data_loaders, Machine.optimizer, epoch, lr, args)
                Machine.record('lr',lr, epoch)
                Machine.train(epoch,lr)

if __name__ == '__main__':
    parser=Options().init(argparse.ArgumentParser(description='WaterMark Removal'))
    args = parser.parse_args()
    print('==================================== WaterMark Removal =============================================')
    print('==> {:50}: {:<}'.format("Start Time",time.ctime(time.time())))
    print(torch.cuda.is_available())
    print('==> {:50}: {:<}'.format("USE GPU",os.environ['CUDA_VISIBLE_DEVICES']))
    print('==================================== Stable Parameters =============================================')
    for arg in vars(args):
        if type(getattr(args, arg)) == type([]):
            if ','.join([ str(i) for i in getattr(args, arg)]) == ','.join([ str(i) for i in parser.get_default(arg)]):
                print('==> {:50}: {:<}({:<})'.format(arg,','.join([ str(i) for i in getattr(args, arg)]),','.join([ str(i) for i in parser.get_default(arg)])))
        else:
            if getattr(args, arg) == parser.get_default(arg):
                print('==> {:50}: {:<}({:<})'.format(arg,getattr(args, arg),parser.get_default(arg)))
    print('==================================== Changed Parameters =============================================')
    for arg in vars(args):
        if type(getattr(args, arg)) == type([]):
            if ','.join([ str(i) for i in getattr(args, arg)]) != ','.join([ str(i) for i in parser.get_default(arg)]):
                print('==> {:50}: {:<}({:<})'.format(arg,','.join([ str(i) for i in getattr(args, arg)]),','.join([ str(i) for i in parser.get_default(arg)])))
        else:
            if getattr(args, arg) != parser.get_default(arg):
                print('==> {:50}: {:<}({:<})'.format(arg,getattr(args, arg),parser.get_default(arg)))
    print('==================================== Start Init Model  ===============================================')
    main(args)
    print('==================================== FINISH WITHOUT ERROR =============================================')
