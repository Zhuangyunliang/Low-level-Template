import argparse
import torch
import os
import argparse
from torch.backends import cudnn
from torch import optim
import warnings
from torch import nn
from data_utils import SID_Dataset
from torch.utils.data import DataLoader
from models.UNet import UNet
from trainer import train, test

warnings.filterwarnings('ignore')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # batch size, lr, steps, device and eval_step
    parser.add_argument('--bs', type=int, default=1, help='batch size')
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--use_multiGPU', type=list, default=[0])
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')

    parser.add_argument('--resume', type=bool, default=False)

    # 模型数据集
    parser.add_argument('--net', type=str, default='unet')
    parser.add_argument('--root_dir', type=str, default='/home/s2020317063/dataset/LOL')
    parser.add_argument('--datasets', type=str, default='lol')
    parser.add_argument('--model_dir', type=str, default='./trained_models')

    # 数据增强
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--crop_size', type=int, default=256, help='Takes effect when using --crop ')
    parser.add_argument('--no_lr_sche', action='store_true', help='no lr cos schedule')
    parser.add_argument('--perloss', action='store_false', help='perceptual loss')

    # 参数
    opt = parser.parse_args()

    # 加载训练集和测试集合
    train_loader = DataLoader(SID_Dataset(os.path.join(opt.root_dir, 'train'), train=True, size='whole_size'),
                              batch_size=opt.bs, shuffle=True)
    test_loader = DataLoader(SID_Dataset(os.path.join(opt.root_dir, 'test'), train=False, size='whole_size'),
                             batch_size=opt.bs)

    models_ = {
        'unet': UNet(n_channels=3, n_classes=3)
    }

    net = models_[opt.net]
    net = net.to(opt.device)

    # 是否多卡训练
    if len(opt.use_multiGPU) > 1:
        net = torch.nn.DataParallel(net, device_ids=opt.use_multiGPU)
        cudnn.benchmark = True

    # 定义损失函数
    criterion = [nn.L1Loss().to(opt.device)]
    # 定义优化器
    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.lr, betas=(0.9, 0.999),
                           eps=1e-08)
    optimizer.zero_grad()
    for epoch in range(opt.steps):
        train(opt, net, train_loader, test_loader, optimizer, criterion, epoch)
