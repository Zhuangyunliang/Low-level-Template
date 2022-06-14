import logging

import numpy as np
import torch
import time
import os

from tqdm import tqdm
from torchvision.utils import save_image
from metrics import ssim, psnr


# 模型训练
def train(opt, net, train_loader, test_loader, optimizer, criterion, epoch):
    losses = []
    ssims = []
    psnrs = []

    max_ssim = 0
    max_psnr = 0
    lr = opt.lr
    # 模型保存名称
    now_time = time.strftime("%Y%m%d", time.localtime())
    model_name = opt.datasets + '_' + opt.net + '_' + now_time + 'pth'
    opt.model_dir = os.path.join(opt.model_dir, model_name)

    # 模型训练
    net.train()
    for dark, gt in tqdm(train_loader, desc='Epoch{}'.format(epoch)):
        dark, gt = dark.to(opt.device), gt.to(opt.device)
        pre = net(dark)

        # 训练图可视化
        save_image(torch.cat([dark, pre, gt], dim=0), './preview/train.png')
        # 反向传播
        loss = criterion[0](pre, gt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    # lr 更新
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print('train loss: {:.5f} | lr: {:.7f}'.format(loss.item(), lr))

    if epoch % opt.eval_step == 0:
        with torch.no_grad():
            ssim_value, psnr_value = test(opt, net, test_loader)

        print('Epoch :{} | ssim :{:.4f} | psnr :{:.4f}'.format(epoch, ssim_value, psnr_value))

        ssims.append(ssim_value)
        psnrs.append(psnr_value)

        if ssim_value > max_ssim and psnr_value > max_psnr:
            max_ssim = max(max_ssim, ssim_value)
            max_psnr = max(max_psnr, psnr_value)

            torch.save({
                'epoch': epoch,
                'max_ssim': max_ssim,
                'max_psnr': max_psnr,
                'model': net.state_dict()
            }, opt.model_dir)

        print('model saved at step :{}| max_psnr:{:.4f} | max_ssim:{:.4f}'.format(epoch, max_psnr, max_ssim))

    np.save(f'./numpy_files/{opt.net}_{opt.steps}_losses.npy', losses)
    np.save(f'./numpy_files/{opt.net}_{opt.steps}_ssims.npy', ssims)
    np.save(f'./numpy_files/{opt.net}_{opt.steps}_psnrs.npy', psnrs)


# 模型测试
def test(opt, net, test_loader):

    ssims = []
    psnrs = []

    net.eval()
    torch.cuda.empty_cache()

    for dark, gt in test_loader:
        dark, gt = dark.to(opt.device), gt.to(opt.device)

        pre = net(dark)
        save_image(torch.cat([dark, pre, gt], dim=0), './preview/test.png')

        ssim_value = ssim(pre, gt).item()
        psnr_value = psnr(pre, gt)
        ssims.append(ssim_value)
        psnrs.append(psnr_value)

    return np.mean(ssims), np.mean(psnrs)
