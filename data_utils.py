from filecmp import cmp
import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os
import numpy as np
import random
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.utils import make_grid


def tensorShow(tensors, titles=None):
        '''
        t:BCWH
        '''
        fig = plt.figure()
        for tensor, tit, i in zip(tensors, titles, range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(211+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()


class SID_Dataset(data.Dataset):
    def __init__(self, path, train, size):
        super(SID_Dataset, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        self.dark_dir = os.listdir(os.path.join(path, 'low'))
        self.dark_imgs = sorted([os.path.join(path, 'low', img) for img in self.dark_dir])
        self.gt_imgs = sorted([os.path.join(path, 'high', img) for img in self.dark_dir])
        # 保证数据集中包含元素数量相同
        assert len(self.dark_imgs) == len(self.gt_imgs), 'Datasets element is not pair'

    def __getitem__(self, index):

        dark = Image.open(self.dark_imgs[index])
        gt = Image.open(self.gt_imgs[index])

        if isinstance(self.size, int):
            assert dark.size[0] > self.size and dark.size[1] > self.size, '裁剪图片小于图像大小'

        assert dark.size == gt.size, 'dark images and gt image shape is not same'
        if not isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(dark, output_size=(self.size, self.size))
            dark = FF.crop(dark, i, j, h, w)
            gt = FF.crop(gt, i, j, h, w)
        dark, gt = self.augData(dark.convert("RGB"), gt.convert("RGB"))
        return dark, gt

    def augData(self, data, target):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = tfs.RandomHorizontalFlip(rand_hor)(data)
            target = tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data = FF.rotate(data, 90*rand_rot)
                target = FF.rotate(target, 90*rand_rot)
            data = tfs.ToTensor()(data)
            data = tfs.Resize([360, 360])(data)
            # data = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(data)
            target = tfs.ToTensor()(target)
            target = tfs.Resize([360, 360])(target)
        else:
            target = tfs.ToTensor()(target)
            data = tfs.ToTensor()(data)
        return data, target

    def __len__(self):
        return len(self.dark_imgs)

