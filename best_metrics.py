import numpy as np

ssim = np.load("./numpy_files/mit_train_net_1_20_100000_ssims.npy")
psnr = np.load("./numpy_files/mit_train_net_1_20_100000_psnrs.npy")

print('ssim: {}'.format(max(ssim)))
print('psnr: {}'.format(max(psnr)))