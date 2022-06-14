import os
import argparse
import time
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
# batch size, lr, steps, device and eval_step
parser.add_argument('--bs', type=int, default=1, help='batch size')
parser.add_argument('--steps', type=int, default=100000)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--use_multiGPU', type=str, default=[0])
parser.add_argument('--eval_step', type=int, default=50)
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')

parser.add_argument('--resume', type=bool, default=False)

# 模型数据集
parser.add_argument('--net', type=str, default='net')
parser.add_argument('--testset', type=str, default='lol_test')
parser.add_argument('--trainset', type=str, default='lol_train')
parser.add_argument('--model_dir', type=str, default='./trained_models/')

# 数据增强
parser.add_argument('--crop', action='store_true')
parser.add_argument('--crop_size', type=int, default=256, help='Takes effect when using --crop ')
parser.add_argument('--no_lr_sche', action='store_true', help='no lr cos schedule')
parser.add_argument('--perloss', action='store_false', help='perceptual loss')


opt = parser.parse_args()

# 模型保存名称 数据集名_模型名称_日期
now_time = time.strftime("%Y%m%d", time.localtime())
model_name = opt.trainset.split('_')[0] + '_' + opt.net.split('.')[0] + '_' + now_time
opt.model_dir = opt.model_dir + model_name + '.pth'

log_dir = 'logs/' + model_name

print(opt)
print('model_dir:', opt.model_dir)


if not os.path.exists('trained_models'):
	os.mkdir('trained_models')
if not os.path.exists('numpy_files'):
	os.mkdir('numpy_files')
if not os.path.exists('logs'):
	os.mkdir('logs')
if not os.path.exists('samples'):
	os.mkdir('samples')
if not os.path.exists(f"samples/{model_name}"):
	os.mkdir(f'samples/{model_name}')
if not os.path.exists(log_dir):
	os.mkdir(log_dir)
