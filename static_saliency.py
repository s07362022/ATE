# -*- coding: utf-8 -*-
import argparse
import logging
import sys
from pathlib import Path
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from utils.dice_score import *
from evaluate import *
from unet.unet_model import *
from torchvision import datasets, models, transforms, models, utils

# +
# 1）建立模型、加载预训练参数
model = ResUnetPlusPlus(n_channels=3, n_classes=2, bilinear=True)
checkpoint=torch.load( "/home/msoc956131/final_unet_dann_v3/checkpoints/checkpoints_1/4_min-shape-aware-sy-2-emsemble_double_NEW-AUG_us-IN_US_ResUnetPlusPlus3d-0.5-1/net1_checkpoint_epoch380.pth", map_location='cpu') #args.resume是预设的模型路径
model.load_state_dict(checkpoint)

# 2）传入数据、对图片进行预处理
src = cv2.imread('/home/msoc956131/final_unet_dann_v3/data/train/3d_leica/19-042712-1-R_125707-39112_4.png')
# src = Image.open(args.img_src).convert('RGB') #args.img_src是预设的图片路径
# data_transform = transforms.Compose([transforms.ToTensor()])
#                  transforms.Normalize((0.47,0.43, 0.39), (0.27, 0.26, 0.27))])
src = src.transpose((2, 0, 1)) # img_ndarray.shape (3, 1024, 1024)
src_tensor = src / 255
src_tensor = torch.as_tensor(src_tensor.copy()).float().contiguous()
src_tensor = torch.unsqueeze(src_tensor, dim=0) 
#这里是因为模型接受的数据维度是[B,C,H,W]，输入的只有一张图片所以需要升维
print(model.ResUnetPlusPlus_FE.residual_conv3)
# 3）指定需要计算CAM的网络结构
target_layers = [model.ResUnetPlusPlus_FE.residual_conv3] #down4()是在Net网络中__init__()方法中定义了的self.down4

# 4）实例化Grad-CAM类
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
grayscale_cam = cam(input_tensor=src_tensor, target=gt_tensor) #调用其中__call__()方法

# 5）可视化展示结果
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
grayscale_cam = cam(input_tensor=src_tensor, target=gt_tensor)

grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(src.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
plt.imshow(visualization)
plt.show()
# -


