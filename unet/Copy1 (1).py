# -*- coding: utf-8 -*-
""" Full assembly of the parts to form the complete network """

# +
import torch
import torch.nn as nn
indim = 3
padding = 0
radius = 1
outdim = 8
band_width = 1
temperature = 0.01
drop_rate = 1
x_old = torch.Tensor(torch.randn(1,3,512,512))
# print(x_old)
conv_info = nn.Conv2d(indim, indim * (radius * 2 + 1)**2, 
                                   kernel_size=radius * 2 + 1,
                                   padding=radius + padding, groups=indim, bias=False)
all_one_conv_indim_wise = nn.Conv2d(indim * (radius * 2 + 1)**2, (radius * 2 + 1)**2,
                                                 kernel_size=3, stride=1, padding=1, dilation=1,
                                                 groups=(radius * 2 + 1)**2, bias=False)

kernel = torch.zeros(((radius * 2 + 1)**2, 1, radius * 2 + 1, radius * 2 + 1), dtype=torch.float)
kernel[:, :, radius, radius] += 1
for i in range(2 * radius + 1):
    for j in range(2 * radius + 1):
        kernel[i * (2 * radius + 1) + j, 0, i, j] -= 1
# print('------------',kernel.shape)
kernel = kernel.repeat(indim, 1, 1, 1)
print('------------0',kernel.size())
# kernel.shape
conv_info.weight.data = kernel
# print('------------0',kernel)
distance = conv_info(x_old)
print('------------1',distance.size())
batch_size, _, h_dis, w_dis = distance.shape
# print('------------2',distance.size())
# print(distance)
distance = distance.reshape(batch_size, indim, (radius * 2 + 1)**2,
                            h_dis, w_dis).transpose(1, 2).\
reshape(batch_size, (radius * 2 + 1)**2 * indim, h_dis, w_dis)
# print('++++++++++++++++++',distance)
print('------------3',distance.size())
all_one_conv_indim_wise.weight.data = torch.ones_like(all_one_conv_indim_wise.weight, dtype=torch.float)
print('all_one_conv_indim_wise',all_one_conv_indim_wise.weight.data.size())
# print('all_one_conv_indim_wise',all_one_conv_indim_wise.weight.data)
# print((distance**2).size())
distance = all_one_conv_indim_wise(distance**2)
print('------------4',distance.size())
all_one_conv_radius_wise = nn.Conv2d((radius * 2 + 1)**2, outdim, kernel_size=1, padding=0, bias=False)
all_one_conv_radius_wise.weight.data = torch.ones_like(all_one_conv_radius_wise.weight, dtype=torch.float)
print('all_one_conv_radius_wise',all_one_conv_radius_wise.weight.data.size())
# print('all_one_conv_radius_wise',all_one_conv_radius_wise.weight.data)
# print('distance',distance)
distance = torch.exp(-distance / distance.mean() / 2 / band_width**2) # using mean of distance to normalize
# print('distance',distance)
print('distance.mean()',distance.mean())
print('------------5',distance.size())
prob = (all_one_conv_radius_wise(distance) / (2 * radius + 1)**2) ** (1 / temperature)
print(all_one_conv_radius_wise(distance).size())
print(prob.size())
# print(prob)
prob /= prob.sum(dim=(-2, -1), keepdim=True)
print('prob',prob.size())
# print('prob',prob)
print('x_old.shape',x_old.shape)
print('x_old.size()',x_old.size())
batch_size, channels, h, w = x_old.shape
# print(batch_size * channels)
print('(prob.view(batch_size * channels, -1).size()',(prob.view(batch_size * channels, -1)).size())
# print('(prob.view(batch_size * channels, -1)',(prob.view(batch_size * channels, -1)))
random_choice = torch.multinomial((prob.view(batch_size * channels, -1) + 1e-8), drop_rate * h * w, replacement=True)
print('random_choice',random_choice.size())
print(random_choice)
# print(conv_info.weight.data.size())

random_mask = torch.ones((batch_size * channels, h * w))
# print(random_mask)
random_mask[torch.arange(batch_size * channels).view(-1, 1), random_choice] = 0
# print(random_mask)
random_mask.view(x_old.shape)
# -
import torch.nn as nn
import torch
m = nn.Dropout(p=0.5)
input_ = torch.randn(1,2,4, 4)
output = m(input_)
output


# +
# from .unet_parts import *
from torch.autograd import Function
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# -

import torch.nn as nn
# import torch
import torch.nn.functional as F
import wandb
import numpy as np
# from torch import *
# from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

np.load('/home/msoc956131/final_unet_dann_v1/data/train/dann/17-022565-2_66350-7656_1.npy')

# +
import torch
#1
x = torch.Tensor([[1, 0, 0]])
# true = torch.Tensor([[1],[0]])
true = torch.Tensor(torch.randn(2))
# true = true.resize(true.size()[0],1,1)
true = true.repeat_interleave(36)
print('true :', true.size())
true= true.reshape(-1)
print('true :', true)
# true = torch.ones([1])
# true = 
print('x :', x)
pre = torch.Tensor(torch.randn(1,3,2,2))
pre_ = pre.permute(0,2,3,1)
print('pre:', pre)
# print('pre[0]:', pre[0])
# print('pre_[0][0]:', pre_[0][0][0])
pre_ = pre_.reshape(-1,3)
print('pre_:', pre_[0])
# print('x1:', x1.size())

# x1 = x1.expand(1,6,6,3)
# print('x2 :', x2[0])
# x2 = x1.reshape(-1,3)
# print('x1 :',x1)
# print('x2 :', x2[0])

# -

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        
        #=========================dann
        self.h_domain_classifier = nn.Sequential()
        self.h_domain_classifier.add_module('d_fc1', nn.Linear(1024 // factor * 64 * 64, 100))
        self.h_domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.h_domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.h_domain_classifier.add_module('d_fc2', nn.Linear(100, 3))
        self.h_domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))
        
        self.s_domain_classifier = nn.Sequential()
        self.s_domain_classifier.add_module('d_fc1', nn.Linear(1024 // factor * 64 * 64, 100))
        self.s_domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.s_domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.s_domain_classifier.add_module('d_fc2', nn.Linear(100, 3))
        self.s_domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))
        
        self.l_domain_classifier = nn.Sequential()
        self.l_domain_classifier.add_module('d_fc1', nn.Linear(1024 // factor * 64 * 64, 100))
        self.l_domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.l_domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.l_domain_classifier.add_module('d_fc2', nn.Linear(100, 3))
        self.l_domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))
#         self.domain_classifier.add_module('d_softmax', nn.Sigmoid())

    def forward(self, x, alpha):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        #=========================dann==============================
        feature = x5.view(-1, 1024 // factor * 64 * 64)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        h_domain_output = self.h_domain_classifier(reverse_feature)
        s_domain_output = self.s_domain_classifier(reverse_feature)
        l_domain_output = self.l_domain_classifier(reverse_feature)
        #=========================dann==============================
        
        return logits, h_domain_output, s_domain_output, l_domain_output


# +
class ReverseLayerF(Function):

#     @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

#     @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


# -

class __AbstractDataset(object):
    """Abstract class for interface of subsequent classes.
    Main idea is to encapsulate how each dataset should parse
    their images and annotations.
    
    """

    def load_img(self, path):
        raise NotImplementedError

    def load_ann(self, path, with_type=False):
        raise NotImplementedError
class __CoNSeP(__AbstractDataset):
    """Defines the CoNSeP dataset as originally introduced in:

    Graham, Simon, Quoc Dang Vu, Shan E. Ahmed Raza, Ayesha Azam, Yee Wah Tsang, Jin Tae Kwak, 
    and Nasir Rajpoot. "Hover-Net: Simultaneous segmentation and classification of nuclei in 
    multi-tissue histology images." Medical Image Analysis 58 (2019): 101563
    
    """

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        # assumes that ann is HxW
        ann_inst = sio.loadmat(path)["inst_map"]
        if with_type:
            ann_type = sio.loadmat(path)["type_map"]

            # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
            # If own dataset is used, then the below may need to be modified
            ann_type[(ann_type == 3) | (ann_type == 4)] = 3
            ann_type[(ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 4

            ann = np.dstack([ann_inst, ann_type])
            ann = ann.astype("int32")
        else:
            ann = np.expand_dims(ann_inst, -1)
            ann = ann.astype("int32")

        return ann
####
class __3d_hamamasu(__AbstractDataset):
    """Defines the CPM 2017 dataset as originally introduced in:

    Vu, Quoc Dang, Simon Graham, Tahsin Kurc, Minh Nguyen Nhat To, Muhammad Shaban, 
    Talha Qaiser, Navid Alemi Koohbanani et al. "Methods for segmentation and classification 
    of digital microscopy tissue images." Frontiers in bioengineering and biotechnology 7 (2019).

    """

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        assert not with_type, "Not support"
        # assumes that ann is HxW
        ann_inst = sio.loadmat(path)["inst_map"]
        ann_inst = ann_inst.astype("int32")
        ann = np.expand_dims(ann_inst, -1)
        return ann
def get_dataset(name):
    """Return a pre-defined dataset object associated with `name`."""
    name_dict = {
        "kumar": lambda: __Kumar(),
        "cpm17": lambda: __CPM17(),
        "consep": lambda: __CoNSeP(),
        "3d_hamamasu": lambda: __3d_hamamasu(),
    }
    if name.lower() in name_dict:
        print(name.lower())
        return name_dict[name]()
    else:
        assert False, "Unknown dataset `%s`" % name


# +
dataset_name = "3d_hamamasu"
save_root = "dataset/training_data/%s/" % dataset_name

dataset_info = {
        "train": {
            "img": (".png", "dataset/CoNSeP/Train/Images/"),
            "ann": (".mat", "dataset/CoNSeP/Train/Labels/"),
        },
        "valid": {
            "img": (".png", "dataset/CoNSeP/Test/Images/"),
            "ann": (".mat", "dataset/CoNSeP/Test/Labels/"),
        },
    }
parser = get_dataset(dataset_name)
# for split_name, split_desc in dataset_info.items():
#     img_ext, img_dir = split_desc["img"]
#     ann_ext, ann_dir = split_desc["ann"]
# #     print(split_name)
# #     print(split_desc)
    print(img_ext)
    print(img_dir)
    print(ann_ext)
    print(ann_dir)

# +
import torch

x = torch.Tensor(torch.randn(2,1,2,2))
y = torch.mean(x,dim=0)   
print(x)
y
# y = np.array([2,1,2,3])
# print(x)
# y = np.expand_dims(x,axis=-1)
# print(y)
# print ("y.shape: ",y.shape)


# +
import torch

import torch.nn.functional as F
x = torch.tensor(torch.randn(2,2,4,4), dtype=torch.float)
x =torch.nn.Softmax(dim=0)(x)
def dice_loss(pred, true, smooth=1e-7):
    iflat = pred.view(-1)
    tflat = true.view(-1)
    intersection = (iflat * tflat).sum()
    dice_coeff = (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)

    return 1 - dice_coeff

def loss_np(true, pred):
    loss = F.binary_cross_entropy(pred, true) + dice_loss(pred, true)

    return loss
np_loss = loss_np(x, x)


# +
# net = UNet(n_channels=3, n_classes=2, bilinear=True)
# up_sum = 0
# n_layer = 0
# m = nn.AdaptiveAvgPool2d((1,1))
# input = torch.randn(1, 64, 3, 3)
# output = nn.AdaptiveAvgPool2d((1,1))(input)
# output.size()
# x = torch.flatten(output, 1)
# x.size()
def js_div(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
    log_mean_output = ((p_output + q_output )/2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2

# loss2 = nn.CrossEntropyLoss()

x = torch.tensor(torch.randn(2.2,4,4), dtype=torch.float)
y = torch.tensor(torch.randn(2,4), dtype=torch.float)
x = torch.flatten(x)
print(x)
output = torch.nn.Softmax(dim=0)(x)
output1 = torch.nn.Softmax(x)
# output1 = torch.nn.Softmax(dim=1)(x)
print('0000000000',output)
print('1111111111',output1)

# -

loss = nn.CrossEntropyLoss()
input_ = torch.randn(2,3, dtype=torch.float)
target = torch.empty(2,dtype=torch.long).random_(3)
print(target)
output = loss(input_, target)
output

 loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
output = loss(input, target)
output

# +
from torch import optim
net = UNet(n_channels=3, n_classes=2, bilinear=True)
optimizer = optim.RMSprop([
                {'params': net.down1.parameters()},
                {'params': net.down2.parameters()},
                {'params': net.down3.parameters()},
                {'params': net.down4.parameters()},
            ], lr=0.01, weight_decay=1e-8, momentum=0.9)


# print(net.h_domain_classifier.parameters())
# print(net.down2.parameters())
# print(net.up1.conv.double_conv[0].weight[0,0::])
# print(net.h_domain_classifier[0].weight) 
# -

# print(net.up1.conv.double_conv[0].weight[0,0])
# print(net.down1.maxpool_conv[1].double_conv[0].weight[0,0])
# print(net.h_domain_classifier[0].weight[0] )
# print(6%6 )
batch_size
for  layer in net.modules():
    if isinstance(layer, torch.nn.Conv2d):
        print('==========',layer)
        print(layer.weight.grad)

print(net.down1.maxpool_conv[1].double_conv[0].weight.grad) 
for  layer in net.modules():
    if isinstance(layer, torch.nn.Conv2d):
        print('=====',layer)
        print(layer.weight)

# +
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
#Prepare the data
digits = datasets.load_digits(n_class=6)
X, y = digits.data, digits.target
print(X.shape)
print('============',y.shape)
# n_samples, n_features = X.shape
# n = 20  
# img = np.zeros((10 * n, 10 * n))
# for i in range(n):
#     ix = 10 * i + 1
#     for j in range(n):
#         iy = 10 * j + 1
#         img[ix:ix + 8, iy:iy + 8] = X[i * n + j].reshape((8, 8))
# plt.figure(figsize=(8, 8))
# plt.imshow(img, cmap=plt.cm.binary)
# plt.xticks([])
# plt.yticks([])
# plt.show()

#t-SNE
X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(X)

#Data Visualization
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize
plt.figure(figsize=(8, 8))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.show()
# -

import torch
import torch.nn as nn
from unet_model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AttentionUNet(n_channels=3, n_classes=2, bilinear=True)
model.load_state_dict(torch.load("/home/msoc956131/final_unet_dann_v1/checkpoints_jsdiv_nres_AttentionUNet_gan_lay1-11_ce_3d_hamamasu-leica_lr-0.0001-0-0-0-1/checkpoint_epoch200.pth"))
model.to(device)

import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from torchvision import datasets, models, transforms, models, utils
from albumentations.pytorch.transforms import ToTensorV2
transform = A.Compose(
    [
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
        A.RGBShift(p=1),
#         A.RandomCrop(always_apply=True, p=1.0, height=512, width=512),
#         A.ShiftScaleRotate(p=0.5),
#         ToTensorV2(transpose_mask=True)
    ],
    additional_targets={'label': 'image'}
)
to_tensor = A.Compose([ToTensorV2(transpose_mask=True)],additional_targets={'label': 'image'})
class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, dann_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        #===============================dann==============================================
        self.dann_dir = Path(dann_dir)
        #===============================dann==============================================
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
#         print('==============',w,h)
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
#         elif not is_mask:
#             img_ndarray = img_ndarray.transpose((2, 0, 1))
        img_ndarray = img_ndarray.transpose((2, 0, 1)) # img_ndarray.shape (3, 1024, 1024)
#         print('img_ndarray.shape',img_ndarray.shape)
#         if not is_mask:
#             img_ndarray = img_ndarray / 255
        img_ndarray = img_ndarray / 255
        if not is_mask:
            return img_ndarray
        if is_mask:
#             print('img_ndarray[0].shape',img_ndarray[0].shape)
            return img_ndarray[0]

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]  # 1002095_2
#         print('name',name)
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        #===============================dann=========================================
        dann_mask_file = list(self.dann_dir.glob(name + '.*'))
#         print('mask_file',mask_file)
#         print('len(mask_file)',len(mask_file))
        #===============================dann=========================================
#         print('dann_h_mask_file',dann_h_mask_file)
        
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        #=================================dann=================================================
        assert len(dann_mask_file) == 1, f'Either no image or multiple images found for the ID {name}: {dann_mask_file}'
        #=================================dann=================================================
        
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])
        #===============================dann=======================================
        dann = np.load(dann_mask_file[0])
        dann = torch.tensor(dann, dtype=torch.int64)
#         print('dann.type()',dann.type())
        #===============================dann=======================================
#         print('dann_h',dann_h)
        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        #-----------------------------------causel-----------------------------------
        img_trsp = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR) # 1024*1024*3
        mask_trsp = cv2.cvtColor(np.asarray(mask), cv2.COLOR_RGB2BGR)
        
        mask_trsp_binary = mask_trsp /255 
        mask_trsp_binary_bg = 1-mask_trsp_binary
        img_nuc = mask_trsp_binary* img_trsp
#         print('mask_trsp_binary.max()',mask_trsp_binary.max())
#         print('mask_trsp_binary.min()',mask_trsp_binary.min())
        
        transformed1_array = transform(image=img_trsp, label=mask_trsp_binary)
        transformed2_array = transform(image=img_trsp, label=mask_trsp_binary)
        transformed3_array = transform(image=img_trsp, label=mask_trsp_binary)
#         print('transformed1_array',transformed1_array)
#         print('transformed1_array["label"].max()',transformed1_array["label"].max())
#         print('transformed1_array["label"].min()',transformed1_array["label"].min())
#         print('transformed1["image"].shape',transformed1["image"].shape)
        img_bg1 = mask_trsp_binary_bg* transformed1_array["image"]
        img_bg2 = mask_trsp_binary_bg* transformed2_array["image"]
        img_bg3 = mask_trsp_binary_bg* transformed3_array["image"]
        img_mix1 = img_bg1 + img_nuc
        img_mix2 = img_bg2 + img_nuc
        img_mix3 = img_bg3 + img_nuc
        
        #=====================1024*1024*3 => 3*1024*1024
        to_tensor_array1 = to_tensor(image=img_mix1, label=mask_trsp) 
        to_tensor_array2 = to_tensor(image=img_mix2, label=mask_trsp)
        to_tensor_array3 = to_tensor(image=img_mix3, label=mask_trsp)
#         cv2.imwrite("./11111/x"+name+".png",img_mix1)
#         cv2.imwrite("./11111/y"+name+".png",img_mix2)
#         cv2.imwrite("./11111/z"+name+".png",img_mix3)
#         print('to_tensor_array["image"].size()',to_tensor_array["image"].size())
        
        img_trsf1 = to_tensor_array1["image"] # 3*1024*1024
        img_trsf2 = to_tensor_array2["image"] # 3*1024*1024
        img_trsf3 = to_tensor_array3["image"] # 3*1024*1024
#         print('=======img_trsf.shape',img_trsf.shape)
        #-----------------------------------causel-----------------------------------

        
        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)
#         print('img.shape',img.shape)
#         print('img.shape',img.shape)
#         print('mask.shape',mask.shape)
        

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'img_trsf1': torch.as_tensor(img_trsf1).float().contiguous(),
            'img_trsf2': torch.as_tensor(img_trsf2).float().contiguous(),
            'img_trsf3': torch.as_tensor(img_trsf3).float().contiguous(),
            #===============================dann==============================================
            'dann':  torch.as_tensor(dann).contiguous(),
            #===============================dann==============================================
        }
class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, dann_dir, scale=1):
        super().__init__(images_dir, masks_dir, dann_dir, scale, mask_suffix='_nuc_filled')    


# +
# import sys

# print(sys.path)
# import ..utlis
# import evaluate  
# from unet.unet_model import *
import argparse
import logging
import sys
from pathlib import Path
img_scale =1
dir_all_img = Path('../data/train/3d_leica_hamamasu')
dir_all_mask = Path('../data/train/mask')
all_dann_dir = Path('../data/train/dann')
loader_args = dict(batch_size=36, num_workers=4, pin_memory=True)
all_set = CarvanaDataset(dir_all_img,dir_all_mask,all_dann_dir,img_scale)
all_loader = DataLoader(all_set, shuffle=False, drop_last=False, **loader_args)      

predict_all =  torch.zeros(0).to(device)
feature_all =  torch.zeros(0).to(device)

for idx, batch in tqdm(enumerate(all_loader), total = len(all_loader)):
    with torch.cuda.amp.autocast(enabled = True):
        with torch.no_grad():
            image = batch['image']
            dann_true = batch['dann']
            image = image.to(device, dtype=torch.float32)
            mask_pred, dann, x5 = model(image)
            _x5 = x5.view(x5.size(0), -1).to(device)
            _dann_true = dann_true.view(x5.size(0), -1).to(device)
#             _i = 0
#             for _i in range(x5.size(0)):
#                 x5i = x5[_i].view(1, -1)
#                 _dan=dann_true[_i].view(1, -1)
# #                 print(_i)
# #                 print(dann_true[_i])
#                 if _i==0:
#                     predict = _dan
#                     feature = x5i
# #                     predict = torch.unsqueeze(_dan,0)
# #                     feature = torch.unsqueeze(x5i,0)
# #                     print('-----------',predict)
# #                     print(feature)
# #                 x5i = x5i.cpu().numpy()
# #                     print(' =======predict.shape', predict.size())
# #                     print(' =======feature.shape', feature.size())
#                 else:
#                     predict=torch.cat((predict,_dan),0)
#                     feature=torch.cat((feature,x5i),0)
            predict_all = torch.cat((predict_all,_dann_true),0)
            feature_all =torch.cat((feature_all,_x5),0)
print(' len(predict)', predict_all.size())
print(' len(feature)', feature_all.size())
# print(' len(predict)', predict.size())
# print(' len(feature)', feature.size())
# df_data = {
#     'predict': predict,
#     'feature': feature,
# }
# df = pd.DataFrame(df_data)

# +
import wandb
import torch
import pandas as pd
train_x =torch.randn((20, 104800)).cpu().numpy().tolist()
train_label =torch.randn((20)).cpu().numpy().tolist()
# print('predict_all.reshape(-1)',predict_all.reshape(-1).size())
# train_x =feature_all.cpu().numpy().tolist()
# train_label =predict_all.reshape(-1).cpu().numpy().tolist()
# print(' len(predict)', len(a))
# print(' len(feature)',len(b))
# wandb.init(project="embedding_tutorial")
# embeddings = [
#     # D1   D2   D3   D4   D5
#     [0.2, 0.4, 0.1, 0.7, 0.5], # embedding 1
#     [0.3, 0.1, 0.9, 0.2, 0.7], # embedding 2
#     [0.4, 0.5, 0.2, 0.2, 0.1], # embedding 3
# ]
# wandb.log({
#     "test": wandb.Table(
#         columns = a, 
#         data    = b
# #         columns = ["D1", "D2", "D3", "D4", "D5"], 
# #         data    = embeddings
#     )
# })
# wandb.finish()
import wandb
# from sklearn.datasets import load_digits
df_data  = {
            'predict': train_label,
            'feature': train_x,
                        }
df_ = pd.DataFrame(df_data)
wandb.init(project="embedding_tutorial")

# # Load the dataset
# ds = load_digits(as_frame=True)
# df = ds.data


# # Create a "target" column
# df["target"] = ds.target.astype(str)
# print('df["target"]',df["target"])
# cols = df.columns.tolist()
# df = df[cols[-1:] + cols[:-1]]

# # Create an "image" column
# df["image"] = df.apply(lambda row: wandb.Image(row[1:].values.reshape(8, 8) / 16.0), axis=1)
# cols = df.columns.tolist()
# df = df[cols[-1:] + cols[:-1]]

wandb.log({"digits": df_})
wandb.finish()
# -

wandb.finish()

#sparse 50
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
#Prepare the data
# digits = datasets.load_digits(n_class=6)
X, y = feature_all.cpu(),predict_all.cpu()
X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(X)
y=y.reshape(-1)
df = pandas.DataFrame(dict(Feature_1=X_tsne[:,0], Feature_2=X_tsne[:,1], label=y))
df.plot(x="Feature_1", y="Feature_2", kind='scatter', c='label', colormap='viridis')

#sparce 1
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
#Prepare the data
# digits = datasets.load_digits(n_class=6)
X, y = feature_all.cpu(),predict_all.cpu()
X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(X)
y=y.reshape(-1)
df = pandas.DataFrame(dict(Feature_1=X_tsne[:,0], Feature_2=X_tsne[:,1], label=y))
df.plot(x="Feature_1", y="Feature_2", kind='scatter', c='label', colormap='viridis')

#ori 200
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
#Prepare the data
# digits = datasets.load_digits(n_class=6)
X, y = feature_all.cpu(),predict_all.cpu()
X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(X)
y=y.reshape(-1)
df = pandas.DataFrame(dict(Feature_1=X_tsne[:,0], Feature_2=X_tsne[:,1], label=y))
df.plot(x="Feature_1", y="Feature_2", kind='scatter', c='label', colormap='viridis')

#causel 200
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
#Prepare the data
# digits = datasets.load_digits(n_class=6)
X, y = feature_all.cpu(),predict_all.cpu()
X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(X)
y=y.reshape(-1)
df = pandas.DataFrame(dict(Feature_1=X_tsne[:,0], Feature_2=X_tsne[:,1], label=y))
df.plot(x="Feature_1", y="Feature_2", kind='scatter', c='label', colormap='viridis')

#jsd 200
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
#Prepare the data
# digits = datasets.load_digits(n_class=6)
X, y = feature_all.cpu(),predict_all.cpu()
X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(X)
y=y.reshape(-1)
df = pandas.DataFrame(dict(Feature_1=X_tsne[:,0], Feature_2=X_tsne[:,1], label=y))
df.plot(x="Feature_1", y="Feature_2", kind='scatter', c='label', colormap='viridis')

# +

import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
#Prepare the data
# digits = datasets.load_digits(n_class=6)
X, y = feature_all.cpu(),predict_all.cpu()
X = X.permute(1, 0)
print(X.size())
print(y.size())

# +
inputs = [[1, 2 ,3], [4, 5, 6]]
         

inputs = torch.tensor(inputs)
print(inputs)
print('Inputs:', inputs.shape)


outputs = inputs.permute(1, 0)
print(outputs)
print('Outputs:', outputs.shape)

# +
# a = torch.randn(2, 2)
# b = X.cpu().numpy().tolist()
# a=predict_all.reshape(-1)
# print(a.shape)
a = a.cpu().numpy().astype(int).tolist()

print(len(b))


# +
print(len(a))
# unet = UNet(3,2)
# inp = torch.rand(2,3,3)
# import torch
# predict =[]
# x5 = torch.rand(5,1,1,1)
# x6 = torch.stack((x5,x5),dim = 0)
# x6.size()
# print((x5.size(0)))
# x = torch.rand(4,2,3,3)
# x5 = torch.flatten(x5)
# batch_size =5
# for i in range(batch_size):
    
# #     x5_ = np.array_split(x5, 5)
#     predict.append(x5[i])
# # x5_ = np.array_split(x5, 5)
# print(len(predict))
# # print(len(predict))
# print(predict)
# print(x5_)
# imp = torch.cat((inp, inp, inp), 0)
# imp.size()
# inp[0:1]
# inp[0]
# def dice_loss(pred, true, smooth=1e-7):
#         iflat = pred.view(-1)
#         tflat = true.view(-1)
#         intersection = (iflat * tflat).sum()
#         dice_coeff = (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)

#         return 1 - dice_coeff
# def loss_np(true, pred):
#         loss = F.binary_cross_entropy(pred, true) + dice_loss(pred, true)

#         return loss
# input_ = torch.randn((2,3))
# target = torch.randn((2))
# target = torch.tensor(target, dtype=torch.int64)
# target = target.to(torch.long)
# print(input_)
# print(target)
# # print('target.type()',target.type())
# print('target.type()',target.type())
# loss = torch.nn.CrossEntropyLoss()(input_, target)
# loss
# -

train_x =torch.randn((2,3))
train_y =torch.randn((2))
train_x=torch.tensor(train_x,dtype=torch.float32).cuda()
train_y=torch.tensor(train_y,dtype=torch.int64).cuda()
# train_x = Variable(train_x)
# train_y = Variable(train_y)
loss = torch.nn.CrossEntropyLoss()(train_x, train_y)
loss

ss = inp[0].view(-1,1)
ss.size()

gt = torch.rand(1,2,4,4)
gt

gtb = torch.rand(1,2,4,4)
print(gtb[:, 1, ...])
gtb[:, 1, ...].size()

# +
xz = torch.softmax(gt, dim=1)

xz
# +
import torch
from scipy import spatial
import numpy as np

a = torch.randn(2, 2)
b = torch.randn(2, 2) # different row number, for the fun
print(a)
print(b)
# Given that cos_sim(u, v) = dot(u, v) / (norm(u) * norm(v))
#                          = dot(u / norm(u), v / norm(v))
# We fist normalize the rows, before computing their dot products via transposition:
a_norm = a / a.norm(dim=1)[:, None]
b_norm = b / b.norm(dim=1)[:, None]
# res = torch.mm(a_norm, b_norm.transpose(0,1))
# print(res)
# #  0.9978 -0.9986 -0.9985
# # -0.8629  0.9172  0.9172

# # -------
# # Let's verify with numpy/scipy if our computations are correct:
# a_n = a.numpy()
# b_n = b.numpy()
# res_n = np.zeros((2, 3))
# for i in range(2):
#     for j in range(3):
#         # cos_sim(u, v) = 1 - cos_dist(u, v)
#         res_n[i, j] = 1 - spatial.distance.cosine(a_n[i], b_n[j])
# print(res_n)
# [[ 0.9978022  -0.99855876 -0.99854881]
#  [-0.86285472  0.91716063  0.9172349 ]]
# -

gtb = torch.rand(1,2,3,3)
# input2 = torch.rand(1,2,3,3)
input2 = torch.nn.functional.normalize(gtb )
print(input1)
print(input2)
# output = F.cosine_similarity(input1, input2,dim=-2)
# print(output)

# +
from tqdm import tqdm

# assume
dataset_num = 100000
batch_size = 8

epoch = 1
d_loss = 1.2345
g_loss = 1.2345

pbar = tqdm(range(int(dataset_num / batch_size)))

for i in pbar:
    epoch += 1
    d_loss += 1.2345
    g_loss += 1.2345
    pbar.set_description('Epoch: %i' % epoch)
    pbar.set_postfix(d_loss=format(d_loss,'.3f'), g_loss=format(g_loss,'.3f'))
# -

# With Learnable Parameters
m = nn.BatchNorm2d(100)
# Without Learnable Parameters
m = nn.BatchNorm2d(100, affine=False)
input = torch.randn(20, 100, 35, 45)
output = m(input)
output.size()


xzzz = torch.softmax(gt, dim=1).argmax(dim=1)
xzzz

gtb = torch.rand(2,4,2)
xzzzz = gtb.argmax(dim=0)
print(xzzzz)
xzzzz.size()

# +
input2 = torch.nn.functional.normalize(xzzzz.float() )
t_re_layer =  torch.transpose(input2,0,1)
matmul_layer =  torch.matmul(input2,t_re_layer)
# output = F.cosine_similarity(input2, input2)
output = torch.triu(matmul_layer, diagonal=1)
sd = torch.sum(output)
print(xzzzz.float())
print(input2)
print(t_re_layer)

print(matmul_layer)
print(output)
print(sd)
# -

gt = torch.rand(1,2,3,3)
xzzz = gt.argmax(dim=1)
print(xzzz)
print(1-xzzz)
# print(xzzz)
xzzz.size()


# +
from sklearn.metrics import f1_score
from torchmetrics import F1Score
print(xzzz*xzzzz)
print(xzzz&xzzzz)
# f1_score(xzzz[0],xzzzz[0], average=None)

# F1Score(num_classes=2)(xzzz,xzzzz)
# -

xzzz = gt.argmax(dim=0)
xzzz

# +
xz_ = torch.softmax(gt, dim=1).argmax(dim=1)

# print('xz_[0]',xz_[0])
xz_
# -

print(inp[:, 1:, ...].shape)
print(gt[:, 1:, ...].shape)

t = F.one_hot(inp.argmax(dim=1), 2).permute(0, 3, 1, 2).float()
gtt = F.one_hot(gt.argmax(dim=1), 2).permute(0, 3, 1, 2).float()
# t =(F.sigmoid(inp) > 0.5).float()
print(t)
print(gtt)
print(gtt[:, 1:, ...] * t[:, 1:, ...])

t.int()
gtt.int()
intersection = (gtt[:, 1:, ...] * t[:, 1:, ...]).sum()
union =  gtt[:, 1:, ...].sum() + t[:, 1:, ...].sum() - intersection
iou = (intersection ) / (union)
print(gtt[:, 1:, ...].sum(),t[:, 1:, ...].sum())
print(intersection,union,iou)

# gtt
(gtt[:, 1:, ...] )&(t[:, 1:, ...])

inp[:, 1:, ...].shape

out = unet(inp, 0)
sss = out1.size()
sss

out1= unet.inc(inp)
s = out1.size()
s

out2= unet.down1(out1)
s2 = out2.size()
s2

out3= unet.down2(out2)
s3 = out3.size()
s3

out4= unet.down3(out3)
s4 = out4.size()
s4

out5= unet.down4(out4)
s5 = out5.size()
s5

feature = out5.view(-1, 1024 // 2 * 64 * 64)
feature.size()

out6= unet.s_domain_classifier(feature)
s6 = out6.size()
s6

inp = torch.rand(1,2,3,3)
print(inp)
ii = torch.softmax(inp, dim=1)
ii

ii.argmax()

ii.argmax(dim=1)



ii.argmax(dim=1)[0]

# +
import torch
import torch.nn as nn

a = torch.tensor([[0.3045, 0.4094, 0.2861],
        [0.4303, 0.2471, 0.3226],
        [0.5010, 0.2288, 0.2702],
        [0.2487, 0.4238, 0.3274],
        [0.4503, 0.1850, 0.3647],
        [0.2684, 0.5116, 0.2201],
        [0.2696, 0.1981, 0.5324],
        [0.4101, 0.4435, 0.1463]]).float()
a_t = torch.transpose(a, 0, 1)
b = torch.tensor([[1., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0., 1., 0.],
        [1., 0., 0.]]).float()
b_ = torch.tensor([0,
        0,
        1,
        0,
        1,
        2,
        2,
        0])
# b_t = torch.transpose(b , 0, 1)  
print('a.size()',a.size())
print('b.size()',b.size())
print(a_t)
print(nn.BCEWithLogitsLoss()(a, b))
print(nn.CrossEntropyLoss()(a, b_))
# -

loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()
print('input',input)
print('target',target)

x_input =torch.tensor([[ 2.8883,  0.1760,  1.0774],
        [ 1.1216, -0.0562,  0.0660],
        [-1.3939, -0.0967,  0.5853]])
y_target =torch.tensor([1, 2, 0])
crossentropyloss=nn.CrossEntropyLoss()
crossentropyloss_output=crossentropyloss(x_input,y_target)
crossentropyloss_output

# +
import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        # number of input channels is a number of filters in the previous layer
        # number of output channels is a number of filters in the current layer
        # "same" convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out
class AttentionUNet_FE(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True):
        super(AttentionUNet_FE, self).__init__()
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(n_channels, 64)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024)
        

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)

        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)
        return e1, e2, e3, e4, e5
class AttentionUNet_DE(nn.Module):
    def __init__(self, n_classes=2, bilinear=True):
        super(AttentionUNet_DE, self).__init__()
        
        self.Up5 = UpConv(1024, 512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        self.UpConv5 = ConvBlock(1024, 512)

        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv4 = ConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ConvBlock(128, 64)

        self.Conv = nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, e1, e2, e3, e4, e5):
        d5 = self.Up5(e5)

        s4 = self.Att5(gate=d5, skip_connection=e4)
        d5 = torch.cat((s4, d5), dim=1) # concatenate attention-weighted skip connection with previous layer output
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv(d2)
        
        return out
class AttentionUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True):
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.AttentionUNet_FE = AttentionUNet_FE(self.n_channels, self.n_classes, self.bilinear)
        self.AttentionUNet_DE = AttentionUNet_DE(self.n_classes)
        self.AttentionUNet_DANN = AttentionUNet_DANN()
        
        
    def forward(self, x):
        x1, x2, x3, x4, x5 = self.AttentionUNet_FE(x)
        logits = self.AttentionUNet_DE(x1, x2, x3, x4, x5)
        domain_output = self.AttentionUNet_DANN(x5)
        return logits, domain_output, x5
class AttentionUNet_DANN(nn.Module):
    def __init__(self, bilinear=True):
        super(AttentionUNet_DANN, self).__init__()
        factor = 2 if bilinear else 2
        #=========================dann
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('h_fc1', nn.Linear(1024//4 * 64 * 64, 100))
#         self.domain_classifier.add_module('h_fc1', nn.Linear(512, 100))
        self.domain_classifier.add_module('h_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('h_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('h_fc2', nn.Linear(100, 3))
        self.domain_classifier.add_module('h_softmax', nn.Softmax(dim=1))
        
#         self.s_domain_classifier = nn.Sequential()
#         self.s_domain_classifier.add_module('s_fc1', nn.Linear(512, 100))
# #         self.s_domain_classifier.add_module('s_fc1', nn.Linear(1024 // factor * 64 * 64, 100))
#         self.s_domain_classifier.add_module('s_bn1', nn.BatchNorm1d(100))
#         self.s_domain_classifier.add_module('s_relu1', nn.ReLU(True))
#         self.s_domain_classifier.add_module('s_fc2', nn.Linear(100, 2))
#         self.s_domain_classifier.add_module('s_softmax', nn.Softmax(dim=1))
        
#         self.l_domain_classifier = nn.Sequential()
#         self.l_domain_classifier.add_module('d_fc1', nn.Linear(512, 100))
# #         self.l_domain_classifier.add_module('d_fc1', nn.Linear(1024 // factor * 64 * 64, 100))
#         self.l_domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
#         self.l_domain_classifier.add_module('d_relu1', nn.ReLU(True))
#         self.l_domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
#         self.l_domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))

    def forward(self, x5):
#         feature = x5.view(-1, 1024 // 2 * 64 * 64)
#         feature = nn.AdaptiveAvgPool2d((1,1))(x5)
        feature = torch.flatten(x5, 1)
#         print('x5.size()',x5.size())ㄋ
#         print('feature.size()',feature.size())
#         reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_output = self.domain_classifier(feature)
#         s_domain_output = self.s_domain_classifier(feature)
#         l_domain_output = self.l_domain_classifier(feature)
        return domain_output


# -

net = AttentionUNet(n_channels=3, n_classes=2, bilinear=True)
net.AttentionUNet_DE.Up2.up[1].weight[0][0][0]
net.AttentionUNet_DANN.domain_classifier.h_fc2.weight[0][0]
# net.AttentionUNet_DE.Up2.up[1].weight[0][0][0]


