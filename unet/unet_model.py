# -*- coding: utf-8 -*-
""" Full assembly of the parts to form the complete network """

import functools
from .unet_parts import *
from torch.autograd import Function
# from seed import seed_everything
from .modules import (
    ResidualConv,
    ASPP,
    ResUnetPlusPlu_AttentionBlock,
    Upsample_,
    Squeeze_Excite_Block,
)


class UNet_ori(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_ori, self).__init__()
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

    def forward(self, x):
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
        return logits


class UNet_FE(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True):
        super(UNet_FE, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5


class UNet_DE(nn.Module):
    def __init__(self, n_classes=2 ,bilinear=True):
        super(UNet_DE, self).__init__()
        self.n_classes = n_classes
        factor = 2 if bilinear else 1
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# +
class UNet_DANN(nn.Module):
    def __init__(self, bilinear=True):
        super(UNet_DANN, self).__init__()
        factor = 2 if bilinear else 2
        #=========================dann
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('h_fc1', nn.Linear(1024 // 8 * 64 * 64, 100))
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
#         print('x5.size()',x5.size())
#         print('feature.size()',feature.size())
#         reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_output = self.domain_classifier(feature)
#         s_domain_output = self.s_domain_classifier(feature)
#         l_domain_output = self.l_domain_classifier(feature)
        return domain_output


# -

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.UNet_FE = UNet_FE(self.n_channels, self.n_classes, self.bilinear)
        self.UNet_DE = UNet_DE(self.n_classes)
        self.UNet_DANN = UNet_DANN()
        
        
    def forward(self, x):
        x1, x2, x3, x4, x5 = self.UNet_FE(x)
        logits = self.UNet_DE(x1, x2, x3, x4, x5)
        domain_output = self.UNet_DANN(x5)
        return logits, domain_output, x5

# +



# class AttentionUNet(nn.Module):

#     def __init__(self, img_ch=3, output_ch=1):
#         super(AttentionUNet, self).__init__()

#         self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.Conv1 = ConvBlock(img_ch, 64)
#         self.Conv2 = ConvBlock(64, 128)
#         self.Conv3 = ConvBlock(128, 256)
#         self.Conv4 = ConvBlock(256, 512)
#         self.Conv5 = ConvBlock(512, 1024)

#         self.Up5 = UpConv(1024, 512)
#         self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
#         self.UpConv5 = ConvBlock(1024, 512)

#         self.Up4 = UpConv(512, 256)
#         self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
#         self.UpConv4 = ConvBlock(512, 256)

#         self.Up3 = UpConv(256, 128)
#         self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
#         self.UpConv3 = ConvBlock(256, 128)

#         self.Up2 = UpConv(128, 64)
#         self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
#         self.UpConv2 = ConvBlock(128, 64)

#         self.Conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

#     def forward(self, x):
#         """
#         e : encoder layers
#         d : decoder layers
#         s : skip-connections from encoder layers to decoder layers
#         """
#         e1 = self.Conv1(x)

#         e2 = self.MaxPool(e1)
#         e2 = self.Conv2(e2)

#         e3 = self.MaxPool(e2)
#         e3 = self.Conv3(e3)

#         e4 = self.MaxPool(e3)
#         e4 = self.Conv4(e4)

#         e5 = self.MaxPool(e4)
#         e5 = self.Conv5(e5)

#         d5 = self.Up5(e5)

#         s4 = self.Att5(gate=d5, skip_connection=e4)
#         d5 = torch.cat((s4, d5), dim=1) # concatenate attention-weighted skip connection with previous layer output
#         d5 = self.UpConv5(d5)

#         d4 = self.Up4(d5)
#         s3 = self.Att4(gate=d4, skip_connection=e3)
#         d4 = torch.cat((s3, d4), dim=1)
#         d4 = self.UpConv4(d4)

#         d3 = self.Up3(d4)
#         s2 = self.Att3(gate=d3, skip_connection=e2)
#         d3 = torch.cat((s2, d3), dim=1)
#         d3 = self.UpConv3(d3)

#         d2 = self.Up2(d3)
#         s1 = self.Att2(gate=d2, skip_connection=e1)
#         d2 = torch.cat((s1, d2), dim=1)
#         d2 = self.UpConv2(d2)

#         out = self.Conv(d2)

#         return out
# +
# # 3*3 AttentionUNet
# import torch
# import torch.nn as nn


# class ConvBlock(nn.Module):

#     def __init__(self, in_channels, out_channels):
#         super(ConvBlock, self).__init__()

#         # number of input channels is a number of filters in the previous layer
#         # number of output channels is a number of filters in the current layer
#         # "same" convolutions
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         return x


# class UpConv(nn.Module):

#     def __init__(self, in_channels, out_channels):
#         super(UpConv, self).__init__()

#         self.up = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         x = self.up(x)
#         return x


# class AttentionBlock(nn.Module):
#     """Attention block with learnable parameters"""

#     def __init__(self, F_g, F_l, n_coefficients):
#         """
#         :param F_g: number of feature maps (channels) in previous layer
#         :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
#         :param n_coefficients: number of learnable multi-dimensional attention coefficients
#         """
#         super(AttentionBlock, self).__init__()

#         self.W_gate = nn.Sequential(
#             nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(n_coefficients)
#         )

#         self.W_x = nn.Sequential(
#             nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(n_coefficients)
#         )

#         self.psi = nn.Sequential(
#             nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )

#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, gate, skip_connection):
#         """
#         :param gate: gating signal from previous layer
#         :param skip_connection: activation from corresponding encoder layer
#         :return: output activations
#         """
#         g1 = self.W_gate(gate)
#         x1 = self.W_x(skip_connection)
#         psi = self.relu(g1 + x1)
#         psi = self.psi(psi)
#         out = skip_connection * psi
#         return out

# class AttentionUNet_FE(nn.Module):
#     def __init__(self, n_channels=3, n_classes=2, bilinear=True):
#         super(AttentionUNet_FE, self).__init__()
#         self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.Conv1 = ConvBlock(n_channels, 64*2)
#         self.Conv2 = ConvBlock(64*2, 128*2)
#         self.Conv3 = ConvBlock(128*2, 256*2)
#         self.Conv4 = ConvBlock(256*2, 512*2)
#         self.Conv5 = ConvBlock(512*2, 1024*2)
        

#     def forward(self, x):
#         e1 = self.Conv1(x)

#         e2 = self.MaxPool(e1)
#         e2 = self.Conv2(e2)

#         e3 = self.MaxPool(e2)
#         e3 = self.Conv3(e3)

#         e4 = self.MaxPool(e3)
#         e4 = self.Conv4(e4)

#         e5 = self.MaxPool(e4)
#         e5 = self.Conv5(e5)
#         return e1, e2, e3, e4, e5
    
# class AttentionUNet_DE(nn.Module):
#     def __init__(self, n_classes=2, bilinear=True):
#         super(AttentionUNet_DE, self).__init__()
        
#         self.Up5 = UpConv(1024*2, 512*2)
#         self.Att5 = AttentionBlock(F_g=512*2, F_l=512*2, n_coefficients=256)
#         self.UpConv5 = ConvBlock(1024*2, 512*2)

#         self.Up4 = UpConv(512*2, 256*2)
#         self.Att4 = AttentionBlock(F_g=256*2, F_l=256*2, n_coefficients=128)
#         self.UpConv4 = ConvBlock(512*2, 256*2)

#         self.Up3 = UpConv(256*2, 128*2)
#         self.Att3 = AttentionBlock(F_g=128*2, F_l=128*2, n_coefficients=64)
#         self.UpConv3 = ConvBlock(256*2, 128*2)

#         self.Up2 = UpConv(128*2, 64*2)
#         self.Att2 = AttentionBlock(F_g=64*2, F_l=64*2, n_coefficients=32)
#         self.UpConv2 = ConvBlock(128*2, 64*2)

#         self.Conv = nn.Conv2d(64*2, n_classes, kernel_size=1, stride=1, padding=0)

#     def forward(self, e1, e2, e3, e4, e5):
#         d5 = self.Up5(e5)

#         s4 = self.Att5(gate=d5, skip_connection=e4)
#         d5 = torch.cat((s4, d5), dim=1) # concatenate attention-weighted skip connection with previous layer output
#         d5 = self.UpConv5(d5)

#         d4 = self.Up4(d5)
#         s3 = self.Att4(gate=d4, skip_connection=e3)
#         d4 = torch.cat((s3, d4), dim=1)
#         d4 = self.UpConv4(d4)

#         d3 = self.Up3(d4)
#         s2 = self.Att3(gate=d3, skip_connection=e2)
#         d3 = torch.cat((s2, d3), dim=1)
#         d3 = self.UpConv3(d3)

#         d2 = self.Up2(d3)
#         s1 = self.Att2(gate=d2, skip_connection=e1)
#         d2 = torch.cat((s1, d2), dim=1)
#         d2 = self.UpConv2(d2)

#         out = self.Conv(d2)
        
#         return out
    
# class AttentionUNet(nn.Module):
#     def __init__(self, n_channels=3, n_classes=2, bilinear=True):
#         super(AttentionUNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
        
#         self.AttentionUNet_FE = AttentionUNet_FE(self.n_channels, self.n_classes, self.bilinear)
#         self.AttentionUNet_DE = AttentionUNet_DE(self.n_classes)
#         self.AttentionUNet_DANN = AttentionUNet_DANN()
        
        
#     def forward(self, x):
#         x1, x2, x3, x4, x5 = self.AttentionUNet_FE(x)
#         logits = self.AttentionUNet_DE(x1, x2, x3, x4, x5)
#         domain_output = self.AttentionUNet_DANN(x5)
#         return logits, domain_output, x5

# class AttentionUNet_DANN(nn.Module):
#     def __init__(self, bilinear=True):
#         super(AttentionUNet_DANN, self).__init__()
#         factor = 2 if bilinear else 2
#         #=========================dann
#         self.domain_classifier = nn.Sequential()
#         self.domain_classifier.add_module('h_fc1', nn.Linear(1024//4 * 64 * 64*2, 100))
# #         self.domain_classifier.add_module('h_fc1', nn.Linear(512, 100))
#         self.domain_classifier.add_module('h_bn1', nn.BatchNorm1d(100))
#         self.domain_classifier.add_module('h_relu1', nn.ReLU(True))
#         self.domain_classifier.add_module('h_fc2', nn.Linear(100, 3))
#         self.domain_classifier.add_module('h_softmax', nn.Softmax(dim=1))
        
# #         self.s_domain_classifier = nn.Sequential()
# #         self.s_domain_classifier.add_module('s_fc1', nn.Linear(512, 100))
# # #         self.s_domain_classifier.add_module('s_fc1', nn.Linear(1024 // factor * 64 * 64, 100))
# #         self.s_domain_classifier.add_module('s_bn1', nn.BatchNorm1d(100))
# #         self.s_domain_classifier.add_module('s_relu1', nn.ReLU(True))
# #         self.s_domain_classifier.add_module('s_fc2', nn.Linear(100, 2))
# #         self.s_domain_classifier.add_module('s_softmax', nn.Softmax(dim=1))
        
# #         self.l_domain_classifier = nn.Sequential()
# #         self.l_domain_classifier.add_module('d_fc1', nn.Linear(512, 100))
# # #         self.l_domain_classifier.add_module('d_fc1', nn.Linear(1024 // factor * 64 * 64, 100))
# #         self.l_domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
# #         self.l_domain_classifier.add_module('d_relu1', nn.ReLU(True))
# #         self.l_domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
# #         self.l_domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))

#     def forward(self, x5):
# #         feature = x5.view(-1, 1024 // 2 * 64 * 64)
# #         feature = nn.AdaptiveAvgPool2d((1,1))(x5)
#         feature = torch.flatten(x5, 1)
# #         print('x5.size()',x5.size())ㄋ
# #         print('feature.size()',feature.size())
# #         reverse_feature = ReverseLayerF.apply(feature, alpha)
#         domain_output = self.domain_classifier(feature)
# #         s_domain_output = self.s_domain_classifier(feature)
# #         l_domain_output = self.l_domain_classifier(feature)
#         return domain_output
# +
# 5*5 AttentionUNet
import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        # number of input channels is a number of filters in the previous layer
        # number of output channels is a number of filters in the current layer
        # "same" convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=True),
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
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=True),
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
    def __init__(self, n_channels=3, n_classes=2, bilinear=True, _k = 1):
        super(AttentionUNet_FE, self).__init__()
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(n_channels, 64*_k)
        self.Conv2 = ConvBlock(64*_k, 128*_k)
        self.Conv3 = ConvBlock(128*_k, 256*_k)
        self.Conv4 = ConvBlock(256*_k, 512*_k)
        self.Conv5 = ConvBlock(512*_k, 1024*_k)
        

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
    def __init__(self, n_classes=2, bilinear=True, _k = 1):
        super(AttentionUNet_DE, self).__init__()
        
        self.Up5 = UpConv(1024*_k, 512*_k)
        self.Att5 = AttentionBlock(F_g=512*_k, F_l=512*_k, n_coefficients=256)
        self.UpConv5 = ConvBlock(1024*_k, 512*_k)

        self.Up4 = UpConv(512*_k, 256*_k)
        self.Att4 = AttentionBlock(F_g=256*_k, F_l=256*_k, n_coefficients=128)
        self.UpConv4 = ConvBlock(512*_k, 256*_k)

        self.Up3 = UpConv(256*_k, 128*_k)
        self.Att3 = AttentionBlock(F_g=128*_k, F_l=128*_k, n_coefficients=64)
        self.UpConv3 = ConvBlock(256*_k, 128*_k)

        self.Up2 = UpConv(128*_k, 64*_k)
        self.Att2 = AttentionBlock(F_g=64*_k, F_l=64*_k, n_coefficients=32)
        self.UpConv2 = ConvBlock(128*_k, 64*_k)

        self.Conv = nn.Conv2d(64*_k, n_classes, kernel_size=1, stride=1, padding=0)

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
    def __init__(self, n_channels=3, n_classes=2, bilinear=True, _k = 1):
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
    def __init__(self, bilinear=True, _k = 1):
        super(AttentionUNet_DANN, self).__init__()
        factor = 2 if bilinear else 2
        #=========================dann
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('h_fc1', nn.Linear(1024//4 * 64 * 64*_k, 100))
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


# +
# 5*5 ResUnetPlusPlus
# # double
# class ResUnetPlusPlus_FE(nn.Module):
#     def __init__(self, n_channels=3, n_classes=2, bilinear=True, filters=[64, 128, 256, 512, 1024]):
#         super(ResUnetPlusPlus_FE, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.input_layer = nn.Sequential(
#             nn.Conv2d(n_channels, filters[0], kernel_size=3, padding=1),
#             nn.BatchNorm2d(filters[0]),
#             nn.ReLU(),
#             nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
#         )
#         self.input_skip = nn.Sequential(
#             nn.Conv2d(n_channels, filters[0], kernel_size=3, padding=1)
#         )

#         self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])

#         self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)

#         self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])

#         self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)

#         self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])

#         self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

#         self.aspp_bridge = ASPP(filters[3], filters[4])
        
#     def forward(self, x):
#         x1 = self.input_layer(x) + self.input_skip(x)

#         x2 = self.squeeze_excite1(x1)
#         x2 = self.residual_conv1(x2)

#         x3 = self.squeeze_excite2(x2)
#         x3 = self.residual_conv2(x3)

#         x4 = self.squeeze_excite3(x3)
#         x4 = self.residual_conv3(x4)

#         x5 = self.aspp_bridge(x4)
#         return x1, x2, x3, x4, x5
# class ResUnetPlusPlus_DE(nn.Module):
#     def __init__(self, n_classes=2, bilinear=True, filters=[64, 128, 256, 512, 1024]):
#         super(ResUnetPlusPlus_DE, self).__init__()
#         self.attn1 = ResUnetPlusPlu_AttentionBlock(filters[2], filters[4], filters[4])
#         self.upsample1 = Upsample_(2)
#         self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

#         self.attn2 = ResUnetPlusPlu_AttentionBlock(filters[1], filters[3], filters[3])
#         self.upsample2 = Upsample_(2)
#         self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

#         self.attn3 = ResUnetPlusPlu_AttentionBlock(filters[0], filters[2], filters[2])
#         self.upsample3 = Upsample_(2)
#         self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

#         self.aspp_out = ASPP(filters[1], filters[0])

#         self.output_layer = nn.Sequential(
#             nn.Conv2d(filters[0], n_classes, 1, 1),
#             # nn.Sigmoid()
#         )
#     def forward(self, x1, x2, x3, x4, x5):
#         x6 = self.attn1(x3, x5)
#         x6 = self.upsample1(x6)
#         x6 = torch.cat([x6, x3], dim=1)
#         x6 = self.up_residual_conv1(x6)

#         x7 = self.attn2(x2, x6)
#         x7 = self.upsample2(x7)
#         x7 = torch.cat([x7, x2], dim=1)
#         x7 = self.up_residual_conv2(x7)

#         x8 = self.attn3(x1, x7)
#         x8 = self.upsample3(x8)
#         x8 = torch.cat([x8, x1], dim=1)
#         x8 = self.up_residual_conv3(x8)

#         x9 = self.aspp_out(x8)
#         out = self.output_layer(x9)

#         return out
# class ResUnetPlusPlus_DANN(nn.Module):
#     def __init__(self, bilinear=True):
#         super(ResUnetPlusPlus_DANN, self).__init__()
#         factor = 2 if bilinear else 2
#         #=========================dann
#         self.domain_classifier = nn.Sequential()
#         self.domain_classifier.add_module('h_fc1', nn.Linear(1024//4 * 64 * 64*2, 100))
#         self.domain_classifier.add_module('h_bn1', nn.BatchNorm1d(100))
#         self.domain_classifier.add_module('h_relu1', nn.ReLU(True))
#         self.domain_classifier.add_module('h_fc2', nn.Linear(100, 3))
#         self.domain_classifier.add_module('h_softmax', nn.Softmax(dim=1))
        
#     def forward(self, x4):
#         feature = torch.flatten(x4, 1)
#         domain_output = self.domain_classifier(feature)
#         return domain_output
# class ResUnetPlusPlus(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True, filters=[32, 64, 128, 256, 512]):
#         super(ResUnetPlusPlus, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
        
#         self.ResUnetPlusPlus_FE = ResUnetPlusPlus_FE(self.n_channels, self.n_classes, self.bilinear)
#         self.ResUnetPlusPlus_DE = ResUnetPlusPlus_DE(self.n_classes)
#         self.ResUnetPlusPlus_DANN = ResUnetPlusPlus_DANN()
        
#     def forward(self, x):
#         x1, x2, x3, x4, x5 = self.ResUnetPlusPlus_FE(x)
#         logits = self.ResUnetPlusPlus_DE(x1, x2, x3, x4, x5)
#         domain_output = self.ResUnetPlusPlus_DANN(x4)
#         return logits, domain_output, x5

# +
import math
class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net" 
    <https://arxiv.org/pdf/1807.09441.pdf>`
    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """
    def __init__(self, planes):
        super(IBN, self).__init__()
#         self.half = int(planes * (1-ratio))
#         self.BN = nn.BatchNorm2d(self.half)
        self.IN = nn.InstanceNorm2d(planes, affine=True)

    def forward(self, x):
#         split = torch.split(x, self.half, 1)
#         out1 = self.BN(split[0].contiguous())
        out2 = self.IN(x.contiguous())
#         out = torch.cat((out1, out2), 1)
        return out2
def random_sample(prob, sampling_num):
    batch_size, channels, h, w = prob.shape
    return torch.multinomial((prob.view(batch_size * channels, -1) + 1e-8), sampling_num, replacement=True)

class Info_Dropout(nn.Module):  # slow version
    def __init__(self, indim, outdim, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, if_pool=False, pool_kernel_size=2, pool_stride=None,
                 pool_padding=0, pool_dilation=1, mode = True):
        super(Info_Dropout, self).__init__()
        if groups != 1:
            raise ValueError('InfoDropout only supports groups=1')
        self.mode = mode
        self.indim = indim
        self.outdim = indim
        self.if_pool = if_pool
        self.drop_rate = 1.5
        self.temperature = 0.01
        self.band_width = 1.0
        self.radius = 1
#         print('---------',self.mode)
        self.conv_info = nn.Conv2d(self.indim, self.indim * (self.radius * 2 + 1)**2, 
                                   kernel_size=self.radius * 2 + 1,
                                   padding=0, groups=self.indim, bias=False)
        self.conv_info.weight.data = self.Create_conv_info_kernel()
        self.conv_info.weight.requires_grad = False

        self.all_one_conv_indim_wise = nn.Conv2d(indim * (self.radius * 2 + 1)**2, self.indim,
                                                 kernel_size=3, stride=stride, padding=0,
                                                 groups=self.indim, bias=False)
        self.all_one_conv_indim_wise.weight.data = torch.ones_like(self.all_one_conv_indim_wise.weight, dtype=torch.float)
        self.all_one_conv_indim_wise.weight.requires_grad = False

#         self.all_one_conv_radius_wise = nn.Conv2d((self.radius * 2 + 1)**2, outdim, kernel_size=1, padding=0, bias=False)
#         self.all_one_conv_radius_wise.weight.data = torch.ones_like(self.all_one_conv_radius_wise.weight, dtype=torch.float)
#         self.all_one_conv_radius_wise.weight.requires_grad = False
        

        if if_pool:
            self.pool = nn.MaxPool2d(pool_kernel_size, pool_stride, pool_padding, pool_dilation)

    def Create_conv_info_kernel(self):
        kernel = torch.zeros(((self.radius * 2 + 1)**2, 1, self.radius * 2 + 1, self.radius * 2 + 1), dtype=torch.float)
        kernel[:, :, self.radius, self.radius] += 1
        for i in range(2 * self.radius + 1):
            for j in range(2 * self.radius + 1):
                kernel[i * (2 * self.radius + 1) + j, 0, i, j] -= 1
        
        return kernel.repeat(self.indim, 1, 1, 1)

    def initialize_parameters(self):
        self.conv_info.weight.data = self.Create_conv_info_kernel()
        self.conv_info.weight.requires_grad = False

        self.all_one_conv_indim_wise.weight.data = torch.ones_like(self.all_one_conv_indim_wise.weight, dtype=torch.float)
        self.all_one_conv_indim_wise.weight.requires_grad = False

#         self.all_one_conv_radius_wise.weight.data = torch.ones_like(self.all_one_conv_radius_wise.weight, dtype=torch.float)
#         self.all_one_conv_radius_wise.weight.requires_grad = False


    def forward(self, x):
        with torch.no_grad():
            distance = self.conv_info(x)
#             print('--------------------------')
#             print('------------0',distance.size())
            sigma = 1
            batch_size, _, h_dis, w_dis = distance.size()
#             distance = distance.reshape(batch_size, self.indim, (self.radius * 2 + 1)**2,
#                                         h_dis, w_dis).transpose(1, 2).\
#                 reshape(batch_size, (self.radius * 2 + 1)**2 * self.indim, h_dis, w_dis)
#             print('------------1',distance.size())
#             print('------------1',distance.size())
#             print('---',distance.max())
#             print('---',distance.min())
#             print('---max',(torch.exp(-distance**2/2/sigma)/(2*math.pi*sigma*sigma)**0.5).max())
#             print('---min',(torch.exp(-distance**2/2/sigma)/(2*math.pi*sigma*sigma)**0.5).min())
            distance = torch.exp((distance*distance)) / torch.sum(torch.exp((distance*distance)))
            distance = self.all_one_conv_indim_wise(distance)
            #distance = self.all_one_conv_indim_wise(torch.exp(-(distance*distance/2)) )
#             print(distance.max())
#             print(distance.min())
            prob = distance/9
#             print(prob.max())
#             print(prob.min())
#             print('------------1',distance.size())
#             print('------------2',distance.size())
#             prob = torch.exp(-distance / distance.mean() / 2 / self.band_width**2) # using mean of distance to normalize
#             prob = self.all_one_conv_radius_wise(distance) 
#             print('------------2',prob.size())
#             _max = (prob.view(prob.size(0), -1)).max(1)[0]
#             _max = _max.unsqueeze(1).unsqueeze(1).unsqueeze(1)
#             prob_max = prob/_max
#             prob_max = 1-prob_max
#             _max = (prob_max.view(prob_max.size(0), -1)).max(1)[0]
#             _max = _max.unsqueeze(1).unsqueeze(1).unsqueeze(1)
#             prob_max = prob_max/_max
#             p1d = (2, 2,2,2)
#             prob_max = torch.nn.functional.pad(prob_max, p1d, "constant", 0)  
#             print('------------3',prob_max.size())
            if self.if_pool:
                prob = -self.pool(-prob)  # min pooling of probability

            
#             if not self.mode:
#                 print('---------',self.mode)
#                 print('x',x.size())
#                 print('prob',prob.size())
#                 print('torch.exp(-self.drop_rate * prob * h * w)',torch.exp(-self.drop_rate * prob * h * w).size())
#                 return x * torch.exp(-self.drop_rate * prob * h * w)

#             random_choice = random_sample(prob, sampling_num=int(self.drop_rate * h * w))

#             random_mask = torch.ones((batch_size * channels, h * w), device='cuda')
#             random_mask[torch.arange(batch_size * channels, device='cuda').view(-1, 1), random_choice] = 0

        return prob


# +
# 5*5 ResUnetPlusPlus
class ResUnetPlusPlus_FE(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True, filters=[32, 64, 128, 256, 512]):
        super(ResUnetPlusPlus_FE, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_layer1 = nn.Sequential(
            nn.Conv2d(n_channels, filters[0], kernel_size=5, padding=2),
            IBN(filters[0]),
            nn.ReLU(),
        )
        self.input_layer2 = nn.Sequential(
            nn.Conv2d(filters[0], filters[0], kernel_size=5, padding=2),
        )
        self.infodrop_1 = Info_Dropout(filters[0], filters[0], kernel_size=3, stride=1)
        self.input_skip = nn.Sequential(
            nn.Conv2d(n_channels, filters[0], kernel_size=5, padding=2)
        )

        self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])

        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)

        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])

        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])

        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.aspp_bridge = ASPP(filters[3], filters[4])
        
    def forward(self, x):
        x_ = self.input_layer1(x) 
        x_i = self.infodrop_1(x_)
        x1 = self.input_layer2(x_) + self.input_skip(x)
        
        
        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)

        x5 = self.aspp_bridge(x4)
        return x1, x2, x3, x4, x5, x_, x_i
    
class ResUnetPlusPlus_DE(nn.Module):
    def __init__(self, n_classes=2, bilinear=True, filters=[32, 64, 128, 256, 512]):
        super(ResUnetPlusPlus_DE, self).__init__()
        self.attn1 = ResUnetPlusPlu_AttentionBlock(filters[2], filters[4], filters[4])
        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

        self.attn2 = ResUnetPlusPlu_AttentionBlock(filters[1], filters[3], filters[3])
        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

        self.attn3 = ResUnetPlusPlu_AttentionBlock(filters[0], filters[2], filters[2])
        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

        self.aspp_out = ASPP(filters[1], filters[0])

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], n_classes, 1, 1),
            # nn.Sigmoid()
        )
    def forward(self, x1, x2, x3, x4, x5):
        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)

        return out
    
# class ResUnetPlusPlus_DANN(nn.Module):
#     """Defines a PatchGAN discriminator"""

#     def __init__(self, input_nc=256, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
#         """Construct a PatchGAN discriminator
#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             ndf (int)       -- the number of filters in the last conv layer
#             n_layers (int)  -- the number of conv layers in the discriminator
#             norm_layer      -- normalization layer
#         """
#         super(ResUnetPlusPlus_DANN, self).__init__()
#         if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d

#         kw = 4
#         padw = 1
#         sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):  # gradually increase the number of filters
#             nf_mult_prev = nf_mult
#             nf_mult = min(2 ** n, 8)
#             sequence += [
#                 nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#                 norm_layer(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, True)
#             ]

#         nf_mult_prev = nf_mult
#         nf_mult = min(2 ** n_layers, 8)
#         sequence += [
#             nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#             norm_layer(ndf * nf_mult),
#             nn.LeakyReLU(0.2, True)
#         ]

#         sequence += [nn.Conv2d(ndf * nf_mult, 3, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
#         self.model = nn.Sequential(*sequence)

#     def forward(self, input):
#         """Standard forward."""
#         return self.model(input)

    
class Projection(nn.Module):
#   """
#   Creates projection head
#   Args:
#     n_in (int): Number of input features
#     n_hidden (int): Number of hidden features
#     n_out (int): Number of output features
#     use_bn (bool): Whether to use batch norm
#   """
    def __init__(self, n_in: int, n_hidden: int, n_out: int,use_bn: bool = True):
        super().__init__()

        # No point in using bias if we've batch norm
        self.lin1 = nn.Linear(n_in, n_hidden, bias=not use_bn)
        self.bn = nn.BatchNorm1d(n_hidden) if use_bn else nn.Identity()
        self.relu = nn.ReLU()
        # No bias for the final linear layer
        self.lin2 = nn.Linear(n_hidden, n_out, bias=False)
    
    def forward(self, x_):
        x_ = self.lin1(x_)
        x_ = self.bn(x_)
        x_ = self.relu(x_)
        x_ = self.lin2(x_)
        return x_   
    
# class ResUnetPlusPlus_DANN(nn.Module):
#     def __init__(self, bilinear=True):
#         super(ResUnetPlusPlus_DANN, self).__init__()
#         factor = 2 if bilinear else 2
#         #=========================dann
#         self.domain_classifier = nn.Sequential()
#         self.domain_classifier.add_module('h_fc1', nn.Linear(1024//4 * 64 * 64, 100))
#         self.domain_classifier.add_module('h_bn1', nn.BatchNorm1d(100))
#         self.domain_classifier.add_module('h_relu1', nn.ReLU(True))
#         self.domain_classifier.add_module('h_fc2', nn.Linear(100, 3))
#         self.domain_classifier.add_module('h_softmax', nn.Softmax(dim=1))
        
#     def forward(self, x4):
#         feature = torch.flatten(x4, 1)
#         domain_output = self.domain_classifier(feature)
#         return domain_output
    
class ResUnetPlusPlus(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True, filters=[64, 128, 256, 512, 1024]):
#     def __init__(self, n_channels, n_classes, bilinear=True, filters=[32, 64, 128, 256, 512]):
    def __init__(self, n_channels, n_classes, bilinear=True, filters=[16, 32, 64, 128, 256]):
#     def __init__(self, n_channels, n_classes, bilinear=True, filters=[8, 16, 32, 64, 128]):
        super(ResUnetPlusPlus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.filters = filters
        self.ResUnetPlusPlus_FE = ResUnetPlusPlus_FE(self.n_channels, self.n_classes, self.bilinear,self.filters)
        self.ResUnetPlusPlus_DE = ResUnetPlusPlus_DE(self.n_classes, self.bilinear,self.filters)
#         self.ResUnetPlusPlus_DANN = ResUnetPlusPlus_DANN()
#         self.Projection = Projection(256*64*64, 128, 128, 1)
    
    def forward(self, x):
        x1, x2, x3, x4, x5, x_, x_i = self.ResUnetPlusPlus_FE(x)
        logits = self.ResUnetPlusPlus_DE(x1, x2, x3, x4, x5)
#         domain_output = self.ResUnetPlusPlus_DANN(x4)
#         x4 = torch.flatten(x4, 1)
#         print(x4.size())
#         Projection = self.Projection(x4)
#         print(Projection.size())
        return logits, x5, x_, x_i

class convolution_1x1(nn.Module):
    def __init__(self):
        super(convolution_1x1, self).__init__()
        self.input_layer = nn.Sequential(
        nn.Conv2d(4, 2, kernel_size=1,))
    def forward(self, x):
        logits = self.input_layer(x)
        return logits
