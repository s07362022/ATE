# -*- coding: utf-8 -*-
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
import imgaug.augmenters as iaa

# +
# transform = A.Compose(
#     [
# #         A.HorizontalFlip(p=0.5),
# #         A.VerticalFlip(p=0.5),
#         A.RGBShift(p=1)#,r_shift_limit=50, g_shift_limit=50, b_shift_limit=50),
# #         A.RandomCrop(always_apply=True, p=1.0, height=512, width=512),
# #         A.ShiftScaleRotate(p=0.5),
# #         ToTensorV2(transpose_mask=True)
#     ],
#     additional_targets={'label': 'image'}
# )

transform = iaa.Sequential([
#                             iaa.Fliplr(0.5),
#                             iaa.Flipud(0.5),
#                             iaa.Rot90((0, 1)),
                            iaa.AddToHue((-3,3)),
                            iaa.MultiplySaturation((0.9,1.1)),
                            iaa.AddToSaturation((-10,10)),
                            iaa.OneOf([iaa.LinearContrast((0.9,1.1)),
                                        iaa.GammaContrast((0.9,1.1))]),
                            iaa.AddToBrightness((-10,10)),
                            iaa.Sometimes(0.5, iaa.OneOf([iaa.MedianBlur(k=(3,5)),
                                                            iaa.GaussianBlur(sigma=(0.0, 1.0))])),
                            
                            iaa.Sometimes(0.5, iaa.OneOf([iaa.AdditiveGaussianNoise(scale=(0, 0.05*255), per_channel=True),
                                                            iaa.AdditiveLaplaceNoise(scale=(0, 0.05*255), per_channel=True)])),
#                             iaa.Affine(scale=(0.875, 1.125), translate_percent=(-0.0625, 0.0625), rotate=(-10,10),mode='reflect'),
#                             iaa.Crop(percent= 0.25, keep_size=False, sample_independently=True),
                            ])

to_tensor = A.Compose([ToTensorV2(transpose_mask=True)],additional_targets={'label': 'image'})

# ALL_transform = A.Compose(
#     [
# #         A.HorizontalFlip(p=0.5),
# #         A.VerticalFlip(p=0.5),
# #         A.augmentations.geometric.rotate.RandomRotate90(p=0.5),
# #         A.augmentations.geometric.rotate.Rotate(p=0.5),
#         A.OneOf([
# #             A.RGBShift(p=0.5),
#             A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=15, val_shift_limit=0,p=0.5)
# #             A.HueSaturationValue(hue_shift_limit=0,p=0.5)
# #             A.HueSaturationValue(p=0.5)
            
# #             A.RandomBrightnessContrast(p=0.5),
# #             A.MotionBlur(p=0.2),   # 使用随机大小的内核将运动模糊应用于输入图像。
# #             A.MedianBlur(blur_limit=3, p=0.1),    # 中值滤波
# #             A.Blur(blur_limit=3, p=0.1),   # 使用随机大小的内核模糊输入图像。
#         ], p=1)
        
# #         A.OneOf([
# #             A.RGBShift(p=0.5),#,r_shift_limit=50, g_shift_limit=50, b_shift_limit=50),
# #             A.RandomBrightnessContrast(p=0.5),
# # #             A.MotionBlur(p=0.2),   # 使用随机大小的内核将运动模糊应用于输入图像。
# # #             A.MedianBlur(blur_limit=3, p=0.1),    # 中值滤波
# # #             A.Blur(blur_limit=3, p=0.1),   # 使用随机大小的内核模糊输入图像。
# #         ], p=1)

# #         A.RandomCrop(always_apply=True, p=1.0, height=512, width=512),
# #         A.ShiftScaleRotate(p=0.5),
# #         ToTensorV2(transpose_mask=True)
#     ],
#     additional_targets={'label': 'image'}
# )
ALL_transform = iaa.Sequential([
                            iaa.Fliplr(0.5),
                            iaa.Flipud(0.5),
                            iaa.Rot90((0, 1)),
                            iaa.AddToHue((-3,3)),
                            iaa.MultiplySaturation((0.9,1.1)),
                            iaa.AddToSaturation((-10,10)),
                            iaa.OneOf([iaa.LinearContrast((0.9,1.1)),
                                        iaa.GammaContrast((0.9,1.1))]),
                            iaa.AddToBrightness((-10,10)),
                            iaa.Sometimes(0.5, iaa.OneOf([iaa.MedianBlur(k=(3,5)),
                                                            iaa.GaussianBlur(sigma=(0.0, 1.0))])),
                            
                            iaa.Sometimes(0.5, iaa.OneOf([iaa.AdditiveGaussianNoise(scale=(0, 0.05*255), per_channel=True),
                                                            iaa.AdditiveLaplaceNoise(scale=(0, 0.05*255), per_channel=True)])),
#                             iaa.Affine(scale=(0.875, 1.125), translate_percent=(-0.0625, 0.0625), rotate=(-10,10),mode='reflect'),
#                             iaa.Crop(percent= 0.25, keep_size=False, sample_independently=True),
                            ])


# +
class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = '_nuc_filled',data_argu = 0):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.data_argu = data_argu

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        img_ndarray = pil_img
#         w, h = pil_img.size
# #         print('==============',w,h)
#         newW, newH = int(scale * w), int(scale * h)
#         assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
#         pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
#         img_ndarray = np.asarray(pil_img)

#         if img_ndarray.ndim == 2 and not is_mask:
#             img_ndarray = img_ndarray[np.newaxis, ...]
# #         elif not is_mask:
# #             img_ndarray = img_ndarray.transpose((2, 0, 1))
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
#         if ext in ['.npz', '.npy']:
#             return Image.fromarray(np.load(filename))
#         elif ext in ['.pt', '.pth']:
#             return Image.fromarray(torch.load(filename).numpy())
#         else:
#             return Image.open(filename)
        return cv2.imread(str(filename))

    def __getitem__(self, idx):
        name = self.ids[idx]  # 1002095_2
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
#         mask = self.load(mask_file[0])
#         img = self.load(img_file[0])
        mask = cv2.imread(str(mask_file[0]))
        img = cv2.imread(str(img_file[0]))
#         print('------------str(mask_file[0])',str(mask_file[0]))
        assert img.shape == mask.shape, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        #-----------------------------------causel-----------------------------------
        img_trsp = img
        mask_trsp = mask
        #######################################################################
        #                       DATA ARGUMENTATION                            #
        #######################################################################
        if self.data_argu:
            mask_trsp = mask_trsp[np.newaxis,:,:,:]
            img_trsp, mask_trsp = ALL_transform(image=img_trsp,segmentation_maps=mask_trsp)
            mask_trsp = np.squeeze(mask_trsp)
#         print('1',img_trsp.shape)
#         print('2',img_trsp.max())
#         print('3',img_trsp.min())
        mask_trsp_binary = mask_trsp /255 
        mask_trsp_binary_bg = 1-mask_trsp_binary
        img_nuc = mask_trsp_binary* img_trsp
#         print('--------------------',img_trsp.shape)
        mask_trsp1 = mask_trsp[np.newaxis,:,:,:]
        transformed1_array, mask_trsp1 = transform(image=img_trsp, segmentation_maps=mask_trsp1)
#         print('1---------',transformed1_array.shape)
#         print('2---------',transformed1_array.max())
#         print('3---------',transformed1_array.min())
        mask_trsp1= np.squeeze(mask_trsp1)
#         transformed2_array = transform(image=img_trsp, label=mask_trsp_binary)
#         transformed3_array = transform(image=img_trsp, label=mask_trsp_binary)
        img_bg1 = mask_trsp_binary_bg* transformed1_array
#         img_bg2 = mask_trsp_binary_bg* transformed2_array["image"]
#         img_bg3 = mask_trsp_binary_bg* transformed3_array["image"]
        img_mix1 = img_bg1 + img_nuc
#         img_mix2 = img_bg2 + img_nuc
#         img_mix3 = img_bg3 + img_nuc
        
        img_trsf1 = self.preprocess(img_mix1, self.scale, is_mask=False) # 3*1024*1024
#         img_trsf2 = self.preprocess(img_mix2, self.scale, is_mask=False) # 3*1024*1024
#         img_trsf3 = self.preprocess(img_mix3, self.scale, is_mask=False) # 3*1024*1024
        #-----------------------------------causel-----------------------------------
        img_ori = self.preprocess(img, self.scale, is_mask=False)
        mask_ori = self.preprocess(mask, self.scale, is_mask=True)
        img_trsp = self.preprocess(img_trsp, self.scale, is_mask=False)
        mask_trsp1 = self.preprocess(mask_trsp1, self.scale, is_mask=True)
        
        return {
            'image': torch.as_tensor(img_ori.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask_ori.copy()).long().contiguous(),
            'image_aug': torch.as_tensor(img_trsp.copy()).float().contiguous(),
            'mask_aug': torch.as_tensor(mask_trsp1.copy()).long().contiguous(),
            'img_trsf1': torch.as_tensor(img_trsf1).float().contiguous(),
#             'img_path':img_file[0]
#             'img_trsf2': torch.as_tensor(img_trsf2).float().contiguous(),
        }


# +
# class CarvanaDataset(BasicDataset):
#     def __init__(self, images_dir, masks_dir, scale=1):
#         super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask',transform = transform)
        
class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1,data_argu=0):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_nuc_filled',data_argu=0)

# +
# class BasicTestDataset(Dataset):
#     def __init__(self, images_dir: str, masks_dir: str, dann_dir: str, scale: float = 1.0, mask_suffix: str = '_nuc_filled',data_argu: int = 0):
#         self.images_dir = Path(images_dir)
#         self.masks_dir = Path(masks_dir)
#         #===============================dann==============================================
#         self.dann_dir = Path(dann_dir)
#         #===============================dann==============================================
#         assert 0 < scale <= 1, 'Scale must be between 0 and 1'
#         self.scale = scale
#         self.mask_suffix = mask_suffix
#         self.data_argu = data_argu

#         self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
#         if not self.ids:
#             raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
#         logging.info(f'Creating dataset with {len(self.ids)} examples')

#     def __len__(self):
#         return len(self.ids)

#     @classmethod
#     def preprocess(cls, pil_img, scale, is_mask):
# #         w, h = pil_img.size
# # #         print('==============',w,h)
#         img_ndarray = pil_img
# #         newW, newH = int(scale * w), int(scale * h)
# #         assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
# #         pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
# #         img_ndarray = np.asarray(pil_img)

# #         if img_ndarray.ndim == 2 and not is_mask:
# #             img_ndarray = img_ndarray[np.newaxis, ...]
# # #         elif not is_mask:
# # #             img_ndarray = img_ndarray.transpose((2, 0, 1))
#         img_ndarray = img_ndarray.transpose((2, 0, 1)) # img_ndarray.shape (3, 1024, 1024)
# #         print('img_ndarray.shape',img_ndarray.shape)
# #         if not is_mask:
# #             img_ndarray = img_ndarray / 255
#         img_ndarray = img_ndarray / 255
#         if not is_mask:
#             return img_ndarray
#         if is_mask:
# #             print('img_ndarray[0].shape',img_ndarray[0].shape)
#             return img_ndarray[0]

#     @classmethod
#     def load(cls, filename):
#         ext = splitext(filename)[1]
#         if ext in ['.npz', '.npy']:
#             return Image.fromarray(np.load(filename))
#         elif ext in ['.pt', '.pth']:
#             return Image.fromarray(torch.load(filename).numpy())
#         else:
#             return Image.open(filename)

#     def __getitem__(self, idx):
#         name = self.ids[idx]  # 1002095_2
# #         print('name',name)
#         mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
#         img_file = list(self.images_dir.glob(name + '.*'))
#         #===============================dann=========================================
#         dann_mask_file = list(self.dann_dir.glob(name + '.*'))
# #         print('mask_file',mask_file)
# #         print('len(mask_file)',len(mask_file))
#         #===============================dann=========================================
# #         print('dann_h_mask_file',dann_h_mask_file)
        
#         assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
#         assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
#         #=================================dann=================================================
#         assert len(dann_mask_file) == 1, f'Either no image or multiple images found for the ID {name}: {dann_mask_file}'
#         #=================================dann=================================================
        
#         mask = self.load(mask_file[0])
#         img = self.load(img_file[0])
#         #===============================dann=======================================
#         dann = np.load(dann_mask_file[0])
#         dann = torch.tensor(dann, dtype=torch.int64)
# #         print('dann.type()',dann.type())
#         #===============================dann=======================================
# #         print('dann_h',dann_h)
#         assert img.size == mask.size, \
#             'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
#         #-----------------------------------causel-----------------------------------
# #         img_trsp = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR) # h*w*3
# #         mask_trsp = cv2.cvtColor(np.asarray(mask), cv2.COLOR_RGB2BGR) # h*w*3
#         img_trsp = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
#         mask_trsp = np.array(mask.getdata()).reshape(mask.size[0], mask.size[1], 3)
# #         print('==============mask_trsp.shape',mask_trsp.shape)
#         #######################################################################
#         #                       DATA ARGUMENTATION                            #
#         #######################################################################
#         if self.data_argu:
#             data_argument = ALL_transform(image=img_trsp, label=mask_trsp)
#             img_trsp = data_argument["image"]
#             mask_trsp = data_argument["label"]
        
# #         print('img_trsp.shape',img_trsp.shape)
# #         print('mask_trsp.shape',mask_trsp.shape)
# #         print('mask_trsp.max()',mask_trsp.max())
# #         print('mask_trsp.min()',mask_trsp.min())
        
#         mask_trsp_binary = mask_trsp /255 
#         mask_trsp_binary_bg = 1-mask_trsp_binary
#         img_nuc = mask_trsp_binary* img_trsp
# #         print('mask_trsp_binary.max()',mask_trsp_binary.max())
# #         print('mask_trsp_binary.min()',mask_trsp_binary.min())
        
#         transformed1_array = transform(image=img_trsp, label=mask_trsp_binary)
#         transformed2_array = transform(image=img_trsp, label=mask_trsp_binary)
#         transformed3_array = transform(image=img_trsp, label=mask_trsp_binary)
# #         print('transformed1_array',transformed1_array)
# #         print('transformed1_array["label"].max()',transformed1_array["label"].max())
# #         print('transformed1_array["label"].min()',transformed1_array["label"].min())
# #         print('transformed1["image"].shape',transformed1["image"].shape)
#         img_bg1 = mask_trsp_binary_bg* transformed1_array["image"]
#         img_bg2 = mask_trsp_binary_bg* transformed2_array["image"]
#         img_bg3 = mask_trsp_binary_bg* transformed3_array["image"]
#         img_mix1 = img_bg1 + img_nuc
#         img_mix2 = img_bg2 + img_nuc
#         img_mix3 = img_bg3 + img_nuc
        
#         #=====================1024*1024*3 => 3*1024*1024
#         to_tensor_array1 = to_tensor(image=img_mix1, label=mask_trsp) 
#         to_tensor_array2 = to_tensor(image=img_mix2, label=mask_trsp)
#         to_tensor_array3 = to_tensor(image=img_mix3, label=mask_trsp)
# #         cv2.imwrite("./11111/x"+name+".png",img_mix1)
# #         cv2.imwrite("./11111/y"+name+".png",img_mix2)
# #         cv2.imwrite("./11111/z"+name+".png",img_mix3)
# #         print('to_tensor_array["image"].size()',to_tensor_array["image"].size())
        
#         img_trsf1 = to_tensor_array1["image"] # 3*1024*1024
#         img_trsf2 = to_tensor_array2["image"] # 3*1024*1024
#         img_trsf3 = to_tensor_array3["image"] # 3*1024*1024
# #         print('=======img_trsf.shape',img_trsf.shape)
#         #-----------------------------------causel-----------------------------------

        
#         img_trsp = self.preprocess(img_trsp, self.scale, is_mask=False)
#         mask_trsp = self.preprocess(mask_trsp, self.scale, is_mask=True)
# #         print('img.shape',img.shape)
# #         print('img.shape',img.shape)
# #         print('mask.shape',mask.shape)
        

#         return {
#             'image': torch.as_tensor(img_trsp.copy()).float().contiguous(),
#             'mask': torch.as_tensor(mask_trsp.copy()).long().contiguous(),
# #             'img_trsf1': torch.as_tensor(img_trsf1).float().contiguous(),
# #             'img_trsf2': torch.as_tensor(img_trsf2).float().contiguous(),
# #             'img_trsf3': torch.as_tensor(img_trsf3).float().contiguous(),
#             #===============================dann==============================================
#             'dann':  torch.as_tensor(dann).contiguous(),
#             #===============================dann==============================================
#         }

# +
# class CarvanaTestDataset(BasicTestDataset):
#     def __init__(self, images_dir, masks_dir, dann_dir, scale=1, data_argu=0):
#         super().__init__(images_dir, masks_dir, dann_dir, scale, mask_suffix='_nuc_filled',data_argu=0)
