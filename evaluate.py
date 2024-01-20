import torch
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
import torch.nn as nn

from utils.dice_score import multiclass_dice_coeff, dice_coeff_19


# +
def evaluate_all(net, dataloader, device, train_ = 0):
    net.eval()
    num_val_batches = len(dataloader)
    if num_val_batches ==0:
        num_val_batches = 1
    iou_all = 0
    precision_all = 0
    recall_all =0
    f1_all = 0
    PRCC_all = 0
    miou_all = 0
#     print('num_val_batches',num_val_batches)
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        if train_ == 1:
            image, mask_true = batch['image_aug'], batch['mask_aug']
        else:
            image, mask_true = batch['image'], batch['mask']
#         print('dann_true',dann_true)
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = mask_true.to(device=device,)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
        EPSILON = 1e-15
        
        with torch.no_grad():
            # predict the mask
#             print('net')
            model = nn.DataParallel(net)
            model.to(device=device)
            mask_pred, x5, _, __ = model(image)
            
#             print('=================mask_pred.size()',mask_pred.size()) #([4, 2, 1024, 1024])
            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                
                tp =  torch.sum(mask_pred.int() & mask_true[:, 1, ...].int())
                tn =  torch.sum((1 -  mask_true[:, 1, ...].int()) * (1 - mask_pred.int()))
                fp =  torch.sum((1 - mask_true[:, 1, ...].int()) * mask_pred.int())
                fn =  torch.sum(mask_true[:, 1, ...] * (1 - mask_pred.int()))
                precision = tp / (tp + fp + EPSILON)
                recall = tp / (tp + fn + EPSILON)
                f1 = 2* (precision*recall) / (precision + recall + EPSILON)
                PRCC = 2* torch.sqrt(precision*recall) / (precision + recall + EPSILON)
                precision_all += precision
                recall_all +=recall
                f1_all += f1
                PRCC_all += PRCC
                
                intersection = (mask_true * mask_pred).sum()
                union = mask_true.sum() + mask_pred.sum() - intersection
                iou = (intersection + EPSILON) / (union + EPSILON)
                iou_all += iou
                
                iou_n =tp/(tp+ fp+ fn)
                iou_bg =tn/(tn+ fp+ fn)
                miou = 0.5*(iou_n+ iou_bg)
                miou_all += miou
            else:
                mask_pred = mask_pred.argmax(dim=1)
                tp =  torch.sum(mask_pred.int() & mask_true[:, 1, ...].int())
                tn =  torch.sum((1 -  mask_true[:, 1, ...].int()) * (1 - mask_pred.int()))
                fp =  torch.sum((1 - mask_true[:, 1, ...].int()) * mask_pred.int())
                fn =  torch.sum(mask_true[:, 1, ...] * (1 - mask_pred.int()))
                precision = tp / (tp + fp + EPSILON)
                recall = tp / (tp + fn + EPSILON)
                f1 = 2* (precision*recall) / (precision + recall + EPSILON)
                PRCC = 2* torch.sqrt(precision*recall) / (precision + recall + EPSILON)
                precision_all += precision
                recall_all +=recall
                f1_all += f1
                PRCC_all += PRCC
                
                intersection = (mask_true[:, 1, ...] * mask_pred).sum()
                union =  mask_true[:, 1, ...].sum() + mask_pred.sum() - intersection
                iou = (intersection + EPSILON) / (union + EPSILON)
                iou_all += iou
                
                iou_n =tp/(tp+ fp+ fn)
                iou_bg =tn/(tn+ fp+ fn)
                
                miou = 0.5*(iou_n+ iou_bg)
                miou_all += miou
                
    precision_avg = precision_all / num_val_batches
    recall_avg =recall_all /num_val_batches
    f1_avg = f1_all /num_val_batches
    PRCC_avg = PRCC_all /num_val_batches
    
    
    iou_avg = iou_all / num_val_batches     
    
    miou_avg = miou_all/ num_val_batches   
#     print('num_val_batches',num_val_batches)

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return iou_avg, f1_avg , PRCC_avg , precision_avg , recall_avg, miou_avg
    return iou_avg, f1_avg , PRCC_avg , precision_avg , recall_avg, miou_avg


# -

def evaluate_all_ensemble(net1, net2, ensemble_net, dataloader, device, train_ = 0):
    net1.eval()
    net2.eval()
    ensemble_net.eval()
    num_val_batches = len(dataloader)
    if num_val_batches ==0:
        num_val_batches = 1
    iou_all1 = 0
    precision_all1 = 0
    recall_all1 =0
    f1_all1 = 0
    PRCC_all1 = 0
    miou_all1 = 0
    
    iou_all2 = 0
    precision_all2 = 0
    recall_all2 =0
    f1_all2 = 0
    PRCC_all2 = 0
    miou_all2 = 0
    
    iou_all3 = 0
    precision_all3 = 0
    recall_all3 =0
    f1_all3 = 0
    PRCC_all3= 0
    miou_all3 = 0
    with torch.no_grad():
        net1.ResUnetPlusPlus_FE.infodrop_1.mode = 0
        net2.ResUnetPlusPlus_FE.infodrop_1.mode = 0
        net1 = nn.DataParallel(net1,device_ids = [ 3, 1, 2, 0]) #device_ids=[0, 1, 2]
        net2 = nn.DataParallel(net2,device_ids = [ 2, 1, 0, 3 ]) #device_ids=[0, 1, 2]
        ensemble_net = nn.DataParallel(ensemble_net,device_ids = [ 1, 0, 2, 3])
        net1.to(device=f'cuda:{net1.device_ids[0]}')
        net2.to(device=f'cuda:{net2.device_ids[0]}')
        ensemble_net.to(device=f'cuda:{ensemble_net.device_ids[0]}')
#     print('num_val_batches',num_val_batches)
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
#         print(net1.module.ResUnetPlusPlus_FE.infodrop_1.mode )
        if train_ == 1:
            image, mask_true = batch['image_aug'], batch['mask_aug']
        else:
            image, mask_true = batch['image'], batch['mask']
#         image, mask_true = batch['image'], batch['mask']
#         print('dann_true',dann_true)
        # move images and labels to correct device and type
        with torch.no_grad():
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=f'cuda:{net1.device_ids[0]}', dtype=torch.long)
            mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()
            EPSILON = 1e-15

            mask_pred1, x5, _, __ = net1(image.to(device=f'cuda:{net1.device_ids[0]}'))
            mask_pred2, x5, _, __ = net2(image.to(device=f'cuda:{net2.device_ids[0]}'))
            masks_mix = torch.cat((mask_pred1.to(device=f'cuda:{ensemble_net.device_ids[0]}'), mask_pred2.to(device=f'cuda:{ensemble_net.device_ids[0]}')), 1)
            masks_final = ensemble_net(masks_mix)
            
        mask_pred1 = mask_pred1.argmax(dim=1)
        mask_pred2 = mask_pred2.argmax(dim=1)
        mask_pred3 = masks_final.argmax(dim=1)
        
        tp1 =  torch.sum(mask_pred1.int() & mask_true[:, 1, ...].int())
        tn1 =  torch.sum((1 -  mask_true[:, 1, ...].int()) * (1 - mask_pred1.int()))
        fp1 =  torch.sum((1 - mask_true[:, 1, ...].int()) * mask_pred1.int())
        fn1 =  torch.sum(mask_true[:, 1, ...] * (1 - mask_pred1.int()))
        precision1 = tp1 / (tp1 + fp1 + EPSILON)
        recall1 = tp1 / (tp1 + fn1 + EPSILON)
        f11 = 2* (precision1*recall1) / (precision1 + recall1 + EPSILON)
        PRCC1 = 2* torch.sqrt(precision1*recall1) / (precision1 + recall1 + EPSILON)
        precision_all1 += precision1
        recall_all1 +=recall1
        f1_all1 += f11
        PRCC_all1 += PRCC1
        intersection1 = (mask_true[:, 1, ...] * mask_pred1).sum()
        union1 =  mask_true[:, 1, ...].sum() + mask_pred1.sum() - intersection1
        iou1 = (intersection1 + EPSILON) / (union1 + EPSILON)
        iou_all1 += iou1
        iou_n1 =tp1/(tp1+ fp1+ fn1)
        iou_bg1 =tn1/(tn1+ fp1+ fn1)
        miou1 = 0.5*(iou_n1+ iou_bg1)
        miou_all1 += miou1
        
        mask_true = mask_true.to(device=f'cuda:{net2.device_ids[0]}', dtype=torch.long)
        tp2 =  torch.sum(mask_pred2.int() & mask_true[:, 1, ...].int())
        tn2 =  torch.sum((1 -  mask_true[:, 1, ...].int()) * (1 - mask_pred2.int()))
        fp2 =  torch.sum((1 - mask_true[:, 1, ...].int()) * mask_pred2.int())
        fn2 =  torch.sum(mask_true[:, 1, ...] * (1 - mask_pred2.int()))
        precision2 = tp2 / (tp2 + fp2 + EPSILON)
        recall2 = tp2 / (tp2 + fn2 + EPSILON)
        f12 = 2* (precision2*recall2) / (precision2 + recall2 + EPSILON)
        PRCC2 = 2* torch.sqrt(precision2*recall2) / (precision2 + recall2 + EPSILON)
        precision_all2 += precision2
        recall_all2 +=recall2
        f1_all2 += f12
        PRCC_all2 += PRCC2
        intersection2 = (mask_true[:, 1, ...] * mask_pred2).sum()
        union2 =  mask_true[:, 1, ...].sum() + mask_pred2.sum() - intersection2
        iou2 = (intersection2 + EPSILON) / (union2 + EPSILON)
        iou_all2 += iou2
        iou_n2 =tp2/(tp2+ fp2+ fn2)
        iou_bg2 =tn2/(tn2+ fp2+ fn2)
        miou2 = 0.5*(iou_n2+ iou_bg2)
        miou_all2 += miou2
        
        mask_true = mask_true.to(device=f'cuda:{ensemble_net.device_ids[0]}', dtype=torch.long)
        tp3 =  torch.sum(mask_pred3.int() & mask_true[:, 1, ...].int())
        tn3 =  torch.sum((1 -  mask_true[:, 1, ...].int()) * (1 - mask_pred3.int()))
        fp3 =  torch.sum((1 - mask_true[:, 1, ...].int()) * mask_pred3.int())
        fn3 =  torch.sum(mask_true[:, 1, ...] * (1 - mask_pred3.int()))
        precision3 = tp3 / (tp3 + fp3 + EPSILON)
        recall3 = tp3 / (tp3 + fn3 + EPSILON)
        f13 = 2* (precision3*recall3) / (precision3 + recall3 + EPSILON)
        PRCC3 = 2* torch.sqrt(precision3*recall3) / (precision3 + recall3 + EPSILON)
        precision_all3 += precision3
        recall_all3 +=recall3
        f1_all3 += f13
        PRCC_all3 += PRCC3
        intersection3 = (mask_true[:, 1, ...] * mask_pred3).sum()
        union3 =  mask_true[:, 1, ...].sum() + mask_pred3.sum() - intersection3
        iou3 = (intersection3 + EPSILON) / (union3 + EPSILON)
        iou_all3 += iou3
        iou_n3 =tp3/(tp3+ fp3+ fn3)
        iou_bg3 =tn3/(tn3+ fp3+ fn3)
        miou3 = 0.5*(iou_n3+ iou_bg3)
        miou_all3 += miou3
    
    precision_avg1 = precision_all1 / num_val_batches
    recall_avg1 =recall_all1 /num_val_batches
    f1_avg1 = f1_all1 /num_val_batches
    PRCC_avg1 = PRCC_all1 /num_val_batches
    iou_avg1 = iou_all1 / num_val_batches     
    miou_avg1 = miou_all1/ num_val_batches  
    
    precision_avg2 = precision_all2 / num_val_batches
    recall_avg2 =recall_all2 /num_val_batches
    f1_avg2 = f1_all2 /num_val_batches
    PRCC_avg2 = PRCC_all2 /num_val_batches
    iou_avg2 = iou_all2 / num_val_batches     
    miou_avg2 = miou_all2 / num_val_batches  
    
    precision_avg3 = precision_all3 / num_val_batches
    recall_avg3 =recall_all3 /num_val_batches
    f1_avg3 = f1_all3 /num_val_batches
    PRCC_avg3 = PRCC_all3 /num_val_batches
    iou_avg3 = iou_all3 / num_val_batches     
    miou_avg3 = miou_all3/ num_val_batches   
#     print('num_val_batches',num_val_batches)
#     print(net1.module.ResUnetPlusPlus_FE.infodrop_1.mode )
    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return iou_avg1, f1_avg1, PRCC_avg1 , precision_avg1 , recall_avg1, miou_avg1 \
            ,iou_avg2, f1_avg2, PRCC_avg2 , precision_avg2 , recall_avg2, miou_avg2 \
            ,iou_avg3, f1_avg3, PRCC_avg3 , precision_avg3 , recall_avg3, miou_avg3
    return iou_avg1, f1_avg1, PRCC_avg1 , precision_avg1 , recall_avg1, miou_avg1 \
        ,iou_avg2, f1_avg2, PRCC_avg2 , precision_avg2 , recall_avg2, miou_avg2 \
        ,iou_avg3, f1_avg3, PRCC_avg3 , precision_avg3 , recall_avg3, miou_avg3


# +
def evaluate_all_net1(net, dataloader, device, train_ = 0):
    net.eval()
    num_val_batches = len(dataloader)
    if num_val_batches ==0:
        num_val_batches = 1
    iou_all = 0
    precision_all = 0
    recall_all =0
    f1_all = 0
    PRCC_all = 0
    miou_all = 0
    with torch.no_grad():
        net.ResUnetPlusPlus_FE.infodrop_1.mode = 0
        net = nn.DataParallel(net1,device_ids = [ 7, 1, 2, 3, 4, 5, 6, 0]) #device_ids=[0, 1, 2]
        net1.to(device=f'cuda:{net1.device_ids[0]}')
#     print('num_val_batches',num_val_batches)
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        if train_ == 1:
            image, mask_true = batch['image_aug'], batch['mask_aug']
        else:
            image, mask_true = batch['image'], batch['mask']
#         print('dann_true',dann_true)
        # move images and labels to correct device and type
        image = image.to(device=f'cuda:{net1.device_ids[0]}', dtype=torch.float32)
        mask_true = mask_true.to(device=f'cuda:{net1.device_ids[0]}', dtype=torch.long)
        mask_true = mask_true.to(device=f'cuda:{net1.device_ids[0]}',)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
        EPSILON = 1e-15
        

            
#             print('=================mask_pred.size()',mask_pred.size()) #([4, 2, 1024, 1024])
            # convert to one-hot format
        if net.n_classes == 1:
            mask_pred = (F.sigmoid(mask_pred) > 0.5).float()

            tp =  torch.sum(mask_pred.int() & mask_true[:, 1, ...].int())
            tn =  torch.sum((1 -  mask_true[:, 1, ...].int()) * (1 - mask_pred.int()))
            fp =  torch.sum((1 - mask_true[:, 1, ...].int()) * mask_pred.int())
            fn =  torch.sum(mask_true[:, 1, ...] * (1 - mask_pred.int()))
            precision = tp / (tp + fp + EPSILON)
            recall = tp / (tp + fn + EPSILON)
            f1 = 2* (precision*recall) / (precision + recall + EPSILON)
            PRCC = 2* torch.sqrt(precision*recall) / (precision + recall + EPSILON)
            precision_all += precision
            recall_all +=recall
            f1_all += f1
            PRCC_all += PRCC

            intersection = (mask_true * mask_pred).sum()
            union = mask_true.sum() + mask_pred.sum() - intersection
            iou = (intersection + EPSILON) / (union + EPSILON)
            iou_all += iou

            iou_n =tp/(tp+ fp+ fn)
            iou_bg =tn/(tn+ fp+ fn)
            miou = 0.5*(iou_n+ iou_bg)
            miou_all += miou
        else:
            mask_pred = mask_pred.argmax(dim=1)
            tp =  torch.sum(mask_pred.int() & mask_true[:, 1, ...].int())
            tn =  torch.sum((1 -  mask_true[:, 1, ...].int()) * (1 - mask_pred.int()))
            fp =  torch.sum((1 - mask_true[:, 1, ...].int()) * mask_pred.int())
            fn =  torch.sum(mask_true[:, 1, ...] * (1 - mask_pred.int()))
            precision = tp / (tp + fp + EPSILON)
            recall = tp / (tp + fn + EPSILON)
            f1 = 2* (precision*recall) / (precision + recall + EPSILON)
            PRCC = 2* torch.sqrt(precision*recall) / (precision + recall + EPSILON)
            precision_all += precision
            recall_all +=recall
            f1_all += f1
            PRCC_all += PRCC

            intersection = (mask_true[:, 1, ...] * mask_pred).sum()
            union =  mask_true[:, 1, ...].sum() + mask_pred.sum() - intersection
            iou = (intersection + EPSILON) / (union + EPSILON)
            iou_all += iou

            iou_n =tp/(tp+ fp+ fn)
            iou_bg =tn/(tn+ fp+ fn)

            miou = 0.5*(iou_n+ iou_bg)
            miou_all += miou
                
    precision_avg = precision_all / num_val_batches
    recall_avg =recall_all /num_val_batches
    f1_avg = f1_all /num_val_batches
    PRCC_avg = PRCC_all /num_val_batches
    
    iou_avg = iou_all / num_val_batches     
    
    miou_avg = miou_all/ num_val_batches   
#     print('num_val_batches',num_val_batches)

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return iou_avg, f1_avg, PRCC_avg , precision_avg , recall_avg, miou_avg
    return iou_avg, f1_avg, PRCC_avg , precision_avg , recall_avg, miou_avg


# +
def evaluate_all_net2(net, dataloader, device, train_ = 0):
    net.eval()
    num_val_batches = len(dataloader)
    if num_val_batches ==0:
        num_val_batches = 1
    iou_all = 0
    precision_all = 0
    recall_all =0
    f1_all = 0
    PRCC_all = 0
    miou_all = 0
    with torch.no_grad():
        net2.ResUnetPlusPlus_FE.infodrop_1.mode = 0
        net2 = nn.DataParallel(net2,device_ids = [ 7, 1, 2, 3, 4, 5, 6, 0]) #device_ids=[0, 1, 2]
        net2.to(device=f'cuda:{net2.device_ids[0]}')
#     print('num_val_batches',num_val_batches)
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        if train_ == 1:
            image, mask_true = batch['image_aug'], batch['mask_aug']
        else:
            image, mask_true = batch['image'], batch['mask']
#         print('dann_true',dann_true)
        # move images and labels to correct device and type
        image = image.to(device=f'cuda:{net2.device_ids[0]}', dtype=torch.float32)
        mask_true = mask_true.to(device=f'cuda:{net2.device_ids[0]}', dtype=torch.long)
        mask_true = mask_true.to(device=f'cuda:{net2.device_ids[0]}',)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
        EPSILON = 1e-15
        

            
#             print('=================mask_pred.size()',mask_pred.size()) #([4, 2, 1024, 1024])
            # convert to one-hot format
        if net.n_classes == 1:
            mask_pred = (F.sigmoid(mask_pred) > 0.5).float()

            tp =  torch.sum(mask_pred.int() & mask_true[:, 1, ...].int())
            tn =  torch.sum((1 -  mask_true[:, 1, ...].int()) * (1 - mask_pred.int()))
            fp =  torch.sum((1 - mask_true[:, 1, ...].int()) * mask_pred.int())
            fn =  torch.sum(mask_true[:, 1, ...] * (1 - mask_pred.int()))
            precision = tp / (tp + fp + EPSILON)
            recall = tp / (tp + fn + EPSILON)
            f1 = 2* (precision*recall) / (precision + recall + EPSILON)
            PRCC = 2* torch.sqrt(precision*recall) / (precision + recall + EPSILON)
            precision_all += precision
            recall_all +=recall
            f1_all += f1
            PRCC_all += PRCC

            intersection = (mask_true * mask_pred).sum()
            union = mask_true.sum() + mask_pred.sum() - intersection
            iou = (intersection + EPSILON) / (union + EPSILON)
            iou_all += iou

            iou_n =tp/(tp+ fp+ fn)
            iou_bg =tn/(tn+ fp+ fn)
            miou = 0.5*(iou_n+ iou_bg)
            miou_all += miou
        else:
            mask_pred = mask_pred.argmax(dim=1)
            tp =  torch.sum(mask_pred.int() & mask_true[:, 1, ...].int())
            tn =  torch.sum((1 -  mask_true[:, 1, ...].int()) * (1 - mask_pred.int()))
            fp =  torch.sum((1 - mask_true[:, 1, ...].int()) * mask_pred.int())
            fn =  torch.sum(mask_true[:, 1, ...] * (1 - mask_pred.int()))
            precision = tp / (tp + fp + EPSILON)
            recall = tp / (tp + fn + EPSILON)
            f1 = 2* (precision*recall) / (precision + recall + EPSILON)
            PRCC = 2* torch.sqrt(precision*recall) / (precision + recall + EPSILON)
            precision_all += precision
            recall_all +=recall
            f1_all += f1
            PRCC_all += PRCC

            intersection = (mask_true[:, 1, ...] * mask_pred).sum()
            union =  mask_true[:, 1, ...].sum() + mask_pred.sum() - intersection
            iou = (intersection + EPSILON) / (union + EPSILON)
            iou_all += iou

            iou_n =tp/(tp+ fp+ fn)
            iou_bg =tn/(tn+ fp+ fn)

            miou = 0.5*(iou_n+ iou_bg)
            miou_all += miou
                
    precision_avg = precision_all / num_val_batches
    recall_avg =recall_all /num_val_batches
    f1_avg = f1_all /num_val_batches
    PRCC_avg = PRCC_all /num_val_batches
    
    iou_avg = iou_all / num_val_batches     
    
    miou_avg = miou_all/ num_val_batches   
#     print('num_val_batches',num_val_batches)

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return iou_avg, f1_avg, PRCC_avg , precision_avg , recall_avg, miou_avg
    return iou_avg, f1_avg, PRCC_avg , precision_avg , recall_avg, miou_avg


# -

def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
#         EPSILON = 1e-15
        
        with torch.no_grad():
            # predict the mask
            mask_pred, dann, x5 = net(image)
            
            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
#                 # compute the Dice score
#                 dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
#                 intersection = (mask_true * mask_pred).sum()
#                 union = mask_true.sum() + mask_pred.sum() - intersection
#                 iou = (intersection + EPSILON) / (union + EPSILON)
            else:
#                 print('mask_pred.size()',mask_pred.size())
#                 print('net.n_classes',net.n_classes)
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
#                 print('____mask_pred.size()',mask_pred.size())
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
                
           

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches


# +
# def evaluate_iou(net, dataloader, device):
#     net.eval()
#     num_val_batches = len(dataloader)

#     # iterate over the validation set
#     for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
#         image, mask_true = batch['image'], batch['mask']
#         # move images and labels to correct device and type
#         image = image.to(device=device, dtype=torch.float32)
#         mask_true = mask_true.to(device=device, dtype=torch.long)
#         mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
#         EPSILON = 1e-15
        
#         with torch.no_grad():
#             # predict the mask
#             mask_pred, h, s, l, x5 = net(image)
# #             print('=================mask_pred.size()',mask_pred.size()) #([4, 2, 1024, 1024])
#             # convert to one-hot format
#             if net.n_classes == 1:
#                 mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
#                 # compute the Dice score
# #                 dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
#                 intersection = (mask_true * mask_pred).sum()
#                 union = mask_true.sum() + mask_pred.sum() - intersection
#                 iou = (intersection + EPSILON) / (union + EPSILON)
#             else:
# #                 print('mask_pred.size()',mask_pred.size())
# #                 print('net.n_classes',net.n_classes)
# #                 mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
# #                 print('mask_pred.size()',mask_pred.size())
# #                 mask_pred = torch.softmax(mask_pred, dim=1).argmax(dim=1)
#                 mask_pred = mask_pred.argmax(dim=1)
# #                 print('mask_pred.size()',mask_pred.size())
# #                 print('mask_true[:, 1, ...].sum()',mask_true[:, 1, ...].sum())
# #                 print('__mask_pred.size()',mask_pred.size())
# #                 print('mask_pred.size()',mask_pred.size()) # [4, 1024, 1024]
# #                 print('mask_true[:, 1, ...].size()',mask_true[:, 1, ...].size()) # [4, 1024, 1024]size())= 
#                 # compute the Dice score, ignoring background
# #                 dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
                
#                 intersection = (mask_true[:, 1, ...] * mask_pred).sum()
# #                 print('mask_true[:, 1, ...].sum()',mask_true[:, 1, ...].sum())
# #                 print('mask_pred.sum()',mask_pred.sum())
#                 union =  mask_true[:, 1, ...].sum() + mask_pred.sum() - intersection
# #                 print('intersection',intersection)
# #                 print('union',union)
#                 iou = (intersection + EPSILON) / (union + EPSILON)
# #                 print('iou',iou)
                
#     net.train()

#     # Fixes a potential division by zero error
#     if num_val_batches == 0:
#         return iou
#     return iou 
# -

def evaluate_iou(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    iou_all = 0
#     print('num_val_batches',num_val_batches)
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
        EPSILON = 1e-15
        
        with torch.no_grad():
            # predict the mask
            mask_pred, dann, x5 = net(image)
#             print('=================mask_pred.size()',mask_pred.size()) #([4, 2, 1024, 1024])
            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
#                 dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                intersection = (mask_true * mask_pred).sum()
                union = mask_true.sum() + mask_pred.sum() - intersection
                iou = (intersection + EPSILON) / (union + EPSILON)
                iou_all += iou
            else:
#                 print('mask_pred.size()',mask_pred.size())
#                 print('net.n_classes',net.n_classes)
#                 mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
#                 print('mask_pred.size()',mask_pred.size())
#                 mask_pred = torch.softmax(mask_pred, dim=1).argmax(dim=1)
                mask_pred = mask_pred.argmax(dim=1)
#                 print('mask_pred.size()',mask_pred.size())
#                 print('mask_true[:, 1, ...].sum()',mask_true[:, 1, ...].sum())
#                 print('__mask_pred.size()',mask_pred.size())
#                 print('mask_pred.size()',mask_pred.size()) # [4, 1024, 1024]
#                 print('mask_true[:, 1, ...].size()',mask_true[:, 1, ...].size()) # [4, 1024, 1024]size())= 
                # compute the Dice score, ignoring background
#                 dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
                
                intersection = (mask_true[:, 1, ...] * mask_pred).sum()
#                 print('mask_true[:, 1, ...].sum()',mask_true[:, 1, ...].sum())
#                 print('mask_pred.sum()',mask_pred.sum())
                union =  mask_true[:, 1, ...].sum() + mask_pred.sum() - intersection
#                 print('intersection',intersection)
#                 print('union',union)
                iou = (intersection + EPSILON) / (union + EPSILON)
                iou_all += iou
#                 print('iou',iou)
#                 print('iou_all',iou_all)
    iou_avg = iou_all / num_val_batches     
    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return iou_avg
    return iou_avg 


def evaluate_f1(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    precision_all = 0
    recall_all =0
    f1_all = 0
    PRCC_all = 0
#     print('num_val_batches',num_val_batches)
    
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
        EPSILON = 1e-15
        
        with torch.no_grad():
            # predict the mask
            mask_pred, dann, x5 = net(image)
#             print('=================mask_pred.size()',mask_pred.size()) #([4, 2, 1024, 1024])
            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                tp =  torch.sum(mask_pred.int() & mask_true[:, 1, ...].int())
                tn =  torch.sum((1 -  mask_true[:, 1, ...].int()) * (1 - mask_pred.int()))
                fp =  torch.sum((1 - mask_true[:, 1, ...].int()) * mask_pred.int())
                fn =  torch.sum(mask_true[:, 1, ...] * (1 - mask_pred.int()))
#                 print('tp',tp)
#                 print('tn',tn)
#                 print('fp',fp)
#                 print('fn',fn)
                
                precision = tp / (tp + fp + EPSILON)
                recall = tp / (tp + fn + EPSILON)
#                 f1 = 2 *tp / (tp+ fp + fn + EPSILON)
                f1 = 2* (precision*recall) / (precision + recall + EPSILON)
                PRCC = 2* torch.sqrt(precision*recall) / (precision + recall + EPSILON)
#                 print('tp',tp)
#                 print('tn',tn)
#                 print('fp',fp)
#                 print('fn',fn)
                precision_all += precision
                recall_all +=recall
                f1_all += f1
                PRCC_all += PRCC
            else:
                mask_pred = mask_pred.argmax(dim=1)
                
                tp =  torch.sum(mask_pred.int() & mask_true[:, 1, ...].int())
                tn =  torch.sum((1 -  mask_true[:, 1, ...].int()) * (1 - mask_pred.int()))
                fp =  torch.sum((1 - mask_true[:, 1, ...].int()) * mask_pred.int())
                fn =  torch.sum(mask_true[:, 1, ...] * (1 - mask_pred.int()))
#                 print('tp',tp)
#                 print('tn',tn)
#                 print('fp',fp)
#                 print('fn',fn)
                
                precision = tp / (tp + fp + EPSILON)
                recall = tp / (tp + fn + EPSILON)
#                 f1 = 2 *tp / (tp+ fp + fn + EPSILON)
                f1 = 2* (precision*recall) / (precision + recall + EPSILON)
                PRCC = 2* torch.sqrt(precision*recall) / (precision + recall + EPSILON)
#                 print('tp',tp)
#                 print('tn',tn)
#                 print('fp',fp)
#                 print('fn',fn)
                precision_all += precision
                recall_all +=recall
                f1_all += f1
                PRCC_all += PRCC
#                 print('precision',precision)
#                 print('recall',recall)
#                 print('f1',f1)
#                 print('precision',precision_all)
#                 print('recall',recall_all)
#                 print('f1',f1_all)
        
    precision_avg = precision_all / num_val_batches
    recall_avg =recall_all /num_val_batches
    f1_avg = f1_all /num_val_batches
    PRCC_avg = PRCC__all /num_val_batches
    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return f1_avg, PRCC_avg , precision_avg , recall_avg 
    return f1_avg, PRCC_avg , precision_avg , recall_avg 


def evaluate_miou(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    miou_all = 0
    
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
        EPSILON = 1e-15
        
        with torch.no_grad():
            # predict the mask
            mask_pred, dann, x5 = net(image)
#             print('=================mask_pred.size()',mask_pred.size()) #([4, 2, 1024, 1024])
            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                tp =  torch.sum(mask_pred.int() & mask_true[:, 1, ...].int())
                tn =  torch.sum((1 -  mask_true[:, 1, ...].int()) * (1 - mask_pred.int()))
                fp =  torch.sum((1 - mask_true[:, 1, ...].int()) * mask_pred.int())
                fn =  torch.sum(mask_true[:, 1, ...] * (1 - mask_pred.int()))
                
                iou_n =tp/(tp+ fp+ fn)
                iou_bg =tn/(tn+ fp+ fn)
                
                miou = 0.5*(iou_n+ iou_bg)
                miou_all += miou
            else:
                mask_pred = mask_pred.argmax(dim=1)
                
                tp =  torch.sum(mask_pred.int() & mask_true[:, 1, ...].int())
                tn =  torch.sum((1 -  mask_true[:, 1, ...].int()) * (1 - mask_pred.int()))
                fp =  torch.sum((1 - mask_true[:, 1, ...].int()) * mask_pred.int())
                fn =  torch.sum(mask_true[:, 1, ...] * (1 - mask_pred.int()))
                
                iou_n =tp/(tp+ fp+ fn)
                iou_bg =tn/(tn+ fp+ fn)
                
                miou = 0.5*(iou_n+ iou_bg)
                miou_all += miou
#                 precision = tp / (tp + fp + EPSILON)
#                 recall = tp / (tp + fn + EPSILON)
#                 f1 = 2* (precision*recall) / (precision + recall + EPSILON)
#                 print('miou_all',miou_all)
#                 print('miou',miou)
#                 print('fp',fp)
#                 print('fn',fn)
#                 print('precision',precision)
#                 print('recall',recall)
#                 print('f1',f1)
    miou_avg = miou_all/ num_val_batches     
    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return miou_avg
    return miou_avg 


def evaluate_iou_inf( mask_pred, mask_gt):

    EPSILON = 1e-15
    intersection = (mask_gt * mask_pred).sum()
    union =  mask_gt.sum() + mask_pred.sum() - intersection
    iou = (intersection + EPSILON) / (union + EPSILON)

    return iou 


def evaluate_miou_inf( mask_pred, mask_gt):

    EPSILON = 1e-15
    tp =  (mask_pred & mask_gt).sum()
    tn =  ((1 -  mask_gt) * (1 - mask_pred)).sum()
    fp =  ((1 - mask_gt) * mask_pred).sum()
    fn =  (mask_gt * (1 - mask_pred)).sum()
#     print('tp',tp)
#     print('tn',tn)   
#     print('fp',fp)   
#     print('fn',fn)   
    iou_n =tp/(tp+ fp+ fn)
    iou_bg =tn/(tn+ fp+ fn)
                
    miou = 0.5*(iou_n+ iou_bg)
    

    return miou 


def evaluate_f1_inf( mask_pred, mask_gt):

    EPSILON = 1e-15
    tp =  (mask_gt & mask_pred).sum()
    tn =  ((1 -  mask_gt) * (1 - mask_pred)).sum()
    fp =  ((1 - mask_gt) * mask_pred).sum()
    fn =  (mask_gt * (1 - mask_pred)).sum()

    precision = tp / (tp + fp + EPSILON)
    recall = tp / (tp + fn + EPSILON)
    f1 = 2*tp/(2*tp+fp+fn)
    PRCC = 2*torch.sqrt(tp)/(2*tp+fp+fn)
#     f1 = 2* (precision*recall) / (precision + recall + EPSILON)
    return f1, PRCC, precision, recall 


def evaluate_all_inf(mask_pred, mask_gt):
    EPSILON = 1e-15
    mask_pred = np.where(mask_pred>0,1,0)
    mask_gt = np.where(mask_gt>0,1,0)
    tp =  np.sum(mask_pred & mask_gt)
    tn =  np.sum((1 -  mask_gt) * (1 - mask_pred))
    fp =  np.sum((1 - mask_gt) * mask_pred)
    fn =  np.sum(mask_gt * (1 - mask_pred))
    precision = tp / (tp + fp + EPSILON)
    recall = tp / (tp + fn + EPSILON)
    f1 = 2* (precision*recall) / (precision + recall + EPSILON)
    PRCC = 2* torch.sqrt(precision*recall) / (precision + recall + EPSILON)

    intersection = (mask_gt * mask_pred).sum()
    union =  mask_gt.sum() + mask_pred.sum() - intersection
    iou = (intersection + EPSILON) / (union + EPSILON)

    iou_n =tp/(tp+ fp+ fn + EPSILON)
    iou_bg =tn/(tn+ fp+ fn + EPSILON)

    miou = 0.5*(iou_n+ iou_bg)

    
    
    return iou, f1, PRCC , precision , recall, miou
