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

from utils.data_loading import BasicDataset, CarvanaDataset#, CarvanaTestDataset, BasicTestDataset
from utils.dice_score import *
from evaluate import *
from unet.unet_model import *

# +
learning_rate_default = 0.001 #0.0001
para_causel = 0.5
para_shape = 1
data_argu = 1
if data_argu:
    CA = 'US'
else:
    CA = ''
    
# optim_select = "RMSprop" "Adamax" "SGD" "Adadelta"  "LBFGS"
optim_select ="RMSprop" 
# sche_select=  "CosineAnnealingLR" "ReduceLROnPlateau" "LambdaLR" "StepLR"  "ExponentialLR" "CyclicLR" "OneCycleLR"
sche_select= "CosineAnnealingLR"
# train_data = 'part_3d_leica_hamamasu'
train_data = 'hamamasu_leica'
test_data = '3d'
# train_data = '3d_leica'
# test_data = 'hamamasu'
# train_data = '3d_hamamasu'
# test_data = 'leica'

criterion = nn.CrossEntropyLoss()
# causel_criterion = torch.nn.HuberLoss()
causel_criterion = js_div

def calculate_polygon_area(vertices):
    if len(vertices) < 3:
        return 0

    area = 0
    x, y = zip(*vertices)
    for i in range(len(vertices)):
        xi, yi = x[i], y[i]
        xi1, yi1 = x[(i + 1) % len(vertices)]
        area += xi * yi1 - xi1 * yi

    return abs(area) / 2.0


# +
def train_net(net1,
              net2,
              ensemble_net, 
              device,
              epochs: int = 50, #50
              batch_size: int = 8,
              learning_rate: float = learning_rate_default,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 1,
              amp: bool = False):
    
    dataset = BasicDataset(dir_img, dir_mask, img_scale, data_argu = data_argu )
    dp_net1 = net1
    net1 = net1.module
    dp_net2 = net2
    net2 = net2.module
    dp_ensemble_net = ensemble_net
    ensemble_net = ensemble_net.module
    global max_test
    max_test = 0.8
    max_test1 = 0.7
    max_test2 = 0.7
    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    half_n_train = int((len(dataset) - n_val)/2)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
#     train_set = BasicDataset(dir_img, dir_mask, img_scale, data_argu = data_argu )
#     val_set = BasicDataset(dir_val_img, dir_mask, img_scale, data_argu = 0 )
    test_set = BasicDataset(dir_test_img,dir_test_mask,img_scale, data_argu = 0)
    test_non_tumor_set = BasicDataset(dir_non_tumor_test_img,dir_non_tumor_test_mask,img_scale, data_argu = 0)
    test_MoNuSeg_set = BasicDataset(dir_test_MoNuSeg_img,dir_test_MoNuSeg_mask,img_scale, data_argu = 0)
    print('-------',test_non_tumor_set)
    loader_args = dict(batch_size=batch_size, num_workers=8, pin_memory=True)
    val_loader_args = dict(batch_size=batch_size, num_workers=64, pin_memory=True)
    test_loader_args = dict(batch_size=56, num_workers=64, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
    inf_train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **test_loader_args)
    inf_val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **test_loader_args)
    inf_test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **test_loader_args)
    inf_test_non_tumor_loader = DataLoader(test_non_tumor_set, shuffle=False, drop_last=False, **test_loader_args)
    inf_test_MoNuSeg_non_tumor_loader = DataLoader(test_MoNuSeg_set, shuffle=False, drop_last=False, **test_loader_args)
    experiment = wandb.init(project='U-Net',name=name_+'_eq17_epoch4005',resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, 
                                  ptim_select=optim_select, sche_select=sche_select,
                                  para_causel=para_causel,para_shape=para_shape,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Network:         {net1.__class__.__name__}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate_default}
        Para_causel:     {para_causel}
        Para_shape:      {para_shape}
        optim_select:    {optim_select}
        sche_select:     {sche_select}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    if optim_select=="RMSprop":
            optimizer1 = torch.optim.RMSprop(dp_net1.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
            optimizer2 = torch.optim.RMSprop(dp_net2.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
            optimizer3 = torch.optim.RMSprop(ensemble_net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
            
    if sche_select=="CosineAnnealingLR":
        scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1,T_max=5)
        scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2,T_max=5)
        
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    
    
    
    global_step = 0

    # 5. Begin training
    precision_tumorlist = []
    precision_out_tumorlist = []
    precision_untumorlist = []
    recall_score_tumorlist = []
    recall_scoreout_tumorlist = []
    recall_scoreout_untumorlist = []
    for epoch in range(epochs):
        dp_net1.train()
        dp_net2.train()
        
        epoch_DE1_loss = 0
        epoch_DE2_loss = 0
        #===============================dann==============================================
        i = 0
        #===============================dann==============================================
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
#             for batch1, batch2  in zip(train_loader1, train_loader2):
            for batch in train_loader:
#                 print(batch1['image_aug'].size())
#                 print(batch2['image_aug'].size())
                images = batch['image_aug']
#                 images1 = batch1['image_aug']
#                 images2 = batch2['image_aug']
                true_masks = batch['mask_aug']
                true_masks = torch.cat((true_masks, true_masks), 0) 
#                 true_masks1 = batch1['mask_aug']
#                 true_masks1 = torch.cat((true_masks1, true_masks1), 0) 
#                 true_masks2 = batch2['mask_aug']
#                 true_masks2 = torch.cat((true_masks2, true_masks2), 0) 
                #-----------------------------------causel-----------------------------------
                img_trsf1 = batch['img_trsf1']
#                 img_trsf11 = batch1['img_trsf1']
#                 img_trsf12 = batch2['img_trsf1']
#                 print('images.shape',images.size())
#                 print('true_masks.shape',true_masks.size())
                images_cat = torch.cat((images, img_trsf1), 0)
#                 images_cat1 = torch.cat((images1, img_trsf11), 0)
#                 images_cat2 = torch.cat((images2, img_trsf12), 0)
                #-----------------------------------causel-----------------------------------
                
                assert images.shape[1] == net1.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                images_cat = images_cat.to(device=device, dtype=torch.float32)
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
#                 images_cat1 = images_cat1.to(device=device, dtype=torch.float32)
#                 images1 = images1.to(device=device, dtype=torch.float32)
#                 true_masks1 = true_masks1.to(device=device, dtype=torch.long)
#                 images_cat2 = images_cat2.to(device=device, dtype=torch.float32)
#                 images2 = images2.to(device=device, dtype=torch.float32)
#                 true_masks2 = true_masks2.to(device=device, dtype=torch.long)
                
                with torch.cuda.amp.autocast(enabled=amp): 
                    images_cat = images_cat.to(device=f'cuda:{dp_net1.device_ids[0]}')
#                     print('images_cat.get_device()',images_cat.get_device())
#                     print(next(dp_net1.parameters()).device)
                    masks_pred1, x51, x_1, x_i1 = dp_net1(images_cat.to(device=f'cuda:{dp_net1.device_ids[0]}'))
                    images_cat = images_cat.to(device=f'cuda:{dp_net2.device_ids[0]}')
#                     print('2images_cat.get_device()',images_cat.get_device())
#                     print(next(dp_net2.parameters()).device)
                    masks_pred2, x52, x_2, x_i2 = dp_net2(images_cat.to(device=f'cuda:{dp_net2.device_ids[0]}'))
                   #-----------------------------------causel-----------------------------------------    
                    causel_loss1 = 0.0
                    causel_loss2 = 0.0
                    shape_loss1 = 0.0
                    shape_loss2 = 0.0
                    features1 = torch.flatten(x51, 1)
                    logits1 = features1[0:batch_size]
                    logits1 = torch.nn.Softmax(dim=1)(logits1).to(device=device)
#                     print(logits.size())
                    logits21 = features1[batch_size:2*batch_size]
                    logits21 = torch.nn.Softmax(dim=1)(logits21).to(device=device)
                    loss11 = nn.KLDivLoss(reduction="batchmean", log_target=True)(logits1,logits21)
                    loss21 = nn.KLDivLoss(reduction="batchmean", log_target=True)(logits21,logits1)
                    causel_loss1 = (loss11+loss21)/2
#                     shape_loss1 = nn.MSELoss()(x_1 , x_i1)
#                     print('1',x_1.size())
#                     print('2',x_i1.size())
                    shape_loss1 = x_i1.mean()
                    features2 = torch.flatten(x52, 1)
                    logits2 = features2[0:batch_size]
                    logits2 = torch.nn.Softmax(dim=1)(logits2).to(device=device)
                    logits22 = features2[batch_size:2*batch_size]
                    logits22 = torch.nn.Softmax(dim=1)(logits22).to(device=device)
                    loss12 = nn.KLDivLoss(reduction="batchmean", log_target=True)(logits2,logits22)
                    loss22 = nn.KLDivLoss(reduction="batchmean", log_target=True)(logits22,logits2)
                    causel_loss2 = (loss12+loss22)/2
#                     shape_loss2 = nn.MSELoss()(x_2 ,x_i2)  
                    shape_loss2 = x_i2.mean()
                optimizer1.zero_grad()
                diceloss_1 = dice_loss(F.softmax(masks_pred1, dim=1).float() \
                                ,F.one_hot(true_masks.to(device=f'cuda:{dp_net1.device_ids[0]}'), net1.n_classes).permute(0, 3, 1, 2).float() \
                                ,multiclass=True)
                loss_DE1 = 1*(criterion(masks_pred1, true_masks.to(device=f'cuda:{dp_net1.device_ids[0]}')) \
                    + diceloss_1 \
                    + para_causel *causel_loss1.to(device=f'cuda:{dp_net1.device_ids[0]}', dtype=torch.float64) 
                    + para_shape *shape_loss1.to(device=f'cuda:{dp_net1.device_ids[0]}', dtype=torch.float64) )
                grad_scaler.scale(loss_DE1).backward()
                grad_scaler.step(optimizer1)
                grad_scaler.update()
                
                optimizer2.zero_grad()
                diceloss_2  = dice_loss(F.softmax(masks_pred2, dim=1).float() \
                                ,F.one_hot(true_masks.to(device=f'cuda:{dp_net2.device_ids[0]}'), net2.n_classes).permute(0, 3, 1, 2).float() \
                                ,multiclass=True)
                loss_DE2 = 1*(criterion(masks_pred2, true_masks.to(device=f'cuda:{dp_net2.device_ids[0]}')) \
                    + diceloss_2 \
                    + para_causel *causel_loss2.to(device=f'cuda:{dp_net2.device_ids[0]}', dtype=torch.float64) 
                    + para_shape *shape_loss2.to(device=f'cuda:{dp_net2.device_ids[0]}', dtype=torch.float64)  )
                grad_scaler.scale(loss_DE2).backward()
                grad_scaler.step(optimizer2)
                grad_scaler.update()
                
                
                pbar.update(images.shape[0])
                global_step += 1
                epoch_DE1_loss += loss_DE1.item()
                epoch_DE2_loss += loss_DE2.item()
                experiment.log({
                    'diceloss_1' : diceloss_1,
                    'diceloss_2' : diceloss_2,
                    'DE1 loss': loss_DE1.item(),
                    'DE2 loss': loss_DE2.item(),
                    'step': global_step,
                    'epoch': epoch 
                })
                i += 1
                scheduler1.step()
                scheduler2.step()
                
#                 if epoch % 10 ==0:
                dp_net1.eval()
                dp_net2.eval()
                ensemble_net.train() 
                images = batch['image']
                true_masks = batch['mask']
#                 images1 = batch1['image']
#                 true_masks1 = batch1['mask']
#                 images2 = batch2['image']
#                 true_masks2 = batch2['mask']
#                 true_masks = torch.cat((true_masks1, true_masks2), 0) 
#                 images_cat = torch.cat((images1, images2), 0)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                images_cat = images.to(device=device, dtype=torch.float32)


                masks_pred1, x51, _, __ = dp_net1(images_cat)
                masks_pred2, x52, _, __ = dp_net2(images_cat)
                masks_mix = torch.cat((masks_pred1.to(device=f'cuda:{dp_ensemble_net.device_ids[0]}'), masks_pred2.to(device=f'cuda:{dp_ensemble_net.device_ids[0]}')), 1)
                masks_final = ensemble_net(masks_mix)    
                
                loss_ensemble = 1*(criterion(masks_final, true_masks.to(device=f'cuda:{dp_ensemble_net.device_ids[0]}')) \
                + dice_loss(F.softmax(masks_final, dim=1).float() \
                            ,F.one_hot(true_masks.to(device=f'cuda:{dp_ensemble_net.device_ids[0]}'), net2.n_classes).permute(0, 3, 1, 2).float() \
                            ,multiclass=True))

                grad_scaler.scale(loss_ensemble).backward()
                grad_scaler.step(optimizer3)
                grad_scaler.update()
                    
                    
                # Evaluation round
                one_round = int(n_train/batch_size)
                division_step = (n_train / (batch_size))
                if division_step > 0:
                    if ((global_step/10) % one_round == 0) or ((global_step) / one_round ==1):
                        dp_net1.eval()
                        dp_net2.eval()
                        dp_ensemble_net.eval()  
                        
#                         train_iou1, train_f11, train_precision1, train_recall1, train_miou1 = evaluate_all(net1, train_loader, device, train_=1)
#                         iou1, f11, precision1, recall1, miou1= evaluate_all(net1, val_loader, device,)
#                         test_iou1, test_f11, test_precision1, test_recall1, test_miou1 = evaluate_all(net1, test_loader, device)
                        
#                         scheduler.step(train_f1)

#                         train_iou2, train_f12, train_precision2, train_recall2, train_miou2 = evaluate_all(net2, train_loader, device, train_=1)
#                         iou2, f12, precision2, recall2, miou2= evaluate_all(net2, val_loader, device,)
#                         test_iou2, test_f12, test_precision2, test_recall2, test_miou2 = evaluate_all(net2, test_loader, device)
                        train_iou1, train_f11, train_PRCC1, train_precision1, train_recall1, train_miou1,train_iou2, train_f12, train_PRCC2, train_precision2, train_recall2, train_miou2, train_iou3, train_f13, train_PRCC3, train_precision3, train_recall3, train_miou3= evaluate_all_ensemble(net1, net2, ensemble_net, inf_train_loader, device, train_=1)
                        iou1, f11, PRCC1, precision1, recall1, miou1, iou2, f12, PRCC2, precision2, recall2, miou2, iou3, f13, PRCC3, precision3, recall3, miou3= evaluate_all_ensemble(net1, net2, ensemble_net, inf_val_loader, device, train_=1)
                        test_iou1, test_f11, test_PRCC1, test_precision1, test_recall1, test_miou1, test_iou2, test_f12, test_PRCC2, test_precision2, test_recall2, test_miou2,test_iou3, test_f13, test_PRCC3, test_precision3, test_recall3, test_miou3 = evaluate_all_ensemble(net1, net2, ensemble_net, inf_test_loader, device)
                        test_non_tumor_iou1, test_non_tumor_f11, test_non_tumor_PRCC1, test_non_tumor_precision1, test_non_tumor_recall1, test_non_tumor_miou1, test_non_tumor_iou2, test_non_tumor_f12, test_non_tumor_PRCC2, test_non_tumor_precision2, test_non_tumor_recall2, test_non_tumor_miou2,test_non_tumor_iou3, test_non_tumor_f13, test_non_tumor_PRCC3, test_non_tumor_precision3, test_non_tumor_recall3, test_non_tumor_miou3 = evaluate_all_ensemble(net1, net2, ensemble_net, inf_test_non_tumor_loader, device)
                        test_MoNuSeg_iou1, test_MoNuSeg_f11, test_MoNuSeg_PRCC1, test_MoNuSeg_precision1, test_MoNuSeg_recall1, test_MoNuSeg_miou1, test_MoNuSeg_iou2, test_MoNuSeg_f12, test_MoNuSeg_PRCC2, test_MoNuSeg_precision2, test_MoNuSeg_recall2, test_MoNuSeg_miou2,test_MoNuSeg_iou3, test_MoNuSeg_f13, test_MoNuSeg_PRCC3, test_MoNuSeg_precision3, test_MoNuSeg_recall3, test_MoNuSeg_miou3 = evaluate_all_ensemble(net1, net2, ensemble_net, inf_test_MoNuSeg_non_tumor_loader, device)
                        logging.info('Name: {}'.format(name_))
            
                        logging.info('train_IOU1: {}'.format(train_iou1))
                        logging.info('train_MIOU1: {}'.format(train_miou1))
                        logging.info('train_F1-score1: {}'.format(train_f11))
                        logging.info('train_PRCC1: {}'.format(train_PRCC1))
                        logging.info('train_Precision1: {}'.format(train_precision1))
                        logging.info('train_Recall1: {}'.format(train_recall1))
                        logging.info('train_IOU2: {}'.format(train_iou2))
                        logging.info('train_MIOU2: {}'.format(train_miou2))
                        logging.info('train_F1-score2: {}'.format(train_f12))
                        logging.info('train_PRCC2: {}'.format(train_PRCC2))
                        logging.info('train_Precision2: {}'.format(train_precision2))
                        logging.info('train_Recall2: {}'.format(train_recall2))
                        logging.info('train_IOU3: {}'.format(train_iou3))
                        logging.info('train_MIOU3: {}'.format(train_miou3))
                        logging.info('train_F1-score3: {}'.format(train_f13))
                        logging.info('train_PRCC3: {}'.format(train_PRCC3))
                        logging.info('train_Precision3: {}'.format(train_precision3))
                        logging.info('train_Recall3: {}'.format(train_recall3))
                        
                        logging.info('IOU1: {}'.format(iou1))
                        logging.info('MIOU1: {}'.format(miou1))
                        logging.info('F1-score1: {}'.format(f11))
                        logging.info('PRCC1: {}'.format(PRCC1))
                        logging.info('Precision1: {}'.format(precision1))
                        logging.info('Recall1: {}'.format(recall1))
                        logging.info('IOU2: {}'.format(iou2))
                        logging.info('MIOU2: {}'.format(miou2))
                        logging.info('F1-score2: {}'.format(f12))
                        logging.info('PRCC2: {}'.format(PRCC2))
                        logging.info('Precision2: {}'.format(precision2))
                        logging.info('Recall2: {}'.format(recall2))
                        logging.info('IOU3: {}'.format(iou3))
                        logging.info('MIOU3: {}'.format(miou3))
                        logging.info('F1-score3: {}'.format(f13))
                        logging.info('PRCC3: {}'.format(PRCC3))
                        logging.info('Precision3: {}'.format(precision3))
                        logging.info('Recall3: {}'.format(recall3))
                        
                        logging.info('Test_IOU1: {}'.format(test_iou1))
                        logging.info('Test_IOU2: {}'.format(test_iou2))
                        logging.info('Test_IOU3: {}'.format(test_iou3))
                        logging.info('Test_MIOU1: {}'.format(test_miou1))
                        logging.info('Test_MIOU2: {}'.format(test_miou2))
                        logging.info('Test_MIOU3: {}'.format(test_miou3))
                        logging.info('Test_F1-score1: {}'.format(test_f11))
                        logging.info('Test_F1-score2: {}'.format(test_f12))
                        logging.info('Test_F1-score3: {}'.format(test_f13))
                        logging.info('Test_PRCC1: {}'.format(test_PRCC1))
                        logging.info('Test_PRCC2: {}'.format(test_PRCC2))
                        logging.info('Test_PRCC3: {}'.format(test_PRCC3))
                        logging.info('Test_Precision1: {}'.format(test_precision1))
                        logging.info('Test_Precision2: {}'.format(test_precision2))
                        logging.info('Test_Precision3: {}'.format(test_precision3))
                        logging.info('Test_Recall1: {}'.format(test_recall1))
                        logging.info('Test_Recall2: {}'.format(test_recall2))
                        logging.info('Test_Recall3: {}'.format(test_recall3))
                        
                        logging.info('Test_MoNuSeg_IOU1: {}'.format(test_MoNuSeg_iou1))
                        logging.info('Test_MoNuSeg_IOU2: {}'.format(test_MoNuSeg_iou2))
                        logging.info('Test_MoNuSeg_IOU3: {}'.format(test_MoNuSeg_iou3))
                        logging.info('Test_MoNuSeg_MIOU1: {}'.format(test_MoNuSeg_miou1))
                        logging.info('Test_MoNuSeg_MIOU2: {}'.format(test_MoNuSeg_miou2))
                        logging.info('Test_MoNuSeg_MIOU3: {}'.format(test_MoNuSeg_miou3))
                        logging.info('Test_MoNuSeg_F1-score1: {}'.format(test_MoNuSeg_f11))
                        logging.info('Test_MoNuSeg_F1-score2: {}'.format(test_MoNuSeg_f12))
                        logging.info('Test_MoNuSeg_F1-score3: {}'.format(test_MoNuSeg_f13))
                        logging.info('Test_MoNuSeg_PRCC1: {}'.format(test_MoNuSeg_PRCC1))
                        logging.info('Test_MoNuSeg_PRCC2: {}'.format(test_MoNuSeg_PRCC2))
                        logging.info('Test_MoNuSeg_PRCC3: {}'.format(test_MoNuSeg_PRCC3))
                        logging.info('Test_MoNuSeg_Precision1: {}'.format(test_MoNuSeg_precision1))
                        logging.info('Test_MoNuSeg_Precision2: {}'.format(test_MoNuSeg_precision2))
                        logging.info('Test_MoNuSeg_Precision3: {}'.format(test_MoNuSeg_precision3))
                        logging.info('Test_MoNuSeg_Recall1: {}'.format(test_MoNuSeg_recall1))
                        logging.info('Test_MoNuSeg_Recall2: {}'.format(test_MoNuSeg_recall2))
                        logging.info('Test_MoNuSeg_Recall3: {}'.format(test_MoNuSeg_recall3))
                        
                        logging.info('Test_non_tumor_IOU1: {}'.format(test_non_tumor_iou1))
                        logging.info('Test_non_tumor_IOU2: {}'.format(test_non_tumor_iou2))
                        logging.info('Test_non_tumor_IOU3: {}'.format(test_non_tumor_iou3))
                        logging.info('Test_non_tumor_MIOU1: {}'.format(test_non_tumor_miou1))
                        logging.info('Test_non_tumor_MIOU2: {}'.format(test_non_tumor_miou2))
                        logging.info('Test_non_tumor_MIOU3: {}'.format(test_non_tumor_miou3))
                        logging.info('Test_non_tumor_F1-score1: {}'.format(test_non_tumor_f11))
                        logging.info('Test_non_tumor_F1-score2: {}'.format(test_non_tumor_f12))
                        logging.info('Test_non_tumor_F1-score3: {}'.format(test_non_tumor_f13))
                        logging.info('Test_non_tumor_PRCC1: {}'.format(test_non_tumor_PRCC1))
                        logging.info('Test_non_tumor_PRCC2: {}'.format(test_non_tumor_PRCC2))
                        logging.info('Test_non_tumor_PRCC3: {}'.format(test_non_tumor_PRCC3))
                        logging.info('Test_non_tumor_Precision1: {}'.format(test_non_tumor_precision1))
                        logging.info('Test_non_tumor_Precision2: {}'.format(test_non_tumor_precision2))
                        logging.info('Test_non_tumor_Precision3: {}'.format(test_non_tumor_precision3))
                        logging.info('Test_non_tumor_Recall1: {}'.format(test_non_tumor_recall1))
                        logging.info('Test_non_tumor_Recall2: {}'.format(test_non_tumor_recall2))
                        logging.info('Test_non_tumor_Recall3: {}'.format(test_non_tumor_recall3))
#                         precision_tumorlist = []
#                         precision_out_tumorlist = []
#                         precision_untumorlist = []
#                         recall_score_tumorlist = []
#                         recall_scoreout_tumorlist = []
#                         recall_scoreout_untumorlist = []
                        precision_tumorlist.append(precision3)
                        precision_out_tumorlist.append(test_precision3)
                        precision_untumorlist.append(test_non_tumor_precision3)
                        recall_score_tumorlist.append(recall3)
                        recall_scoreout_tumorlist.append(test_recall3)
                        recall_scoreout_untumorlist.append(test_non_tumor_recall3)
                        experiment.log({
                            'learning rate': optimizer1.param_groups[0]['lr'],
                            
                            'train_IOU1': train_iou1,
                            'train_MIOU1': train_miou1,
                            'train_F1-score1' : train_f11,
                            'train_PRCC1' : train_PRCC1,
                            'train_Precision1' : train_precision1,
                            'train_Recall1': train_recall1,
                            'IOU1': iou1,
                            'MIOU1': miou1,
                            'F1-score1' : f11,
                            'PRCC1' : PRCC1,
                            'Precision1' : precision1,
                            'Recall1': recall1,
                            
                            'Test_IOU1': test_iou1,
                            'Test_MIOU1': test_miou1,
                            'Test_F1-score1' : test_f11,
                            'Test_PRCC1' : test_PRCC1,
                            'Test_Precision1' : test_precision1,
                            'Test_Recall1': test_recall1,
                            'Test_non_tumor_IOU1': test_non_tumor_iou1,
                            'Test_non_tumor_MIOU1': test_non_tumor_miou1,
                            'Test_non_tumor_F1-score1' : test_non_tumor_f11,
                            'Test_non_tumor_PRCC1' : test_non_tumor_PRCC1,
                            'Test_non_tumor_Precision1' : test_non_tumor_precision1,
                            'Tes_non_tumort_Recall1': test_non_tumor_recall1,
                            
                            'train_IOU': train_iou3,
                            'train_MIOU': train_miou3,
                            'train_F1-score' : train_f13,
                            'train_PRCC' : train_PRCC3,
                            'train_Precision' : train_precision3,
                            'train_Recall': train_recall3,
                            'IOU': iou3,
                            'MIOU': miou3,
                            'F1-score' : f13,
                            'PRCC' : PRCC3,
                            'Precision' : precision3,
                            'Recall': recall3,
                            'Test_IOU': test_iou3,
                            'Test_MIOU': test_miou3,
                            'Test_F1-score' : test_f13,
                            'Test_PRCC' : test_PRCC3,
                            'Test_Precision' : test_precision3,
                            'Test_Recall': test_recall3,
                            'Test_non_tumor_IOU': test_non_tumor_iou3,
                            'Test_non_tumor_MIOU': test_non_tumor_miou3,
                            'Test_non_tumor_F1-score' : test_non_tumor_f13,
                            'Test_non_tumor_PRCC' : test_non_tumor_PRCC3,
                            'Test_non_tumor_Precision' : test_non_tumor_precision3,
                            'Test_non_tumor_Recall': test_non_tumor_recall3,
                            
                            'train_IOU2': train_iou2,
                            'train_MIOU2': train_miou2,
                            'train_F1-score2' : train_f12,
                            'train_PRCC2' : train_PRCC2,
                            'train_Precision2' : train_precision2,
                            'train_Recall2': train_recall2,
                            'IOU2': iou2,
                            'MIOU2': miou2,
                            'F1-score2' : f12,
                            'PRCC2' : PRCC2,
                            'Precision2' : precision2,
                            'Recall2': recall2,
                            'Test_IOU2': test_iou2,
                            'Test_MIOU2': test_miou2,
                            'Test_F1-score2' : test_f12,
                            'Test_PRCC2' : test_PRCC2,
                            'Test_Precision2' : test_precision2,
                            'Test_Recall2': test_recall2,
                            'Test_non_tumor_IOU2': test_non_tumor_iou2,
                            'Test_non_tumor_MIOU2': test_non_tumor_miou2,
                            'Test_non_tumor_F1-score2' : test_non_tumor_f12,
                            'Test_non_tumor_PRCC2' : test_non_tumor_PRCC2,
                            'Test_non_tumor_Precision2' : test_non_tumor_precision2,
                            'Test_non_tumor_Recall2': test_non_tumor_recall2,
                            
                            'Test_MoNuSeg_IOU1': test_MoNuSeg_iou1,
                            'Test_MoNuSeg_IOU2': test_MoNuSeg_iou2,
                            'Test_MoNuSeg_IOU3': test_MoNuSeg_iou3,
                            'Test_MoNuSeg_MIOU1': test_MoNuSeg_miou1,
                            'Test_MoNuSeg_MIOU2': test_MoNuSeg_miou2,
                            'Test_MoNuSeg_MIOU3': test_MoNuSeg_miou3,
                            'Test_MoNuSeg_F1-score1': test_MoNuSeg_f11,
                            'Test_MoNuSeg_F1-score2': test_MoNuSeg_f12,
                            'Test_MoNuSeg_F1-score3': test_MoNuSeg_f13,
                            'Test_MoNuSeg_PRCC1': test_MoNuSeg_PRCC1,
                            'Test_MoNuSeg_PRCC2': test_MoNuSeg_PRCC2,
                            'Test_MoNuSeg_PRCC3': test_MoNuSeg_PRCC3,
                            'Test_MoNuSeg_Precision1': test_MoNuSeg_precision1,
                            'Test_MoNuSeg_Precision2': test_MoNuSeg_precision2,
                            'Test_MoNuSeg_Precision3': test_MoNuSeg_precision3,
                            'Test_MoNuSeg_Recall1': test_MoNuSeg_recall1,
                            'Test_MoNuSeg_Recall2': test_MoNuSeg_recall2,
                            'Test_MoNuSeg_Recall3': test_MoNuSeg_recall3,
#                             'images2': wandb.Image(cv2.cvtColor(images2[0].cpu().numpy().transpose((1,2,0 )) , cv2.COLOR_RGB2BGR)),
#                             'masks2': {
#                                 'true2': wandb.Image(true_masks2[0].float().cpu()),
#                                 'pred2': wandb.Image(torch.softmax(masks_pred2, dim=1).argmax(dim=1)[0].float().cpu()),
#                             },
                            'step': global_step,
                            'epoch': epoch,
                        })
                        if save_checkpoint:
                            test_f11 = test_f11.to(device=device)
                            test_f12 = test_f12.to(device=device)
                            test_PRCC1 = test_PRCC1.to(device=device)
                            test_PRCC2 = test_PRCC2.to(device=device)
                            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)            
                            if (test_f11 >= max_test1 and test_f12 >= max_test1) or ((global_step/20) % one_round == 0) :
                                max_test1 = test_f11
                                torch.save(net1.state_dict(), dir_checkpoint / 'net1_checkpoint_epoch{}.pth'.format((epoch + 1)))
                                max_test2 = test_f12
                                torch.save(net2.state_dict(), dir_checkpoint / 'net2_checkpoint_epoch{}.pth'.format((epoch + 1)))
                                logging.info(f'Checkpoint {epoch + 1} saved!')
                            if (test_f13 >= max_test) or ((global_step/20) % one_round == 0) :
                                max_test1 = test_f11
                                torch.save(net1.state_dict(), dir_checkpoint / 'net1_checkpoint_epoch{}.pth'.format((epoch + 1)))
                                max_test2 = test_f12
                                torch.save(net2.state_dict(), dir_checkpoint / 'net2_checkpoint_epoch{}.pth'.format((epoch + 1)))
                                max_test = test_f13
                                torch.save(ensemble_net.state_dict(), dir_checkpoint / 'ensemble_net_checkpoint_epoch{}.pth'.format((epoch + 1)))
                                logging.info(f'Checkpoint {epoch + 1} saved!')
                            if (test_PRCC1 >= max_test1 and test_f12 >= max_test1) or ((global_step/20) % one_round == 0) :
                                max_test1 = test_PRCC1
                                torch.save(net1.state_dict(), dir_checkpoint / 'net1_checkpoint_epoch{}.pth'.format((epoch + 1)))
                                max_test2 = test_PRCC2
                                torch.save(net2.state_dict(), dir_checkpoint / 'net2_checkpoint_epoch{}.pth'.format((epoch + 1)))
                                logging.info(f'Checkpoint {epoch + 1} saved!')
                            if (test_PRCC3 >= max_test) or ((global_step/20) % one_round == 0) :
                                max_test1 = test_PRCC1
                                torch.save(net1.state_dict(), dir_checkpoint / 'net1_checkpoint_epoch{}.pth'.format((epoch + 1)))
                                max_test2 = test_PRCC2
                                torch.save(net2.state_dict(), dir_checkpoint / 'net2_checkpoint_epoch{}.pth'.format((epoch + 1)))
                                max_test = test_PRCC3
                                torch.save(ensemble_net.state_dict(), dir_checkpoint / 'ensemble_net_checkpoint_epoch{}.pth'.format((epoch + 1)))
                                logging.info(f'Checkpoint {epoch + 1} saved!')
#         precision_tumorlist.append(precision3)
#         precision_out_tumorlist.append(test_precision3)
#         precision_untumorlist.append(test_non_tumor_precision3)
#         recall_score_tumorlist.append(recall3)
#         recall_scoreout_tumorlist.append(test_recall3)
#         recall_scoreout_untumorlist.append(test_non_tumor_recall3)
        area_in_tumor=calculate_polygon_area([precision_tumorlist,recall_score_tumorlist])
        area_out_tumor=calculate_polygon_area([precision_out_tumorlist,recall_scoreout_tumorlist])
        area_out_untumor=calculate_polygon_area([precision_untumorlist,recall_scoreout_untumorlist])
        experiment.log({'area_in_tumor':area_in_tumor,'area_out_tumor':area_out_tumor,'area_out_untumor':area_out_untumor})


# +
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=2, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    return parser.parse_args()
if __name__ == '__main__':
    args = get_args()

    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    net1 = ResUnetPlusPlus(n_channels=3, n_classes=2, bilinear=True)
    net2 = ResUnetPlusPlus(n_channels=3, n_classes=2, bilinear=True)
    ensemble_net = convolution_1x1()
    net1.ResUnetPlusPlus_FE.infodrop_1.initialize_parameters()
    net2.ResUnetPlusPlus_FE.infodrop_1.initialize_parameters()
    ######################################################################
    #                         user define                                #
    ######################################################################
    name_ = '1/2_KDE-IN_'+str(CA)+'_'+str(net1.__class__.__name__)+ test_data +'-'+ str(para_causel)+'-'+ str(para_shape)
    
    dir_img = Path('/home/msoc956131/final_unet_dann_v3/data/train/'+ train_data)
#     dir_val_img = Path('/home/msoc956131/final_unet_dann_v3/data/train/valid')
#     dir_mask = Path('/home/msoc956131/final_unet_dann_v3/data/train/semantic-mask')
#     dir_mask = Path('/home/msoc956131/final_unet_dann_v3/data/train/inside_mask')
    dir_mask = Path('/home/msoc956131/final_unet_dann_v3/data/train/mask')
    dir_test_img = Path('/home/msoc956131/final_unet_dann_v3/data/test/3d_v1')
    dir_test_MoNuSeg_img = Path('/home/msoc956131/final_unet_dann_v3/data/test/MoNuSeg_data')
    dir_non_tumor_test_img = Path('/home/msoc956131/final_unet_dann_v3/data/test/3d-non_tumor')
#     dir_test_mask = Path('/home/msoc956131/final_unet_dann_v3/data/train/semantic-mask')
    dir_test_mask = Path('/home/msoc956131/final_unet_dann_v3/data/test/3d_v1_inside_mask')
    dir_test_MoNuSeg_mask = Path('/home/msoc956131/final_unet_dann_v3/data/test/MoNuSeg_data_mask')
    dir_non_tumor_test_mask = Path('/home/msoc956131/final_unet_dann_v3/data/test/3d-non_tumor_inside_mask')
    dir_checkpoint = Path('/home/msoc956131/final_unet_dann_v3/checkpoints/checkpoints_'+ name_+'/')
    
    net1 = nn.DataParallel(net1,device_ids = [ 3, 1, 2, 0]) #device_ids=[0, 1, 2]
    net2 = nn.DataParallel(net2,device_ids = [ 2, 1, 0, 3 ]) #device_ids=[0, 1, 2]
    ensemble_net = nn.DataParallel(ensemble_net,device_ids = [ 1, 0, 2, 3])
    logging.info(f'Network:\n'
                 f'\t{net1.module.n_channels} input channels\n'
                 f'\t{net1.module.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net1.module.bilinear else "Transposed conv"} upscaling')

    if args.load:
        
        net1.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net1.to(device=f'cuda:{net1.device_ids[0]}')
    net2.to(device=f'cuda:{net2.device_ids[0]}')
    ensemble_net.to(device=f'cuda:{ensemble_net.device_ids[0]}')
    #===========================================================================================
    
    try:
        train_net(net1=net1,
                  net2=net2,
                  ensemble_net=ensemble_net, 
                  epochs=400, #400
                  batch_size= 4, #3*3: attention 40  ,ResUnetPlusPlus 32 5*5: attention 30  ,ResUnetPlusPlus 12 
                  learning_rate=args.lr,
                  device=device,
                  img_scale=1,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
#         torch.save(net.state_dict(), 'INTERRUPTED.pth')
#         logging.info('Saved interrupt')
        sys.exit(0)



