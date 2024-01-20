# -*- coding: utf-8 -*-
# +
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
# import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# -

def dice_coeff_19(input_: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
#     print('input.size()',input.size())
#     print('target.size()',target.size())
#     print("78",input.size() == target.size())
    assert input_.size() == target.size()
    if input_.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

#     if input_.dim() == 2 or reduce_batch_first:
# #         print('input.dim()',input_.dim())
#         inter = torch.dot(input_.reshape(-1), target.reshape(-1))
#         sets_sum = torch.sum(input_) + torch.sum(target)
#         if sets_sum.item() == 0:
#             sets_sum = 2 * inter

#         return (2 * inter + epsilon) / (sets_sum + epsilon)
    if input_.dim() == 2 or reduce_batch_first:
#         print('input.dim()',input_.dim())
        inter = torch.sqrt(torch.dot(input_.reshape(-1), (1-input_.reshape(-1)))) #target.reshape(-1) # 開跟號(q*p)
        sets_sum = torch.sum(input_) + torch.sum(1-input_)#torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter
        
        return (2 * inter + epsilon) / ((sets_sum + epsilon)*1)
    
    else:
        # compute and average metric for each batch element
        dice = 0
        
        for i in range(input_.shape[0]):
            dice += dice_coeff(input_[i, ...], target[i, ...])
        return dice / input_.shape[0]

def dice_coeff_20(input_: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
#     print('input.size()',input.size())
#     print('target.size()',target.size())
#     print("78",input.size() == target.size())
    assert input_.size() == target.size()
    if input_.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

#     if input_.dim() == 2 or reduce_batch_first:
# #         print('input.dim()',input_.dim())
#         inter = torch.dot(input_.reshape(-1), target.reshape(-1))
#         sets_sum = torch.sum(input_) + torch.sum(target)
#         if sets_sum.item() == 0:
#             sets_sum = 2 * inter

#         return (2 * inter + epsilon) / (sets_sum + epsilon)
    if input_.dim() == 2 or reduce_batch_first:
#         print('input.dim()',input_.dim())
        inter = torch.dot(input_.reshape(-1), (1-input_.reshape(-1))) #target.reshape(-1) # 開跟號(q*p)
        tensor_with_twos = torch.ones_like(input_) * 2.0
        sets_sum = torch.sum(torch.pow(input_,tensor_with_twos)) + torch.sum(torch.pow((1-input_),tensor_with_twos))#torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter
        
        return (2 * inter + epsilon) / ((sets_sum + epsilon))
    
    else:
        # compute and average metric for each batch element
        dice = 0
        
        for i in range(input_.shape[0]):
            dice += dice_coeff_19(input_[i, ...], target[i, ...])
        return dice / input_.shape[0]


def dice_coeff_21(input_: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
#     print('input.size()',input.size())
#     print('target.size()',target.size())
#     print("78",input.size() == target.size())
    assert input_.size() == target.size()
    if input_.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

#     if input_.dim() == 2 or reduce_batch_first:
# #         print('input.dim()',input_.dim())
#         inter = torch.dot(input_.reshape(-1), target.reshape(-1))
#         sets_sum = torch.sum(input_) + torch.sum(target)
#         if sets_sum.item() == 0:
#             sets_sum = 2 * inter

#         return (2 * inter + epsilon) / (sets_sum + epsilon)
    if input_.dim() == 2 or reduce_batch_first:
#         print('input.dim()',input_.dim())
        inter = torch.sqrt(torch.dot(input_.reshape(-1), (1-input_.reshape(-1)))) #target.reshape(-1) # 開跟號(q*p)
        sets_sum = torch.sum(input_) + torch.sum(1-input_)
#         tensor_with_twos = torch.ones_like(input_) * 2.0
#         sets_sum = torch.sum(torch.pow(input_,tensor_with_twos)) + torch.sum(torch.pow((1-input_),tensor_with_twos))#torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter
        
        return (2 * inter + epsilon) / ((sets_sum + epsilon))
    
    else:
        # compute and average metric for each batch element
        dice = 0
        
        for i in range(input_.shape[0]):
            dice += dice_coeff_21(input_[i, ...], target[i, ...])
        return dice / input_.shape[0] 

def multiclass_dice_coeff(input_: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input_.size() == target.size()
    dice = 0
    for channel in range(input_.shape[1]):
        dice += dice_coeff_19(input_[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon) # 第16式

    return dice / input_.shape[1]


def dice_loss(input_: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input_.size() == target.size()
#     print('input.dim()',input_.dim())
    fn = multiclass_dice_coeff if multiclass else dice_coeff1
    
    return 1 - fn(input_, target, reduce_batch_first=True)



def dice_coeff1(input_: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
#     print('input.size()',input.size())
#     print('target.size()',target.size())
#     print("78",input.size() == target.size())
    assert input_.size() == target.size()
#     print('input.dim()',input_.dim())
    inter = torch.dot(input_.reshape(-1), target.reshape(-1))
    sets_sum = torch.sum(input_.reshape(-1)) + torch.sum(target.reshape(-1))
    if sets_sum.item() == 0:
        sets_sum = 2 * inter

    return (2 * inter + epsilon) / (sets_sum + epsilon)



def js_div(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    assert p_output.size() == q_output.size()
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
#     print('p_output.size()',p_output.size())
#     print('q_output.size()',q_output.size())
    p_output = torch.flatten(p_output)
    q_output = torch.flatten(q_output)
#     print('p_output_.size()',p_output.size())
#     print('q_output_.size()',q_output.size())
    if get_softmax:
        p_output = torch.nn.Softmax(dim=0)(p_output)
        q_output = torch.nn.Softmax(dim=0)(q_output)
#         print('p_output_.size()',p_output.size())
#         print('q_output_.size()',q_output.size())
    log_mean_output = ((p_output.cpu() + q_output.cpu() )/2).log()
    return (KLDivLoss(log_mean_output, p_output.cpu()) + KLDivLoss(log_mean_output, q_output.cpu()))/2

