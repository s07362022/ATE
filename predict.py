import argparse
import logging
import os
import cv2

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from evaluate import *
import csv

from utils.data_loading import *
from unet.unet_model import *
from utils.utils import plot_img_and_mask


# +
def predict_img(net1,
                net2,
                ensemble_net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net1.eval()
    net2.eval()
    ensemble_net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
#         output = net(img)
        output1, x51, x_1, x_i1 = net1(img)
        output2, x52, x_2, x_i2 = net2(img)
        masks_mix = torch.cat((output1.to(device=device),output2.to(device=device)),1)
        masks_final = ensemble_net(masks_mix)  
        probs1 = output1.argmax(dim=1)
        probs2 = output2.argmax(dim=1)
        probs3 = masks_final.argmax(dim=1)
        tf = transforms.Compose([
            transforms.ToPILImage(),
#             transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

#         full_mask = tf(probs.cpu()).squeeze()
#         probs.squeeze()
#     if net.n_classes == 1:
#         return (full_mask > out_threshold).numpy()
#     else:
#         full_mask[full_mask < out_threshold] = 0
#         return full_mask.numpy()
#     return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()
    return probs1.cpu().squeeze().numpy(),probs2.cpu().squeeze().numpy(),probs3.cpu().squeeze().numpy()


# -

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model1', '-m1', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--model2', '-m2', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--model_En', '-m3', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images')
    parser.add_argument('--mode', '-mo', help='test or not')
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--input_folder', '-if', help='Foldername of input images')
    parser.add_argument('--mask_gt_folder', '-mg', help='Foldername of mask gt images')
    parser.add_argument('--output_folder', '-of', help='Foldername of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask_threshold', '-t', type=float, default=0,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help='Scale factor for the input images')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
#         return Image.fromarray((mask * 255).astype(np.uint8))
        return 
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    args = get_args()
    # in_files = args.input
    in_folder = args.input_folder
    out_folder = args.output_folder
    mask_gt_folder = args.mask_gt_folder
    mode = args.mode
    if os.path.exists(out_folder)!=1:
        os.mkdir(out_folder)
    # out_files = get_output_filenames(args)

    in_files = []
    out_files = []
    
    for f in os.listdir(in_folder):
        in_files.append(os.path.join(in_folder, f))
        out_files.append(os.path.join(out_folder, f))
    
    net1 = ResUnetPlusPlus(n_channels=3, n_classes=2, bilinear=True)
    net2 = ResUnetPlusPlus(n_channels=3, n_classes=2, bilinear=True)
    ensemble_net = convolution_1x1()
#     net = UNet(n_channels=3, n_classes=2)
#     net = ResUnetPlusPlus(n_channels=3, n_classes=2, bilinear=True)
#     net = AttentionUNet(n_channels=3, n_classes=2, bilinear=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model1}')
    logging.info(f'Using device {device}')

    
    net1.to(device=device)
    net2.to(device=device)
    ensemble_net.to(device=device)
    net1.load_state_dict(torch.load(args.model1, map_location=device))
    net2.load_state_dict(torch.load(args.model2, map_location=device))
    ensemble_net.load_state_dict(torch.load(args.model_En, map_location=device))
    logging.info('Model loaded!')
    print('len(in_folder)',len(in_folder))
    for i, filename in enumerate(in_files):
        logging.info(f'\nPredicting image {filename} ...')
        
        img = cv2.imread(filename)

        mask1,mask2,mask3 = predict_img(net1=net1,
                           net2=net2,
                           ensemble_net=ensemble_net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
#         if not args.no_save:
        out_filename = out_files[i]
        print(i,':',out_filename)
        cv2.imwrite(out_filename[0:-4]+'_1.png', mask1*255)
        cv2.imwrite(out_filename[0:-4]+'_2.png', mask2*255)
        cv2.imwrite(out_filename[0:-4]+'_3.png', mask3*255)
        img_comb = img.copy()
        img_comb[mask3>0] = [0,255,0]
        cv2.imwrite(out_filename[0:-4]+'_comb.png', img_comb)
        
        logging.info(f'Mask saved to {out_filename}')
    if mode =='test':    
        result = []
        result.append(["Filename", "iou", "f1", "PRCC", "precision", "recall", "miou",'object_acc'])
        # print(patch_name)
        total_iou = 0
        total_miou = 0
        total_f1 = 0
        total_PRCC = 0
        total_precision = 0
        total_recall = 0
        total_object_acc = 0
        num_zero_img = 0
        for i, filename in enumerate(in_files):
#             print('=============',filename)
#             print('mask_gt_path',mask_gt_folder + filename[len(in_folder):-4]+ '_nuc_filled.png')
            img_path = '/home/msoc956131/final_unet_dann_v3/data/train/3d_leica_hamamasu' + filename[len(in_folder):-4]+ '.png'
            instance_path = '/home/msoc956131/final_unet_dann_v3/data/test/3d_v1_inside_mask' + filename[len(in_folder):-4]+ '_nuc_filled.png'
            mask_gt_path = mask_gt_folder + filename[len(in_folder):-4]+ '_nuc_filled.png'
            write_img_path = out_folder + filename[len(in_folder):-4]+ '_ori.png'
            mask_pred_path = out_folder + filename[len(in_folder):-4]+ '_3.png'
            mask_view = out_folder + filename[len(in_folder):-4]+ '_view'+ '.png'
            
            img_ = cv2.imread(img_path)
            instance_img = cv2.imread(instance_path)
            
            cnts, hierarchy = cv2.findContours(instance_img[:,:,0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            map_ = np.zeros((instance_img.shape[0],instance_img.shape[1]), np.uint8)
#             print('--------------',len(cnts))
            for c in cnts:
#                 print('````````````',c)
                M = cv2.moments(c)
#                 print('M["m10"]',M["m10"])
#                 print('M["m00"]',M["m00"])
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(map_, (cX, cY), 0, 1, -1)
            if len(cnts) == 0:
                num_zero_img +=1
#             print('mask_pred_path',mask_pred_path)
            mask_pred = cv2.imread(mask_pred_path,0)
            mask_gt = cv2.imread(mask_gt_path,0)
            mask_pred_ = np.zeros((mask_pred.shape[0],mask_pred.shape[1]), np.uint8)
            mask_pred_[mask_pred>0]=1
#             print('====',len(cnts))
#             print('---',np.sum(mask_pred_*map_))
#             print('+++++++++++++',np.sum(mask_pred_*map_)/len(cnts))
            object_acc = np.sum(mask_pred_*map_)/len(cnts)
#             print(mask_pred_path)
#             print(mask_gt_path)
            mask_gt[mask_gt>0] = 1
            mask_pred[mask_pred>0] = 1
            pict = img_.copy()
            mask_tp = np.where(mask_pred & mask_gt,1,0)
            mask_fp = np.where((1 - mask_gt) * mask_pred,1,0)
            mask_fn = np.where(mask_gt * (1 - mask_pred),1,0)
            pict[mask_tp==1] =[0,255,0]
            pict[mask_fp==1] =[255,0,0]
            pict[mask_fn==1] =[0,0,255]
            cv2.imwrite(mask_view, pict)
            cv2.imwrite(write_img_path, img_)
            
            iou, f1 , PRCC , precision , recall, miou = evaluate_all_inf(mask_pred, mask_gt)
            result.append([filename, iou, f1, PRCC, precision, recall, miou,object_acc])
            if len(cnts) != 0:
                total_iou +=iou
                total_miou +=miou
                total_f1 +=f1
                total_PRCC +=PRCC
                total_precision +=precision
                total_recall +=recall
                total_object_acc +=object_acc
            print('========',object_acc)
        print('========',(i+1))
        
        avg_iou =total_iou/(i+1-num_zero_img)
        avg_miou =total_miou/(i+1-num_zero_img)
        avg_f1 =total_f1/(i+1-num_zero_img)
        avg_PRCC =total_PRCC/(i+1-num_zero_img)
        avg_precision =total_precision/(i+1-num_zero_img)
        avg_recall =total_recall/(i+1-num_zero_img)
        avg_object_acc = total_object_acc/(i+1-num_zero_img)
        print('-----',avg_object_acc)
        result.append(['Average', avg_iou, avg_f1, avg_PRCC, avg_precision, avg_recall, avg_miou,avg_object_acc])
        with open(out_folder+'/metric.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(result)



# +
# if __name__ == '__main__':
#     args = get_args()
#     # in_files = args.input
#     in_folder = args.input_folder
#     out_folder = args.output_folder
#     mask_gt_folder = args.mask_gt_folder
#     mode = args.mode
#     print(args.output_folder)
#     if os.path.exists(out_folder)!=1:
#         os.mkdir(out_folder)
#     # out_files = get_output_filenames(args)

#     in_files = []
#     out_files = []
    
#     for f in os.listdir(in_folder):
#         in_files.append(os.path.join(in_folder, f))
#         out_files.append(os.path.join(out_folder, f))
    
#     net1 = ResUnetPlusPlus(n_channels=3, n_classes=2, bilinear=True)
#     net2 = ResUnetPlusPlus(n_channels=3, n_classes=2, bilinear=True)
#     ensemble_net = convolution_1x1()
# #     net = UNet(n_channels=3, n_classes=2)
# #     net = ResUnetPlusPlus(n_channels=3, n_classes=2, bilinear=True)
# #     net = AttentionUNet(n_channels=3, n_classes=2, bilinear=True)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logging.info(f'Loading model {args.model1}')
#     logging.info(f'Using device {device}')

    
#     net1.to(device=device)
#     net2.to(device=device)
#     ensemble_net.to(device=device)
#     net1.load_state_dict(torch.load(args.model1, map_location=device))
#     net2.load_state_dict(torch.load(args.model2, map_location=device))
#     ensemble_net.load_state_dict(torch.load(args.model_En, map_location=device))
#     logging.info('Model loaded!')
#     print('len(in_folder)',len(in_folder))
#     for i, filename in enumerate(in_files):
#         logging.info(f'\nPredicting image {filename} ...')
        
#         img = cv2.imread(filename)

#         mask1,mask2,mask3 = predict_img(net1=net1,
#                            net2=net2,
#                            ensemble_net=ensemble_net,
#                            full_img=img,
#                            scale_factor=args.scale,
#                            out_threshold=args.mask_threshold,
#                            device=device)
# #         if not args.no_save:
#         out_filename = out_files[i]
#         print(i,':',out_filename)
#         cv2.imwrite(out_filename[0:-4]+'_1.png', mask1*255)
#         cv2.imwrite(out_filename[0:-4]+'_2.png', mask2*255)
#         cv2.imwrite(out_filename[0:-4]+'_3.png', mask3*255)
#         logging.info(f'Mask saved to {out_filename}')
#     if mode =='test':    
#         result = []
#         result.append(["Filename", "iou", "f1", "precision", "recall", "miou",'object_acc'])
#         # print(patch_name)
#         total_iou = 0
#         total_miou = 0
#         total_f1 = 0
#         total_precision = 0
#         total_recall = 0
#         total_object_acc = 0
#         num_zero_img = 0
#         for i, filename in enumerate(in_files):
# #             print('=============',filename)
# #             print('mask_gt_path',mask_gt_folder + filename[len(in_folder):-4]+ '_nuc_filled.png')
#             img_path = '/home/msoc956131/final_unet_dann_v3/data/train/3d_leica_hamamasu' + filename[len(in_folder):-4]+ '.png'
#             instance_path = '/home/msoc956131/final_unet_dann_v3/data/test/3d_v1_inside_mask' + filename[len(in_folder):-4]+ '_nuc_filled.png'
#             mask_gt_path = mask_gt_folder + filename[len(in_folder):-4]+ '_nuc_filled.png'
#             write_img_path = out_folder + filename[len(in_folder):-4]+ '_ori.png'
#             mask_pred_path = out_folder + filename[len(in_folder):-4]+ '_3.png'
#             mask_view = out_folder + filename[len(in_folder):-4]+ '_view'+ '.png'
            
#             img_ = cv2.imread(img_path)
#             instance_img = cv2.imread(instance_path)
            
#             cnts, hierarchy = cv2.findContours(instance_img[:,:,0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#             map_ = np.zeros((instance_img.shape[0],instance_img.shape[1]), np.uint8)
# #             print('--------------',len(cnts))
#             for c in cnts:
# #                 print('````````````',c)
#                 M = cv2.moments(c)
# #                 print('M["m10"]',M["m10"])
# #                 print('M["m00"]',M["m00"])
#                 cX = int(M["m10"] / M["m00"])
#                 cY = int(M["m01"] / M["m00"])
#                 cv2.circle(map_, (cX, cY), 0, 1, -1)
#             if len(cnts) == 0:
#                 num_zero_img +=1
# #             print('mask_pred_path',mask_pred_path)
#             mask_pred = cv2.imread(mask_pred_path,0)
#             mask_gt = cv2.imread(mask_gt_path,0)
#             mask_pred_ = np.zeros((mask_pred.shape[0],mask_pred.shape[1]), np.uint8)
#             mask_pred_[mask_pred>0]=1
# #             print('====',len(cnts))
# #             print('---',np.sum(mask_pred_*map_))
# #             print('+++++++++++++',np.sum(mask_pred_*map_)/len(cnts))
#             object_acc = np.sum(mask_pred_*map_)/len(cnts)
# #             print(mask_pred_path)
# #             print(mask_gt_path)
#             mask_gt[mask_gt>0] = 1
#             mask_pred[mask_pred>0] = 1
#             pict = img_.copy()
#             mask_tp = np.where(mask_pred & mask_gt,1,0)
#             mask_fp = np.where((1 - mask_gt) * mask_pred,1,0)
#             mask_fn = np.where(mask_gt * (1 - mask_pred),1,0)
#             pict[mask_tp==1] =[0,255,0]
#             pict[mask_fp==1] =[255,0,0]
#             pict[mask_fn==1] =[0,0,255]
#             cv2.imwrite(mask_view, pict)
#             cv2.imwrite(write_img_path, img_)
            
#             iou, f1 , precision , recall, miou = evaluate_all_inf(mask_pred, mask_gt)
#             result.append([filename, iou, f1, precision, recall, miou,object_acc])
#             if len(cnts) != 0:
#                 total_iou +=iou
#                 total_miou +=miou
#                 total_f1 +=f1
#                 total_precision +=precision
#                 total_recall +=recall
#                 total_object_acc +=object_acc
#             print('========',object_acc)
#         print('========',(i+1))
        
#         avg_iou =total_iou/(i+1-num_zero_img)
#         avg_miou =total_miou/(i+1-num_zero_img)
#         avg_f1 =total_f1/(i+1-num_zero_img)
#         avg_precision =total_precision/(i+1-num_zero_img)
#         avg_recall =total_recall/(i+1-num_zero_img)
#         avg_object_acc = total_object_acc/(i+1-num_zero_img)
#         print('-----',avg_object_acc)
#         result.append(['Average', avg_iou, avg_f1, avg_precision, avg_recall, avg_miou,avg_object_acc])
#         with open(out_folder+'/metric.csv', 'w', newline='') as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerows(result)
        
        
        
# -





