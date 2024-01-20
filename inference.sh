# parser = argparse.ArgumentParser(description='Predict masks from input images')
# parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
#                     help='Specify the file in which the model is stored')
# parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images')
# parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', help='Filenames of output images')
# parser.add_argument('--input_folder', '-if', help='Foldername of input images')
# parser.add_argument('--output_folder', '-of', help='Foldername of output images')
# parser.add_argument('--viz', '-v', action='store_true',
#                     help='Visualize the images as they are processed')
# parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
# parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
#                     help='Minimum probability value to consider a mask pixel white')
# parser.add_argument('--scale', '-s', type=float, default=0.5,
#                     help='Scale factor for the input images')

# +
# modify                              ckpt & output_folder
# para_sp =0
# para_dann =0
# para_cau =0.1

# +

# python predict.py -m1 "/home/msoc956131/final_unet_dann_v3/checkpoints/checkpoints_1/4_min-shape-aware-sy-2-emsemble_double_NEW-AUG_us-IN_US_ResUnetPlusPlus3d-0.5-1/net1_checkpoint_epoch380.pth" \
#                     -m2 "/home/msoc956131/final_unet_dann_v3/checkpoints/checkpoints_1/4_min-shape-aware-sy-2-emsemble_double_NEW-AUG_us-IN_US_ResUnetPlusPlus3d-0.5-1/net2_checkpoint_epoch380.pth" \
#                     -m3 "/home/msoc956131/final_unet_dann_v3/checkpoints/checkpoints_1/4_min-shape-aware-sy-2-emsemble_double_NEW-AUG_us-IN_US_ResUnetPlusPlus3d-0.5-1/ensemble_net_checkpoint_epoch380.pth" \
#                     -mo "test" --input_folder "/home/msoc956131/final_unet_dann_v3/data/test/3d_v1" \
#                     --mask_gt_folder "/home/msoc956131/final_unet_dann_v3/data/test/3d_v1_inside_mask" \
#                     --output_folder "/home/msoc956131/inf_proposed" --mask_threshold 0
python predict.py -m1 "/home/msoc956131/final_unet_dann_v3/checkpoints/checkpoints_1/4_min-shape-aware-sy-2-emsemble_double_NEW-AUG_us-IN_US_ResUnetPlusPlus3d-0.5-1/net1_checkpoint_epoch380.pth" \
                    -m2 "/home/msoc956131/final_unet_dann_v3/checkpoints/checkpoints_1/4_min-shape-aware-sy-2-emsemble_double_NEW-AUG_us-IN_US_ResUnetPlusPlus3d-0.5-1/net2_checkpoint_epoch380.pth" \
                    -m3 "/home/msoc956131/final_unet_dann_v3/checkpoints/checkpoints_1/4_min-shape-aware-sy-2-emsemble_double_NEW-AUG_us-IN_US_ResUnetPlusPlus3d-0.5-1/ensemble_net_checkpoint_epoch380.pth" \
                    -mo "test" --input_folder "/home/msoc956131/zzzz/data" \
                    --mask_gt_folder "/home/msoc956131/final_unet_dann_v3/data/test/3d_v1_inside_mask" \
                    --output_folder "/home/msoc956131/zzzz/aa" --mask_threshold 0
