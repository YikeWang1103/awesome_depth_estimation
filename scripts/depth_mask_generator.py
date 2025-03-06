import sys, os
sys.path.append(os.getcwd())

from inference.depth_estimator import DepthEstimator
from utils.toolchain import add_mask_overlay, mask2colormap
import argparse
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt

def arg_parser():
    parser = argparse.ArgumentParser(description="Depth Mask Generator")
    parser.add_argument('--dumped_file_folder', type=str, required=True, help='Path to the input image')
    parser.add_argument('--output_folder_name', type=str, default='sky_mask', help='Path to save the output depth mask')
    parser.add_argument('--model_name', type=str, required=True, help='Model chosed for depth estimation')
    parser.add_argument('--depth_threshold', type=float, default=35.0, help='Threshold in depth for mask generation')
    parser.add_argument('--save_depth_image', action="store_true", help='Save overlayed image as a png image')
    parser.add_argument('--save_overlayed_mask', action="store_true", help='Save overlayed image as a png image')
    
    return parser.parse_args()

def main(dumped_file_folder, output_folder_name, model_name, depth_threshold, save_depth_image, save_overlayed_mask):
    depth_estimator = DepthEstimator(model_name)

    sequences = [seq for seq in os.listdir(dumped_file_folder) if os.path.isdir(os.path.join(dumped_file_folder,seq))]

    sequences.sort()

    print(f"sequences: {sequences}")

    for seq in tqdm(sequences, desc="Processing sequences"):
        seq_path = os.path.join(dumped_file_folder, seq)

        output_folder_path = os.path.join(seq_path, output_folder_name)
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        rgb_image_folder_path = os.path.join(seq_path, 'input_rgb_0')
        rgb_image_files = [rgb_image_file for rgb_image_file in os.listdir(rgb_image_folder_path) if rgb_image_file.endswith('.png')]

        for rgb_image_file in tqdm(rgb_image_files, desc="  Processing images"):
            rgb_image_file_path = os.path.join(rgb_image_folder_path, rgb_image_file)
            depth_image_tensor = depth_estimator.inference(rgb_image_file_path)

            # generate sky mask using depth_image
            depth_image_npy = depth_image_tensor.cpu().numpy()
            depth_mask = (depth_image_npy > depth_threshold).astype(int)

            # generate overlayed image using rgb_image and depth_mask
            if save_overlayed_mask:
                overlayed_image_folder_path = os.path.join(output_folder_path, 'rgb')
                if not os.path.exists(overlayed_image_folder_path):
                    os.makedirs(overlayed_image_folder_path)

                rgb_image = cv2.imread(rgb_image_file_path)
                overlayed_image = add_mask_overlay(rgb_image, depth_mask, (0,0,255), 0.3)
                # overlayed_image = add_mask_contour(overlayed_image, depth_mask, (0,255,255), 2)
                # overlayed_image = cv2.hconcat([rgb_image, overlayed_image])
                # 保存带有轮廓的叠加图像
                overlayed_image_path = os.path.join(overlayed_image_folder_path, rgb_image_file)
                cv2.imwrite(overlayed_image_path, overlayed_image)

            if save_depth_image:
                depth_image_folder_path = os.path.join(output_folder_path, 'depth')
                if not os.path.exists(depth_image_folder_path):
                    os.makedirs(depth_image_folder_path)
                
                depth_image_path = os.path.join(depth_image_folder_path, os.path.splitext(rgb_image_file)[0] + '.npy')
                np.save(depth_image_path, depth_image_npy)


            # save depth_mask in .npy format
            depth_mask_folder_path = os.path.join(output_folder_path, 'mask')
            if not os.path.exists(depth_mask_folder_path):
                os.makedirs(depth_mask_folder_path)

            depth_mask_path = os.path.join(depth_mask_folder_path, os.path.splitext(rgb_image_file)[0] + '.npy')
            np.save(depth_mask_path, depth_mask)


if __name__ == "__main__":
    args = arg_parser()

    main(args.dumped_file_folder, 
         args.output_folder_name, 
         args.model_name, 
         args.depth_threshold,
         args.save_depth_image,
         args.save_overlayed_mask)

    
    