import math
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from natsort import natsorted


import argparse
from tqdm import tqdm
import numpy as np
import torch
import cv2
from PIL import Image
from glob import glob
from pycocotools import mask as masktool
from lib.pipeline.masked_droid_slam import *
from lib.pipeline.est_scale import *
from hawor.utils.process import block_print, enable_print
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__) + '/thirdparty/Metric3D')
from metric import Metric3D

video_folder = "/mnt/homes/kefan-ldap/HaWoR/example/video2"
output_dir = "/mnt/homes/kefan-ldap/HaWoR/example/video2/"

img_folder = f'{video_folder}/extracted_images'
imgfiles = natsorted(glob(f'{img_folder}/*.jpg'))

focal = None
if focal is None:
        try:
            with open(os.path.join(video_folder, 'est_focal.txt'), 'r') as file:
                focal = file.read()
                focal = float(focal)
        except:
            
            print('No focal length provided')
            focal = 600
            with open(os.path.join(video_folder, 'est_focal.txt'), 'w') as file:
                file.write(str(focal))
calib = np.array(est_calib(imgfiles)) # [focal, focal, cx, cy]
center = calib[2:]        
calib[:2] = focal
H, W = get_dimention(imgfiles)

# Estimate scale  
block_print()  
metric = Metric3D('thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth') 
enable_print() 

pred_depths = []
for t in range(len(imgfiles)):
    pred_depth = metric(imgfiles[t], calib)
    pred_depth = cv2.resize(pred_depth, (W, H))
    pred_depths.append(pred_depth)

if output_dir:
        depth_output_dir = os.path.join(output_dir, 'depth')
        os.makedirs(depth_output_dir, exist_ok=True)
        
        # for i, depth in enumerate(pred_depths):  # 保存前n帧
        #     # depth = pred_depths[frame_idx]
            
        #     depth_min = depth.min()
        #     depth_max = depth.max()
            
        #     # 保存伪彩色深度图
        #     depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-6)
        #     depth_colored = plt.cm.plasma(depth_normalized)
        #     depth_rgb = (depth_colored[:, :, :3] * 255).astype(np.uint8)
            
        #     depth_path = os.path.join(depth_output_dir, f'depth_frame_{i}.png')
        #     cv2.imwrite(depth_path, cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR))
            
        #     # 保存原始深度值（16位）
        #     depth_16bit = (depth * 1000).astype(np.uint16)  # 转换为毫米
        #     depth_raw_path = os.path.join(depth_output_dir, f'depth_frame_{i}_raw.png')
        #     cv2.imwrite(depth_raw_path, depth_16bit)
        
        # print(f"Saved individual depth maps to {depth_output_dir}")
        for i, depth in enumerate(pred_depths):
            np.savez_compressed(f"{depth_output_dir}/depth_{i}.npz", depth=depth)