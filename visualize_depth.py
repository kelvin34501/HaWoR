import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob
from natsort import natsorted

def visualize_depth_maps(slam_file, output_dir=None, num_frames=None, overlay_images=True):
    """
    可视化 hawor_slam 生成的深度图
    
    Args:
        slam_file: SLAM 输出的 npz 文件路径
        output_dir: 输出目录，如果为 None 则不保存
        num_frames: 要可视化的帧数，如果为 None 则显示所有帧或选择关键帧
        overlay_images: 是否叠加原始图像
    """
    # 加载 SLAM 数据
    print(f"Loading SLAM data from {slam_file}...")
    data = np.load(slam_file)
    
    # disps 的形状是 (N, H, W)，其中：
    # - N: 关键帧数量
    # - H: 图像高度（像素）
    # - W: 图像宽度（像素）
    # 每个像素都有自己独立的视差值！
    disps = data['disps']  # 视差图 (N, H, W)，每个像素都有视差值
    tstamp = data['tstamp']  # 时间戳
    scale = data.get('scale', 1.0)  # 尺度
    
    print(f"Loaded {len(disps)} depth maps")
    print(f"Disparity map shape per frame: {disps[0].shape} (H={disps[0].shape[0]}, W={disps[0].shape[1]})")
    print(f"Total pixels per frame: {disps[0].shape[0] * disps[0].shape[1]}")
    print(f"Each pixel has its own disparity value!")
    
    # ============================================
    # 视差转换为深度的逻辑说明
    # ============================================
    # 1. 视差（Disparity）的概念：
    #    - 视差是双目视觉中两个相机看到同一物体时，该物体在图像中的水平位置差
    #    - 在单目 SLAM 中，视差表示的是"逆深度"（inverse depth），即 1/depth
    #    - 视差越大，物体越近；视差越小，物体越远
    #
    # 2. 基本转换公式：
    #    depth = 1 / disparity
    #    这是最简单的反比关系：视差越大 → 深度越小（物体越近）
    #
    # 3. 尺度因子（Scale）的作用：
    #    - SLAM 系统只能估计相对深度，无法直接得到真实度量单位的深度（米/厘米）
    #    - 需要通过 Metric3D 等单目深度估计网络来估计尺度因子
    #    - 真实深度 = 相对深度 × 尺度因子
    #
    # 4. 完整转换公式：
    #    depth_real = (1 / disparity) × scale
    #
    # 5. 数值稳定性处理：
    #    - 添加小的 epsilon (1e-6) 避免除零错误
    #    - 当视差为 0 或接近 0 时，深度会非常大（表示无限远）
    # ============================================
    
    # 对每个关键帧的视差图进行逐像素转换
    # disp 的形状是 (H, W)，每个像素都有视差值
    # 转换后的 depth 也是 (H, W)，每个像素都有深度值
    depths = []
    for disp in disps:
        # 视差转换为深度（逐像素操作）：
        # - disp 是形状为 (H, W) 的数组，每个像素都有视差值
        # - 1.0 / disp: 对每个像素执行反比关系（视差 → 逆深度）
        # - 添加 1e-6: 防止除零错误，提高数值稳定性
        # - × scale: 将相对深度转换为真实度量深度（米）
        # 结果：depth 也是形状为 (H, W) 的数组，每个像素都有深度值
        depth = 1.0 / (disp + 1e-6) * scale
        depths.append(depth)
    depths = np.array(depths)  # 最终形状: (N, H, W)
    print(f"Converted to depth maps with shape: {depths.shape}")
    
    # 确定要显示的帧
    if num_frames is None:
        # 默认显示前几个关键帧（均匀采样）
        total_frames = len(disps)
        num_frames = min(9, total_frames)  # 显示最多9帧
    
    # 均匀采样帧索引
    frame_indices = np.linspace(0, len(disps) - 1, num_frames, dtype=int)
    
    # 尝试加载对应的图像
    images = None
    if overlay_images:
        # 尝试从 SLAM 文件所在目录查找图像
        slam_dir = os.path.dirname(slam_file)
        video_dir = os.path.dirname(slam_dir)
        img_folder = os.path.join(video_dir, 'extracted_images')
        
        if os.path.exists(img_folder):
            imgfiles = natsorted(glob(os.path.join(img_folder, '*.jpg')))
            if len(imgfiles) > 0:
                images = []
                for idx in frame_indices:
                    if idx < len(imgfiles):
                        img = cv2.imread(imgfiles[tstamp[idx]])
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            # 调整图像大小以匹配深度图
                            if img.shape[:2] != disps[0].shape:
                                img = cv2.resize(img, (disps[0].shape[1], disps[0].shape[0]))
                            images.append(img)
                        else:
                            images.append(None)
                    else:
                        images.append(None)
                print(f"Loaded {len([x for x in images if x is not None])} images for overlay")
            else:
                print("No images found for overlay")
        else:
            print(f"Image folder not found: {img_folder}")
    
    # 创建可视化
    cols = 3
    rows = (num_frames + cols - 1) // cols
    fig = plt.figure(figsize=(18, 6 * rows))
    
    for i, frame_idx in enumerate(frame_indices):
        disp = disps[frame_idx]
        depth = depths[frame_idx]
        
        # 计算深度统计信息
        valid_mask = disp > 0
        if valid_mask.sum() > 0:
            depth_min = depth[valid_mask].min()
            depth_max = depth[valid_mask].max()
            depth_mean = depth[valid_mask].mean()
        else:
            depth_min, depth_max, depth_mean = 0, 0, 0
        
        # 创建子图
        if overlay_images and images and images[i] is not None:
            # 如果有图像，创建2列：原始图像+深度图，图像+深度叠加
            ax1 = fig.add_subplot(rows, cols * 2, i * 2 + 1)
            ax2 = fig.add_subplot(rows, cols * 2, i * 2 + 2)
            
            # 显示原始图像
            ax1.imshow(images[i])
            ax1.set_title(f'Frame {tstamp[frame_idx]} (Original)', fontsize=10)
            ax1.axis('off')
            
            # 显示深度图叠加在图像上
            ax2.imshow(images[i], alpha=0.5)
            depth_colored = plt.cm.plasma(depth / (depth_max + 1e-6))
            ax2.imshow(depth_colored, alpha=0.5)
            ax2.set_title(f'Frame {tstamp[frame_idx]} (Depth Overlay)\n'
                         f'Depth: {depth_min:.2f}-{depth_max:.2f}m (mean: {depth_mean:.2f}m)', 
                         fontsize=10)
            ax2.axis('off')
        else:
            # 只显示深度图
            ax = fig.add_subplot(rows, cols, i + 1)
            im = ax.imshow(depth, cmap='plasma', vmin=depth_min, vmax=depth_max)
            ax.set_title(f'Frame {tstamp[frame_idx]} - Depth Map\n'
                        f'Range: {depth_min:.2f}-{depth_max:.2f}m (mean: {depth_mean:.2f}m)', 
                        fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(f'HaWoR SLAM Depth Maps (Scale: {scale:.4f})', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # 保存图像
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'slam_depth_maps.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved depth visualization to {output_path}")
    
    plt.show()
    
    # 同时保存一些单独的深度图
    if output_dir:
        depth_output_dir = os.path.join(output_dir, 'depth_maps')
        os.makedirs(depth_output_dir, exist_ok=True)
        
        for i, frame_idx in enumerate(frame_indices[:5]):  # 保存前5帧
            depth = depths[frame_idx]
            valid_mask = disps[frame_idx] > 0
            
            if valid_mask.sum() > 0:
                depth_min = depth[valid_mask].min()
                depth_max = depth[valid_mask].max()
                
                # 保存伪彩色深度图
                depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-6)
                depth_colored = plt.cm.plasma(depth_normalized)
                depth_rgb = (depth_colored[:, :, :3] * 255).astype(np.uint8)
                
                depth_path = os.path.join(depth_output_dir, f'depth_frame_{tstamp[frame_idx]}.png')
                cv2.imwrite(depth_path, cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR))
                
                # 保存原始深度值（16位）
                depth_16bit = (depth * 1000).astype(np.uint16)  # 转换为毫米
                depth_raw_path = os.path.join(depth_output_dir, f'depth_frame_{tstamp[frame_idx]}_raw.png')
                cv2.imwrite(depth_raw_path, depth_16bit)
        
        print(f"Saved individual depth maps to {depth_output_dir}")


if __name__ == '__main__':
    # 默认路径
    slam_file = '/mnt/homes/kefan-ldap/HaWoR/example/24fps/SLAM/hawor_slam_w_scale_0_121.npz'
    
    # 输出目录
    output_dir = '/mnt/homes/kefan-ldap/HaWoR/example/24fps/SLAM/depth_visualization'
    
    # 可视化深度图
    visualize_depth_maps(
        slam_file=slam_file,
        output_dir=output_dir,
        num_frames=9,  # 显示9帧
        overlay_images=True  # 叠加原始图像
    )

