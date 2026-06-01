import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import joblib
# data = np.load('/mnt/homes/kefan-ldap/HaWoR/example/video_0/SLAM/hawor_slam_w_scale_0_121.npz')
# for k in data.files:
#     arr = data[k]
#     print(k, arr.shape, arr.dtype)

data = joblib.load("/mnt/homes/kefan-ldap/HaWoR/example/24fps/world_space_res.pth")
print(data)

# traj = data['traj']  # shape: (N,7)
# scale = data['scale']

# # 拆解
# trans = traj[:, :3] * scale         # 平移向量
# quat = traj[:, 3:]                  # 四元数

# # 转换为旋转矩阵
# rots = R.from_quat(quat).as_matrix()  # (N,3,3)

# # 拼成 4x4 齐次矩阵
# traj_4x4 = np.tile(np.eye(4), (len(traj), 1, 1))
# traj_4x4[:, :3, :3] = rots
# traj_4x4[:, :3, 3] = trans

# positions = traj_4x4[:, :3, 3]*scale

# # 创建2x2子图布局，展示不同视角
# fig = plt.figure(figsize=(16, 12))

# # 辅助函数：绘制轨迹和标记
# def plot_trajectory(ax, x, y, z=None, title='', xlabel='X', ylabel='Y', zlabel='Z', is_3d=False):
#     if is_3d:
#         # 3D图
#         ax.plot(x, y, z, color='b', linewidth=1.5, label='Camera trajectory', alpha=0.7)
#         ax.scatter(x[0], y[0], z[0], color='red', s=300, marker='*', label='Start', 
#                    edgecolors='darkred', linewidths=2, zorder=5)
#         ax.scatter(x[-1], y[-1], z[-1], color='green', s=200, marker='o', label='End', 
#                    edgecolors='darkgreen', linewidths=2, zorder=5)
#         ax.set_xlabel(xlabel)
#         ax.set_ylabel(ylabel)
#         ax.set_zlabel(zlabel)
#     else:
#         # 2D图
#         ax.plot(x, y, color='b', linewidth=1.5, label='Camera trajectory', alpha=0.7)
#         ax.scatter(x[0], y[0], color='red', s=300, marker='*', label='Start', 
#                    edgecolors='darkred', linewidths=2, zorder=5)
#         ax.scatter(x[-1], y[-1], color='green', s=200, marker='o', label='End', 
#                    edgecolors='darkgreen', linewidths=2, zorder=5)
#         ax.set_xlabel(xlabel)
#         ax.set_ylabel(ylabel)
#         ax.grid(True, alpha=0.3)
#         ax.set_aspect('equal', adjustable='box')
#     ax.set_title(title, fontsize=12, fontweight='bold')
#     ax.legend()

# # 1. 左上：3D视角（默认）
# ax1 = fig.add_subplot(2, 2, 1, projection='3d')
# plot_trajectory(ax1, positions[:, 0], positions[:, 1], positions[:, 2], 
#                 '3D View (Default)', 'X', 'Y', 'Z', is_3d=True)
# ax1.view_init(elev=20, azim=45)

# # 2. 右上：XY平面视图（俯视图，从Z轴方向看）
# ax2 = fig.add_subplot(2, 2, 2)
# plot_trajectory(ax2, positions[:, 0], positions[:, 1], 
#                 'Top View (XY Plane)', 'X', 'Y')

# # 3. 左下：XZ平面视图（侧视图，从Y轴方向看）
# ax3 = fig.add_subplot(2, 2, 3)
# plot_trajectory(ax3, positions[:, 0], positions[:, 2], 
#                 'Side View (XZ Plane)', 'X', 'Z')

# # 4. 右下：YZ平面视图（前视图，从X轴方向看）
# ax4 = fig.add_subplot(2, 2, 4)
# plot_trajectory(ax4, positions[:, 1], positions[:, 2], 
#                 'Front View (YZ Plane)', 'Y', 'Z')

# plt.suptitle('HaWoR SLAM Trajectory - Multiple Views', fontsize=16, fontweight='bold', y=0.995)
# plt.tight_layout()
# plt.savefig('HaWoR_SLAM_Trajectory.png', dpi=150, bbox_inches='tight')
# plt.show()