import os
import shutil
import cv2
import numpy as np
import torch
import trimesh

import lib.vis.viewer as viewer_utils
from lib.vis.wham_tools.tools import checkerboard_geometry


def _dims_from_images(image_names):
    """(height, width) from a FrameSource/subset (frame_shape) or a path list."""
    if hasattr(image_names, "frame_shape"):
        h, w = image_names.frame_shape
        return int(h), int(w)
    img0 = cv2.imread(image_names[0])
    return img0.shape[0], img0.shape[1]


def _materialize_frames(image_names, output_pth):
    """Make on-disk paths for the billboard background.

    The camera-space billboard needs image *files*. When given a FrameSource (RGB
    frames decoded on demand), write just the vis-range frames to a temp dir and
    return (paths, tmp_dir) so the caller can delete it afterwards. For a legacy
    path list, return it unchanged with tmp_dir=None.
    """
    if not hasattr(image_names, "frame_shape"):
        return list(image_names), None
    tmp_dir = os.path.join(output_pth, "_vis_frames")
    os.makedirs(tmp_dir, exist_ok=True)
    paths = []
    for i in range(len(image_names)):
        frame_rgb = image_names[i]
        p = os.path.join(tmp_dir, f"{i:06d}.jpg")
        cv2.imwrite(p, frame_rgb[:, :, ::-1])  # back to BGR for cv2.imwrite
        paths.append(p)
    return paths, tmp_dir


def camera_marker_geometry(radius, height):
    vertices = np.array(
        [
            [-radius, -radius, 0],
            [radius, -radius, 0],
            [radius, radius, 0],
            [-radius, radius, 0],
            [0, 0, - height],
        ]
    )


    faces = np.array(
        [[0, 1, 2], [0, 2, 3], [1, 0, 4], [2, 1, 4], [3, 2, 4], [0, 3, 4],]
    )

    face_colors = np.array(
        [
            [0.5, 0.5, 0.5, 1.0],
            [0.5, 0.5, 0.5, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ]
    )
    return vertices, faces, face_colors


def _edge_box(p0, p1, radius):
    """Thin box around segment p0->p1 (a poor man's 3D line, culling-proof)."""
    d = p1 - p0
    d = d / np.linalg.norm(d)
    helper = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(d, helper)
    u = u / np.linalg.norm(u)
    w = np.cross(d, u)
    verts = np.array([p + radius * (su * u + sw * w)
                      for p in (p0, p1)
                      for su, sw in ((-1, -1), (-1, 1), (1, 1), (1, -1))])
    quads = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    faces = np.array([tri for a, b, c, d_ in quads for tri in ((a, b, c), (a, c, d_))])
    return verts, faces


def camera_frustum_geometry(width, height, focal, depth):
    """Wireframe camera frustum: apex at the optical center, image-plane
    rectangle (aspect and FoV from K) at z=depth, 8 edges drawn as thin boxes.
    OpenCV convention (+z forward, +y down); the image-top edge is green and
    the image-bottom edge red so roll is readable. Render double-sided."""
    x = 0.5 * width / focal * depth
    y = 0.5 * height / focal * depth
    r = max(0.015 * depth, 1e-4)
    apex = np.zeros(3)
    c = np.array([[-x, -y, depth], [x, -y, depth], [x, y, depth], [-x, y, depth]])
    gray = [0.55, 0.55, 0.6, 1.0]
    green = [0.1, 0.8, 0.2, 1.0]
    red = [0.85, 0.2, 0.1, 1.0]
    edges = [
        (apex, c[0], gray), (apex, c[1], gray), (apex, c[2], gray), (apex, c[3], gray),
        (c[0], c[1], green),  # image-top edge
        (c[1], c[2], gray),
        (c[2], c[3], red),    # image-bottom edge
        (c[3], c[0], gray),
    ]
    all_v, all_f, all_fc = [], [], []
    offset = 0
    for p0, p1, col in edges:
        v, f = _edge_box(np.asarray(p0, dtype=float), np.asarray(p1, dtype=float), r)
        all_v.append(v)
        all_f.append(f + offset)
        all_fc.append(np.tile(np.array(col)[None, :], (len(f), 1)))
        offset += len(v)
    return np.concatenate(all_v), np.concatenate(all_f), np.concatenate(all_fc)


def world_axes_geometry(length=0.3, radius=0.008):
    """RGB triad at the world origin: +X red, +Y green, +Z blue (thin boxes)."""
    colors = [(0.9, 0.1, 0.1, 1.0), (0.1, 0.8, 0.1, 1.0), (0.15, 0.25, 0.9, 1.0)]
    box_faces = np.array(
        [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
         [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    )
    all_v, all_f, all_fc = [], [], []
    for axis, color in enumerate(colors):
        lo = np.full(3, -radius)
        hi = np.full(3, radius)
        lo[axis], hi[axis] = 0.0, length
        corners = np.array([[x, y, z]
                            for x in (lo[0], hi[0])
                            for y in (lo[1], hi[1])
                            for z in (lo[2], hi[2])])
        all_v.append(corners)
        all_f.append(box_faces + 8 * axis)
        all_fc.append(np.tile(np.array(color)[None, :], (len(box_faces), 1)))
    return np.concatenate(all_v), np.concatenate(all_f), np.concatenate(all_fc)


def follow_camera_rt(R_c2w, t_c2w, offset_rub=(0.0, 0.4, 1.2)):
    """Per-frame OpenCV w2c extrinsics for a camera rigidly attached to the
    SLAM camera: both translation AND rotation follow. ``offset_rub`` is
    (right, up, back) in the SLAM camera's own frame; the chase camera keeps
    the SLAM camera's orientation (over-the-shoulder view).
    """
    R = np.asarray(R_c2w, dtype=np.float64)  # (T, 3, 3)
    t = np.asarray(t_c2w, dtype=np.float64)  # (T, 3)
    r_, u_, b_ = offset_rub
    # camera-local OpenCV axes: x right, y down, z forward
    off_local = np.array([r_, -u_, -b_], dtype=np.float64)
    pos = t + np.einsum('tij,j->ti', R, off_local)
    R_w2c = R.transpose(0, 2, 1)
    Rt = np.zeros((len(t), 3, 4), dtype=np.float32)
    Rt[:, :3, :3] = R_w2c
    Rt[:, :3, 3] = -np.einsum('tij,tj->ti', R_w2c, pos)
    return Rt


def run_vis2_on_video(res_dict,
                      res_dict2,
                      output_pth,
                      focal_length,
                      image_names,
                      R_c2w=None,
                      t_c2w=None,
                      interactive=True,
                      show_traj=False,
                      show_ghost=False,
                      show_frustum=False,
                      frustum_depth=0.2,
                      show_axes=False,
                      axes_length=0.3,
                      show_ground=True,
                      ground_up="z",
                      ground_height=-2.0,
                      follow_camera=False,
                      follow_offset=(0.0, 0.4, 1.2),
                      points=None,
                      ghost_stride=35,
                      ghost_count=4,
                      ghost_alpha_decay=0.1,
                      ghost_alpha_min=0.05):

    # World-space viz does not overlay the source frames; only dimensions are needed.
    height, width = _dims_from_images(image_names)

    world_mano = {}
    world_mano['vertices'] = res_dict['vertices']
    world_mano['faces'] = res_dict['faces']

    world_mano2 = {}
    world_mano2['vertices'] = res_dict2['vertices']
    world_mano2['faces'] = res_dict2['faces']

    vis_dict = {}
    color_idx = 0
    world_mano['vertices'] = world_mano['vertices']
    for _id, _verts in enumerate(world_mano['vertices']):
        verts = _verts.cpu().numpy() # T, N, 3
        body_faces = world_mano['faces']
        body_meshes = {
            "v3d": verts,
            "f3d": body_faces,
            "vc": None,
            "name": f"hand_{_id}",
            # "color": "pace-green",
            "color": "director-purple",
        }
        vis_dict[f"hand_{_id}"] = body_meshes
        color_idx += 1
        # add trajectory markers (dotted trail) for this hand (optional)
        if show_traj:
            traj = verts.mean(axis=1)  # (T, 3)
            base_v, base_f, base_fc = camera_marker_geometry(0.01, 0.02)
            nv = base_v.shape[0]
            nf = base_f.shape[0]
            all_v = []
            all_f = []
            all_fc = []
            for i in range(traj.shape[0]):
                offset = i * nv
                v_i = base_v + traj[i][None, :]
                f_i = base_f + offset
                all_v.append(v_i)
                all_f.append(f_i)
                all_fc.append(base_fc)
            all_v = np.concatenate(all_v, axis=0)  # (T*nv, 3)
            all_f = np.concatenate(all_f, axis=0)  # (T*nf, 3)
            all_fc = np.concatenate(all_fc, axis=0)  # (T*nf, 4)
            T = verts.shape[0]
            v3d_traj = np.tile(all_v[None, :, :], (T, 1, 1))
            traj_mesh = {
                "v3d": v3d_traj,
                "f3d": all_f,
                "vc": None,
                "name": f"hand_{_id}_traj",
                "fc": all_fc,
                "color": -1,
            }
            vis_dict[f"hand_{_id}_traj"] = traj_mesh
        # add ghost hands every few frames with fading color
        if show_ghost:
            base_rgba = np.array([0.804, 0.6, 0.820, 1.0], dtype=np.float32)
            for g in range(1, ghost_count + 1):
                lag = g * ghost_stride
                ghost_v = []
                for t in range(verts.shape[0]):
                    src_idx = max(t - lag, 0)
                    ghost_v.append(verts[src_idx])
                ghost_v = np.stack(ghost_v, axis=0)
                alpha = max(ghost_alpha_min, (ghost_alpha_decay**g) * base_rgba[3])
                ghost_color = np.concatenate([base_rgba[:3], [alpha]], axis=0)
                face_colors = np.tile(ghost_color[None, :], (body_faces.shape[0], 1))
                ghost_mesh = {
                    "v3d": ghost_v,
                    "f3d": body_faces,
                    "vc": None,
                    "name": f"hand_{_id}_ghost_{g}",
                    "fc": face_colors,
                    "color": -1,
                }
                vis_dict[ghost_mesh["name"]] = ghost_mesh

    world_mano2['vertices'] = world_mano2['vertices']
    for _id, _verts in enumerate(world_mano2['vertices']):
        verts = _verts.cpu().numpy() # T, N, 3
        body_faces = world_mano2['faces']
        body_meshes = {
            "v3d": verts,
            "f3d": body_faces,
            "vc": None,
            "name": f"hand2_{_id}",
            # "color": "pace-blue",
            "color": "director-blue",
        }
        vis_dict[f"hand2_{_id}"] = body_meshes
        color_idx += 1
        if show_traj:
            traj = verts.mean(axis=1)  # (T, 3)
            base_v, base_f, base_fc = camera_marker_geometry(0.01, 0.02)
            nv = base_v.shape[0]
            nf = base_f.shape[0]
            all_v = []
            all_f = []
            all_fc = []
            for i in range(traj.shape[0]):
                offset = i * nv
                v_i = base_v + traj[i][None, :]
                f_i = base_f + offset
                all_v.append(v_i)
                all_f.append(f_i)
                all_fc.append(base_fc)
            all_v = np.concatenate(all_v, axis=0)
            all_f = np.concatenate(all_f, axis=0)
            all_fc = np.concatenate(all_fc, axis=0)
            T = verts.shape[0]
            v3d_traj = np.tile(all_v[None, :, :], (T, 1, 1))
            traj_mesh = {
                "v3d": v3d_traj,
                "f3d": all_f,
                "vc": None,
                "name": f"hand2_{_id}_traj",
                "fc": all_fc,
                "color": -1,
            }
            vis_dict[f"hand2_{_id}_traj"] = traj_mesh
        if show_ghost:
            base_rgba = np.array([0.207, 0.596, 0.792, 1.0], dtype=np.float32)
            for g in range(1, ghost_count + 1):
                lag = g * ghost_stride
                ghost_v = []
                for t in range(verts.shape[0]):
                    src_idx = max(t - lag, 0)
                    ghost_v.append(verts[src_idx])
                ghost_v = np.stack(ghost_v, axis=0)
                alpha = max(ghost_alpha_min, (ghost_alpha_decay**g) * base_rgba[3])
                ghost_color = np.concatenate([base_rgba[:3], [alpha]], axis=0)
                face_colors = np.tile(ghost_color[None, :], (body_faces.shape[0], 1))
                ghost_mesh = {
                    "v3d": ghost_v,
                    "f3d": body_faces,
                    "vc": None,
                    "name": f"hand2_{_id}_ghost_{g}",
                    "fc": face_colors,
                    "color": -1,
                }
                vis_dict[ghost_mesh["name"]] = ghost_mesh

    if show_ground:
        v, f, vc, fc = checkerboard_geometry(length=100, c1=0, c2=0, up=ground_up)
        v[:, 1 if ground_up == "y" else 2] += ground_height
        gound_meshes = {
            "v3d": v,
            "f3d": f,
            "vc": vc,
            "name": "ground",
            "fc": fc,
            "color": -1,
            "double_sided": True,
        }
        vis_dict["ground"] = gound_meshes

    if show_axes:
        av, af, afc = world_axes_geometry(length=axes_length)
        vis_dict["world_axes"] = {
            "v3d": av,
            "f3d": af,
            "vc": None,
            "name": "world_axes",
            "fc": afc,
            "color": -1,
            "double_sided": True,
        }

    num_frames = len(world_mano['vertices'][_id])
    Rt = np.zeros((num_frames, 3, 4))
    Rt[:, :3, :3] = R_c2w[:num_frames]
    Rt[:, :3, 3] = t_c2w[:num_frames]

    if show_frustum:
        f_px = float(focal_length) if focal_length else float(max(height, width))
        verts, faces, face_colors = camera_frustum_geometry(width, height, f_px, frustum_depth)
    else:
        verts, faces, face_colors = camera_marker_geometry(0.05, 0.1)
    verts = np.einsum("tij,nj->tni", Rt[:, :3, :3], verts) + Rt[:, None, :3, 3]
    camera_meshes = {
        "v3d": verts,
        "f3d": faces,
        "vc": None,
        "name": "camera",
        "fc": face_colors,
        "color": -1,
        "double_sided": True,
    }
    vis_dict["camera"] = camera_meshes

    if follow_camera:
        # Chase camera through the fixed world, rigidly attached to the SLAM
        # camera (translation + rotation). setup_billboard turns this Rt into
        # the scene's active OpenCVCamera; the free camera is one click away.
        viewer_Rt = follow_camera_rt(np.asarray(R_c2w[:num_frames]),
                                     np.asarray(t_c2w[:num_frames]), follow_offset)
    else:
        side_source = torch.tensor([0.463, -0.478, 2.456])
        side_target = torch.tensor([0.026, -0.481, -3.184])
        up = torch.tensor([0.0, 1.0, 0.0]) if ground_up == "y" else torch.tensor([1.0, 0.0, 0.0])
        view_camera = lookat_matrix(side_source, side_target, up)
        viewer_Rt = np.tile(view_camera[:3, :4], (num_frames, 1, 1))

    meshes = viewer_utils.construct_viewer_meshes(
        vis_dict, draw_edges=False, flat_shading=False
    )

    if points is not None:
        # (F, N, 3) per-frame SLAM depth point cloud; any renderable works in
        # the meshes dict (render_seq just scene.add()s them).
        from aitviewer.renderables.point_clouds import PointClouds
        pc = PointClouds(points, color=(0.62, 0.62, 0.68, 1.0), name="slam_points")
        try:
            pc.point_size = 2.0
        except Exception:
            pass
        meshes["slam_points"] = pc

    vis_h, vis_w = (height, width)
    K = np.array(
        [
            [1000, 0, vis_w / 2],
            [0, 1000, vis_h / 2],
            [0, 0, 1]
        ]
    )
    
    data = viewer_utils.ViewerData(viewer_Rt, K, vis_w, vis_h)
    batch = (meshes, data)

    if interactive:
        viewer = viewer_utils.ARCTICViewer(interactive=True, size=(vis_w, vis_h))
    else:
        viewer = viewer_utils.ARCTICViewer(interactive=False, size=(vis_w, vis_h), render_types=['video'])
        if os.path.exists(os.path.join(output_pth, 'aitviewer', "video_0.mp4")):
            os.remove(os.path.join(output_pth, 'aitviewer', "video_0.mp4"))

    viewer.render_seq(batch, out_folder=os.path.join(output_pth, 'aitviewer'))
    if not interactive:
        return os.path.join(output_pth, 'aitviewer', "video_0.mp4")

def run_vis2_on_video_cam(res_dict, res_dict2, output_pth, focal_length, image_names, R_w2c=None, t_w2c=None):

    height, width = _dims_from_images(image_names)

    world_mano = {}
    world_mano['vertices'] = res_dict['vertices']
    world_mano['faces'] = res_dict['faces']

    world_mano2 = {}
    world_mano2['vertices'] = res_dict2['vertices']
    world_mano2['faces'] = res_dict2['faces']

    vis_dict = {}
    color_idx = 0
    world_mano['vertices'] = world_mano['vertices']
    for _id, _verts in enumerate(world_mano['vertices']):
        verts = _verts.cpu().numpy() # T, N, 3
        body_faces = world_mano['faces']
        body_meshes = {
            "v3d": verts,
            "f3d": body_faces,
            "vc": None,
            "name": f"hand_{_id}",
            # "color": "pace-green",
            "color": "director-purple",
        }
        vis_dict[f"hand_{_id}"] = body_meshes
        color_idx += 1
    
    world_mano2['vertices'] = world_mano2['vertices']
    for _id, _verts in enumerate(world_mano2['vertices']):
        verts = _verts.cpu().numpy() # T, N, 3
        body_faces = world_mano2['faces']
        body_meshes = {
            "v3d": verts,
            "f3d": body_faces,
            "vc": None,
            "name": f"hand2_{_id}",
            # "color": "pace-blue",
            "color": "director-blue",
        }
        vis_dict[f"hand2_{_id}"] = body_meshes
        color_idx += 1

    meshes = viewer_utils.construct_viewer_meshes(
        vis_dict, draw_edges=False, flat_shading=False
    )

    num_frames = len(world_mano['vertices'][_id])
    Rt = np.zeros((num_frames, 3, 4))
    Rt[:, :3, :3] = R_w2c[:num_frames]
    Rt[:, :3, 3] = t_w2c[:num_frames]

    cols, rows = (width, height)
    K = np.array(
        [
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ]
    )
    vis_h = height
    vis_w = width

    # The billboard needs image files; decode the vis-range frames to a temp dir.
    image_paths, tmp_dir = _materialize_frames(image_names, output_pth)
    data = viewer_utils.ViewerData(Rt, K, cols, rows, imgnames=image_paths)
    batch = (meshes, data)

    viewer = viewer_utils.ARCTICViewer(interactive=True, size=(vis_w, vis_h))
    try:
        viewer.render_seq(batch, out_folder=os.path.join(output_pth, 'aitviewer'))
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

def lookat_matrix(source_pos, target_pos, up):
    """
    IMPORTANT: USES RIGHT UP BACK XYZ CONVENTION
    :param source_pos (*, 3)
    :param target_pos (*, 3)
    :param up (3,)
    """
    *dims, _ = source_pos.shape
    up = up.reshape(*(1,) * len(dims), 3)
    up = up / torch.linalg.norm(up, dim=-1, keepdim=True)
    back = normalize(target_pos - source_pos)
    right = normalize(torch.linalg.cross(up, back))
    up = normalize(torch.linalg.cross(back, right))
    R = torch.stack([right, up, back], dim=-1)
    return make_4x4_pose(R, source_pos)

def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)

def normalize(x):
    return x / torch.linalg.norm(x, dim=-1, keepdim=True)

def save_mesh_to_obj(vertices, faces, file_path):
    # 创建一个 Trimesh 对象
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # 导出为 .obj 文件
    mesh.export(file_path)
    print(f"Mesh saved to {file_path}")

