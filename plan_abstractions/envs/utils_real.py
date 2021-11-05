import numpy as np
import matplotlib.pyplot as plt

from autolab_core import RigidTransform, Point
from visualization.visualizer3d import Visualizer3D as vis3d

from skimage.feature import match_template
from skimage.transform import rotate
from skimage.measure import label, regionprops


def imshow(im):
    plt.figure()
    plt.imshow(im.data)
    plt.show()


def get_crop_args_from_dims(crop_horiz, crop_vert):
    crop_height = crop_vert[1] - crop_vert[0]
    crop_width = crop_horiz[1] - crop_horiz[0]

    crop_center_i = crop_height // 2 + crop_vert[0]
    crop_center_j = crop_width // 2 + crop_horiz[0]
    return crop_height, crop_width, crop_center_i, crop_center_j


def get_block_2d_poses_seg_masks(depth, depth_block_th, block_size_px, match_score_th, vis=False):
    depth_mask = depth < depth_block_th
    
    if vis:
        imshow(depth_mask)

    block_border_size_px = int(block_size_px * np.sqrt(2)) + 2
    block_outline_template = np.zeros((block_border_size_px, block_border_size_px))
    delta = (block_border_size_px - block_size_px) // 2
    block_outline_template[delta:-delta, delta:-delta] = 1

    match_scores_by_deg = {}
    match_scores_filtered_union = np.zeros_like(depth_mask)

    for deg in range(-45, 45, 5):
        block_outline_template_rot = rotate(block_outline_template, deg)
        match_scores = match_template(depth_mask, block_outline_template_rot, pad_input=True, mode='minimum')
        match_scores_filtered = match_scores > match_score_th
        
        match_scores_by_deg[deg] = match_scores
        match_scores_filtered_union = np.logical_or(match_scores_filtered_union, match_scores_filtered)

    segs = label(match_scores_filtered_union, background=0)
    seg_props = regionprops(segs)
    block_2d_poses = []
    for seg_idx, seg_prop in enumerate(seg_props):
        seg_mask = segs == seg_idx + 1
        
        best_deg = None
        max_match_score = -np.inf
        for deg, match_scores in match_scores_by_deg.items():
            match_score = match_scores[seg_mask].max()
            if match_score > max_match_score:
                max_match_score = match_score
                best_deg = deg
            
        center_i, center_j = seg_prop.centroid
        block_2d_poses.append(np.array([center_i, center_j, np.deg2rad(best_deg)]))

    return block_2d_poses, segs


def plot_2d_detections(color, block_size_px, block_2d_poses):
    plt.figure()
    ax = plt.gca()

    plt.imshow(color)

    corners_local = np.array([
        [-1, -1],
        [-1, 1],
        [1, 1],
        [1, -1],
        [-1, -1]
    ]) * block_size_px / 2
    for block_2d_pose in block_2d_poses:
        block_pos = block_2d_pose[1], block_2d_pose[0]
        plt.scatter(*block_pos)
        
        rot = -block_2d_pose[2]
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([
            [c, -s],
            [s, c]
        ])
        corners_world = block_pos + corners_local @ R.T
        for i in range(len(corners_world) - 1):
            plt.plot(
                [corners_world[i][0], corners_world[i + 1][0]], 
                [corners_world[i][1], corners_world[i + 1][1]], 
                color='yellow'
            )

    ax.set_aspect('equal')
    plt.show()


def block_2d_to_3d_pose(block_2d_pose, depth_im, intr, T_camera_world, z_offset, color_idx, n_idx):
    px_pt = Point(block_2d_pose[:2][::-1], intr.frame)
    block_pt_camera = intr.deproject_pixel(depth_im.data[int(px_pt.y), int(px_pt.x)], px_pt)
    block_pt_world = T_camera_world * block_pt_camera
    
    T_block = RigidTransform(
        translation=block_pt_world.vector + np.array([0, 0, z_offset]),
        rotation=RigidTransform.z_axis_rotation(block_2d_pose[2]),
        from_frame=f'block_c{color_idx}_n{n_idx}', to_frame=T_camera_world.to_frame
    )

    return T_block


def get_block_color_idxs(color, seg, color_vals):
    color_patch = color[seg]
    sum_color_patch = np.sum(color_patch)
    scores = [
        np.mean(color_patch @ color_val) / sum_color_patch / np.mean(color_val)
        for color_val in color_vals
    ]
    return np.argmax(scores)      


def detect_blocks(color_table, depth_table, color_bin, depth_bin, 
        T_camera_table_world, T_camera_bin_world,
        intr, crop_cfg, block_depth_cfg, block_size_cfg, 
        block_height, block_colors, match_score_th, vis=False,
        debug_vis_data=None):

    crop_args_table = get_crop_args_from_dims(crop_cfg['table']['horiz'], crop_cfg['table']['vert'])
    crop_args_bin = get_crop_args_from_dims(crop_cfg['bin']['horiz'], crop_cfg['bin']['vert'])

    color_table_crop = color_table.crop(*crop_args_table)
    color_bin_crop = color_bin.crop(*crop_args_bin)

    depth_table_crop = depth_table.crop(*crop_args_table)
    depth_bin_crop = depth_bin.crop(*crop_args_bin)

    intr_table_crop = intr.crop(*crop_args_table)
    intr_bin_crop = intr.crop(*crop_args_bin)

    pc_table = T_camera_table_world * intr_table_crop.deproject(depth_table_crop)
    pc_bin = T_camera_bin_world * intr_bin_crop.deproject(depth_bin_crop)

    bin_block_2d_poses, bin_segs = get_block_2d_poses_seg_masks(
        depth_bin_crop.data, block_depth_cfg['bin'], 
        block_size_cfg['bin'], match_score_th, vis=vis
    )
    table_block_2d_poses, table_segs = get_block_2d_poses_seg_masks(
        depth_table_crop.data,  block_depth_cfg['table'], 
        block_size_cfg['table'], match_score_th, vis=vis
    )

    if vis:
        plot_2d_detections(color_table_crop.data, block_size_cfg['table'], table_block_2d_poses)
        plot_2d_detections(color_bin_crop.data, block_size_cfg['bin'], bin_block_2d_poses)

    block_color_idxs = [0] * len(block_colors)
    block_3d_poses = []
    for idx, block_2d_pose in enumerate(table_block_2d_poses):
        color_idx = get_block_color_idxs(
            color_table_crop.data / 255, table_segs == idx + 1, block_colors
        )
        n_idx = block_color_idxs[color_idx]
        block_3d_pose = block_2d_to_3d_pose(
            block_2d_pose, depth_table_crop, intr_table_crop, T_camera_table_world, -block_height / 2, color_idx, n_idx
        )
        block_3d_poses.append(block_3d_pose)
        block_color_idxs[color_idx] += 1
    for idx, block_2d_pose in enumerate(bin_block_2d_poses):
        color_idx = get_block_color_idxs(
            color_bin_crop.data / 255, bin_segs == idx + 1, block_colors
        )
        n_idx = block_color_idxs[color_idx]
        block_3d_pose = block_2d_to_3d_pose(
            block_2d_pose, depth_bin_crop, intr_bin_crop, T_camera_bin_world, -block_height / 2, color_idx, n_idx
        )
        block_3d_poses.append(block_3d_pose)
        block_color_idxs[color_idx] += 1

    if vis:
        vis3d.figure()
        vis3d.pose(RigidTransform())
        vis3d.pose(T_camera_table_world)
        vis3d.pose(T_camera_bin_world)
        vis3d.points(pc_table.subsample(10)[0].data.T, scale=2e-3)
        vis3d.points(pc_bin.subsample(10)[0].data.T, scale=2e-3)

        for block_3d_pose in block_3d_poses:
            vis3d.pose(block_3d_pose)

        vis3d.show()

    if debug_vis_data is not None:
        debug_vis_data.update({
            'T_camera_table_world': T_camera_table_world,
            'T_camera_bin_world': T_camera_bin_world,
            'pc_table': pc_table,
            'pc_bin': pc_bin,
            'block_3d_poses': block_3d_poses,
        })

    return block_3d_poses

