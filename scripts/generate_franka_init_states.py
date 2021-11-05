import os
from pathlib import Path
import argparse

import numpy as np
import quaternion
from autolab_core import YamlConfig
from tqdm import trange

from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymFranka
from isaacgym_utils.policy import EEImpedanceWaypointPolicy
from isaacgym_utils.draw import draw_transforms
from isaacgym_utils.math_utils import np_quat_to_quat, np_to_vec3, transform_to_np, quat_to_np

from plan_abstractions.utils import ee_yaw_to_np_quat, angle_axis_between_quats


def policy_gen(policies):
    def policy(scene, env_idx, t_step, t_sim):
        return policies[env_idx](scene, env_idx, t_step, t_sim)
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg/generate_franka_init_states.yaml')
    parser.add_argument('--output', '-o', type=str, default='data/franka_init_states.npz')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

    if cfg['scene']['gui'] == 0:
        cfg['scene']['es'] = 0

    output_dir = Path(args.output).parents[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scene = GymScene(cfg['scene'])
    franka = GymFranka(cfg['franka'], scene, actuation_mode='torques')
    franka_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, 0.01))
    franka_name = 'franka'

    def setup(scene, env_idx):
        scene.add_asset(franka_name, franka, franka_transform, collision_filter=2) # avoid self-collision
    scene.setup_all_envs(setup)

    draw_data = {'goal_transforms': None}
    def custom_draws(scene):
        for env_idx in scene.env_idxs:
            transforms = [
                franka_transform, 
                franka.get_ee_transform(env_idx, franka_name), 
                franka.get_links_transforms(env_idx, franka_name)[3]
            ]
            if draw_data['goal_transforms'] is not None:
                transforms.append(draw_data['goal_transforms'][env_idx])
            draw_transforms(scene, [env_idx], transforms, length=0.2)

    data = {
        'ee_poses': [],
        'joints': [],
        'finger_left_poses': [],
        'finger_right_poses': [],
        'gripper_widths': []
    }
    latest_n_iter = -1
    quat_z_down = quaternion.from_rotation_matrix(np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ]))
    max_vert_angle = np.deg2rad(cfg['max_vert_angle'])
    def cb(scene, t_step, t_sim):
        if latest_n_iter > 0 and t_step % cfg['save_every_step'] == 0:
            for env_idx in scene.env_idxs:
                ee_transform = franka.get_ee_transform(env_idx, franka_name)
                ee_quat = quaternion.from_float_array(quat_to_np(ee_transform.r, format='wxyz'))
                vert_angle = np.linalg.norm(angle_axis_between_quats(ee_quat, quat_z_down))
                # only save data where ee is sufficiently vertical
                if True or vert_angle < max_vert_angle:
                    data['ee_poses'].append(transform_to_np(ee_transform, format='wxyz'))
                    data['joints'].append(franka.get_joints(env_idx, franka_name)[:7])

                    finger_poses = [
                        transform_to_np(transform, format='wxyz')
                        for transform in franka.get_finger_transforms(env_idx, franka_name)
                    ]
                    data['finger_left_poses'].append(finger_poses[0])
                    data['finger_right_poses'].append(finger_poses[1])

                    data['gripper_widths'].append(franka.get_gripper_width(env_idx, franka_name))

    # ignore first iter b/c it starts w/ franka in the default init pose, which is very high
    for n_iter in trange(cfg['n_iters'] + 1):
        latest_n_iter = n_iter
        T = 300 if n_iter == 0 else 200
        policies = []
        goal_transforms = []
        for env_idx in scene.env_idxs:
            init_ee_transform = franka.get_ee_transform(env_idx, franka_name)

            goal_pos = np.array([
                np.random.uniform(cfg['ee_goal_range']['x'][0], cfg['ee_goal_range']['x'][1]),
                np.random.uniform(cfg['ee_goal_range']['y'][0], cfg['ee_goal_range']['y'][1]),
                np.random.choice(cfg['ee_goal_range']['z'], 1)
            ])
            goal_yaw = np.random.uniform(
                np.deg2rad(cfg['ee_goal_range']['yaw']['lo']),
                np.deg2rad(cfg['ee_goal_range']['yaw']['hi'])
            )
            goal_ee_transform = gymapi.Transform(
                p=np_to_vec3(goal_pos),
                r=np_quat_to_quat(ee_yaw_to_np_quat(goal_yaw))
            )
            goal_transforms.append(goal_ee_transform)

            policies.append(EEImpedanceWaypointPolicy(
                franka_name, init_ee_transform, goal_ee_transform, T=T
            ))
        policy = policy_gen(policies)

        draw_data['goal_transforms'] = goal_transforms
        scene.run(policy=policy, time_horizon=T, custom_draws=custom_draws, cb=cb)

    for k, v in data.items():
        data[k] = np.array(v)
    np.savez(args.output, **data)
