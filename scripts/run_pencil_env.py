import argparse

import numpy as np
from autolab_core import YamlConfig
from pyquaternion import Quaternion

from isaacgym_utils.draw import draw_contacts, draw_transforms
from isaacgym_utils.math_utils import set_seed

from plan_abstractions.envs import make_pusher_rod_test_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg/push_rods_toy.yaml')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)
    set_seed(cfg['seed'])
    cfg['scene']['gui'] = 1 #otherwise why are you running this
    env = make_pusher_rod_test_env(cfg)

    def custom_draws(scene):
        for env_idx in scene.env_idxs:
            pusher_transform = env.get_pusher_transform(env_idx)
            rod_transforms = env.get_rod_transforms(env_idx)
            transforms = rod_transforms + [pusher_transform]
            draw_transforms(scene, [env_idx], transforms)
        draw_contacts(scene, scene.env_idxs)

    while True:
        actions = (np.random.random((env._scene.n_envs, 3)) * 2 - 1) * 1e-3
        actions = np.zeros_like(actions)
        #actions[:, 2] = -0.001
        for i in range(100):
            ft_arr = [0.1,0.005,0]
            env.apply_force_torque_to_pusher(ft_arr, 0)
            env.step()
            quat = env.get_pusher_transform(0).r
            pyquat = Quaternion(w=quat.w, x=quat.x, y = quat.y, z = quat.z)
        # print(quat_to_rpy(env.get_pusher_transform(1).r).round(2))
