from pathlib import Path
import numpy as np
import copy
import os
import logging
from collections import OrderedDict

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif
from autolab_core import YamlConfig

from .base_env import BaseEnv
from .water_transport_env import WaterEnv2D
from .utils import is_pose_of_object_close, get_pose_pillar_state, set_pose, get_joint_position_pillar_state, \
    get_joint_velocity_pillar_state, get_gripper_width_pillar_state, is_state_of_robot_close, make_env_with_init_states
from ..utils import yaw_from_quat, yaw_from_np_quat, get_rod_rel_goal_RigidTransforms_x_in, pillar_state_to_shapes, \
    object_name_to_asset_name, pillar_state_obj_to_transform, xyz_yaw_to_transform, r_flip_yz, \
    get_object_names_in_pillar_state, set_franka_pillar_properties_from_init_states_arr, is_obj_in_gripper, \
    extract_xy_dims_and_height, params_cause_collision_franka, transform_to_xy_yaw, \
    set_fingers_and_visualize_pillar_state, place_grippers_in_pillar_state


class WaterEnv3D(WaterEnv2D):
    @property
    def n_envs(self):
        return self._n_envs

    @property
    def env_idxs(self):
        return self._env_idxs

    def get_sem_state(self, should_reset_to_viewable=False):
        assert not isinstance(self._saved_data[0][0][0], np.ndarray)
        state_vector = self._saved_data[0][0]
        #dis_x is length, dis_w is height
        #[x, y, rot, dis_x, dis_z, height, distance+x, poured_height, poured_dis_x, poured_dis_z,current_water_height, in_poured, in_control ]
        return state_vector

    def reset(self, n_steps=2):
        self._scene.reset()
        null_action = np.array([0, 0, 0])
        self.save_action(null_action)
        self.frames = []
        for i in range(n_steps):
            self.step()



if __name__ == "__main__":
    cfg = YamlConfig("cfg/envs/pour_env.yaml")
    env = WaterEnv(cfg)
    env.save_action(np.array([0.0,0.05 , 0]))
    for i in range(50):
        env.step()
        print(env.get_sem_state().round(2))
    env.save_video()
