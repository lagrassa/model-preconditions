from pathlib import Path
import numpy as np
import copy
import logging
from collections import OrderedDict

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from autolab_core import YamlConfig

from .base_env import BaseEnv
from .utils import is_pose_of_object_close, get_pose_pillar_state, set_pose, get_joint_position_pillar_state, \
    get_joint_velocity_pillar_state, get_gripper_width_pillar_state, is_state_of_robot_close, make_env_with_init_states
from ..utils import yaw_from_quat, yaw_from_np_quat, get_rod_rel_goal_RigidTransforms_x_in, pillar_state_to_shapes, \
    object_name_to_asset_name, pillar_state_obj_to_transform, xyz_yaw_to_transform, r_flip_yz, \
    get_object_names_in_pillar_state, set_franka_pillar_properties_from_init_states_arr, is_obj_in_gripper, \
    extract_xy_dims_and_height, params_cause_collision_franka, transform_to_xy_yaw, \
    set_fingers_and_visualize_pillar_state, place_grippers_in_pillar_state


class WaterEnv(BaseEnv):
    def __init__(self, cfg, setup_callbacks=[], for_mde_training=False, baseboard = True):
        super().__init__(cfg, for_mde_training=for_mde_training, is_ig_env=False)
        softgym_env_name = "PassWater"
        env_kwargs = env_arg_dict[softgym_env_name]

        # Generate and save the initial states for running this environment for the first time
        env_kwargs['use_cached_states'] = False
        env_kwargs['save_cached_states'] = False
        env_kwargs['num_variations'] = 1
        env_kwargs['render'] = True
        env_kwargs['headless'] = True

        if not env_kwargs['use_cached_states']:
            print('Waiting to generate environment variations. May take 1 minute for each variation...')
        self._scene = normalize(SOFTGYM_ENVS[softgym_env_name](**env_kwargs))

        self._scene.reset()
        self.save_action(np.array([0]))
        self.step()

    @property
    def n_envs(self):
        return self._n_envs

    @staticmethod
    def pillar_state_to_sem_state(pillar_state, sem_state_obj_names, anchor_obj_name=None, ref_pillar_state=None):
        return None

    @staticmethod
    def sem_state_to_pillar_state(sem_state, ref_pillar_state, sem_state_obj_names, anchor_obj_name=None):
        return None

    def get_sem_state(self, should_reset_to_viewable=False):
        return self._saved_data[0][0]


    def save_action(self, action):
        self._saved_action = action

    def step(self):
        self._saved_data = self._scene.step(self._saved_action, record_continuous_video=True, img_size=720)


    def generate_init_states(self, cfg, min_samples=10, max_samples=1000, init_state_flag=None,
                             choose_from_multiple_rod_configs=False, return_init_state_info=False):
        ''' Generator for initial env states.

        choose_from_multiple_rod_configs:  If True use `rod_configs` Flag from the env config to select
            range
        return_init_state_info:  Use only if `choose_from_multiple_rod_configs: True`. If True, returns info
            dict regarding which config was selected.
        '''
        return None

    def _make_collision_shapes(pillar_state):
        return []
    def _update_pillar_state(self, env_idx):
        pass
    def set_state(self):
        pass



if __name__ == "__main__":
    cfg = YamlConfig("cfg/envs/water_env.yaml")
    env = WaterEnv(cfg)
    env.save_action(np.array([0.01]))
    env.step()
    state = env.get_sem_state(should_reset_to_viewable=False)
