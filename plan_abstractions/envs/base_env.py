from abc import ABC, abstractmethod
from copy import deepcopy

from pillar_state import State
from isaacgym_utils.scene import GymScene
import numpy as np

from ..utils import shapes_in_collision, transform_to_xy_yaw, pillar_state_obj_to_transform
from isaacgym_utils.math_utils import transform_to_np
import logging
logger = logging.getLogger(__name__)
logger.setLevel('INFO')


class BaseEnv(ABC):

    def __init__(self, cfg, for_mde_training=False, is_ig_env=True): #:c bad hack lagrassa is adding to indicate this env is for MDE training and should be treated differently
        self._cfg = cfg
        self._for_mde_training = for_mde_training
        if is_ig_env:
            self._scene = GymScene(cfg['scene'])
            self._pillar_states = [State() for _ in range(self._scene.n_envs)]
        self._real_robot = False

    @classmethod
    def is_in_collision(cls, pillar_state, body_names=None, plot=False, asset_name_to_eps_arr=None):
        shapes = cls._make_collision_shapes(pillar_state, body_names=body_names, asset_name_to_eps_arr=asset_name_to_eps_arr, plot=plot)
        return shapes_in_collision(shapes, body_names)

    @staticmethod
    @abstractmethod
    def _make_collision_shapes(pillar_state):
        """
        Returns: A list of env specific shapely collision shapes
        positioned according to pillar_state
        """
        pass

    @staticmethod
    def pillar_state_to_sem_state(pillar_state, sem_state_obj_names, anchor_obj_name=None, ref_pillar_state=None):
        ''' Converts a pillar_state to an sem_state as np array

        If anchor_obj_name is not None, then poses in pillar_state will be expressed
        in the frame of anchor obj in the ref_pillar_state
        '''
        raise NotImplementedError()

    @staticmethod
    def sem_state_to_pillar_state(sem_state, ref_pillar_state, sem_state_obj_names):
        ''' Convert sem_state (np array) to a pillar state by making a copy of and
            updating relevant parts of ref_pillar_state
        '''
        raise NotImplementedError()

    @property
    def n_envs(self):
        return self._scene.n_envs

    @property
    def asset_name_to_eps_arr(self):
        return deepcopy(self._asset_name_to_eps_arr)

    @abstractmethod
    def _update_pillar_state(self, env_idx):
        pass

    @abstractmethod
    def set_state(self, pillar_state, env_idx, n_steps=0):
        """
        Sets state of simulator to state denoted by pillar state
        """
        pass

    def _custom_draws(self, scene):
        pass

    def set_all_states(self, states, env_idxs=None, n_steps=0):
        if env_idxs is None:
            env_idxs = self._scene.env_idxs
        assert len(states) == len(env_idxs)

        for env_idx, state in enumerate(states):
            self.set_state(state, env_idx, n_steps=0)
        desired_pos = transform_to_np(self._franka.get_desired_ee_transform(0, "franka"))[:3]
        actual_pos = transform_to_np(self._franka.get_ee_transform(0, "franka"))[:3]
        logger.debug(f"Distance before set state : {np.linalg.norm(actual_pos - desired_pos)}")

        for _ in range(n_steps):
            self._scene.step()
            self._scene.render(custom_draws=self._custom_draws)
        actual_pos = transform_to_np(self._franka.get_ee_transform(0, "franka"))[:3]
        logger.debug(f"Distance after set state : {np.linalg.norm(actual_pos - desired_pos)}")


        if n_steps > 0:
            for env_idx in env_idxs:
                self._update_pillar_state(env_idx)

    def get_state(self, env_idx):
        return State.create_from_serialized_string(self._pillar_states[env_idx].get_serialized_string())

    def get_all_states(self, env_idxs=None):
        if env_idxs is None:
            env_idxs = self._scene.env_idxs
        return [self.get_state(env_idx) for env_idx in env_idxs]

    def _compute_costs(self):
        return np.zeros(self.n_envs)

    def step(self):
        self._scene.step()
        self._scene.render(custom_draws=self._custom_draws)
        for env_idx in range(self.n_envs):
            self._update_pillar_state(env_idx)
        return self._compute_costs()

    def planner_state_to_viz_string(self, planner_state):
        pillar_state = planner_state.pillar_state
        action_in = planner_state.action_in
        body_names = self.get_object_names()
        poses = [transform_to_xy_yaw(pillar_state_obj_to_transform(pillar_state, body_name))
                 for body_name in body_names]
        state_str = ''
        for body_name, pose in zip(body_names, poses):
            state_str += f"{body_name}: {np.around(pose, 2)}\n"
        state_str += f"h: {np.around(planner_state.h, 3)}"
        if action_in:
            action_str = f"{action_in.skill_idx}, {np.around(action_in.params, 2)}"
        else:
            action_str = ''
        return state_str, action_str

