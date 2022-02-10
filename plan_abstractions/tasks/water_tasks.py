import numpy as np
import copy
from itertools import combinations
import logging

import quaternion

from isaacgym_utils.math_utils import np_to_quat, transform_to_RigidTransform, np_to_transform
from .base_task import BaseTask
from ..skills import WaterTransport1D

from ..utils.utils import pretty_print_state_with_params, pretty_print_array, min_distance_between_angles
try:
    from ..utils.ar_perception import ObjectDetector
except ImportError:
    print("Not able to import ObjectDetector")
from autolab_core import YamlConfig

logger = logging.getLogger(__name__)

class MoveWater(BaseTask):
    def __init__(self, cfg, real_robot=False):
        super().__init__(cfg)
        self._water_out_tol = cfg["goal"]["water_out_tol"]
        self._position_tol = cfg["goal"]["position_tol"]
        self._same_tol = 0.01
        self._goal_pose = cfg["goal"]["pose"]
        self._setup_callbacks = [] # already in env self.add_real_drawer_to_env_cb]


    def states_similar(self, vector_state_1, vector_state_2):
        if np.linalg.norm(np.array(vector_state_1)-np.array(vector_state_2)) > self._same_tol: #want this to be more tolerant:
            return False

        return True


    def evaluate(self, pillar_state):
        return self.distance_to_goal_state(pillar_state)

    def is_goal_state(self, vector_state):
        if self.distance_to_goal_state(vector_state) < self._position_tol:
            if vector_state[-1] > self._water_out_tol:
                return True
        return False

    def distance_to_goal_state(self, vector_state):
        return np.linalg.norm(vector_state[0]-self._goal_pose)
