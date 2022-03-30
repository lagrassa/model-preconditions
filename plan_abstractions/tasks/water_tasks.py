import numpy as np
import copy
from itertools import combinations
import logging

import quaternion

from isaacgym_utils.math_utils import np_to_quat, transform_to_RigidTransform, np_to_transform
from .base_task import BaseTask
from ..skills import WaterTransport2D

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
        self._same_tol = 0.03
        self._goal_pose = cfg["goal"]["pose"]
        self._goal_pose_ranges = cfg["goal"]["goal_pose_ranges"]
        self._setup_callbacks = [] # already in env self.add_real_drawer_to_env_cb]


    def states_similar(self, vector_state_1, vector_state_2):
        if np.linalg.norm(np.array(vector_state_1)-np.array(vector_state_2)) > self._same_tol: #want this to be more tolerant:
            return False
        return True


    def evaluate(self, pillar_state):
        return self.distance_to_goal_state(pillar_state)

    def is_goal_state(self, vector_state):
        if self.distance_to_goal_state(vector_state) < self._position_tol:
            if (1-vector_state[-1]) < self._water_out_tol:
                return True
        return False

    def distance_to_goal_state(self, vector_state):
        #return np.linalg.norm(vector_state[0]-self._goal_pose)
        dist = np.linalg.norm(vector_state[0:2]-self._goal_pose)
        return dist

    def resample_goal(self, env=None):
        old_goal = np.copy(self._goal_pose)
        new_pose = np.random.uniform(size=(2,), low=self._goal_pose_ranges["low"], high=self._goal_pose_ranges["high"])
        self._goal_pose = new_pose
        return old_goal, new_pose

class WaterInBox(BaseTask):
    def __init__(self, cfg, real_robot=False):
        super().__init__(cfg)
        self._water_out_tol = cfg["goal"]["water_out_tol"]
        self._volume_tol = cfg["goal"]["volume_tol"]
        self._same_tol = 0.03
        self._setup_callbacks = [] # already in env self.add_real_drawer_to_env_cb]
        self._goal_volume = cfg["goal"]["volume"]
        self._goal_volume_ranges = cfg["goal"]["volume_ranges"]


    def states_similar(self, vector_state_1, vector_state_2):
        if np.linalg.norm(np.array(vector_state_1)-np.array(vector_state_2)) > self._same_tol: 
            return False
        return True


    def evaluate(self, pillar_state):
        return self.distance_to_goal_state(pillar_state)

    def is_goal_state(self, vector_state):
        if abs(vector_state[-2]-self._goal_volume) < self._volume_tol:
            water_in_total = vector_state[-1] + vector_state[-2]
            if (1-water_in_total) < self._water_out_tol:
                return True
        return False

    def distance_to_goal_state(self, state):
        vector_state = state
        height_control_cup = state[1]
        height_target = state[7]
        glass_x = state[0]
        pos_control = state[0] #x coordinate of box, starts at 0
        glass_distance = state[6]-state[0]
        pos_target = glass_distance-glass_x  + 0.15
        dx = abs(pos_target-vector_state[0]) #distance from the edge
        #dz = abs(vector_state[1] - vector_state[7])
        dz = 0 #min(abs(vector_state[1] - vector_state[7]), 2*vector_state[7])
        #dist = abs(vector_state[-2]-self._goal_volume)
        #print("dx", dx)
        return dx + dz

    def resample_goal(self, env=None):
        old_goal = np.copy(self._goal_volume)
        new_volume = np.random.uniform(size=(1,), low=self._goal_volume_ranges["low"], high=self._goal_volume_ranges["high"])
        self._goal_volume = new_volume
        return old_goal, new_volume

