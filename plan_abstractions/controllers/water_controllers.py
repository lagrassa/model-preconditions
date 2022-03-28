from time import time
import numpy as np
from scipy.linalg import block_diag
from pyquaternion import Quaternion

from isaacgym_utils.math_utils import min_jerk, quat_to_np
import quaternion

from ..utils import ee_yaw_to_np_quat, angle_axis_between_quats
from .base_controller import BaseController


class WaterTransportController(BaseController):

    def __init__(self):
        super().__init__()
        self._kp = 1.2
        self._end_buffer = 5

    def _plan(self, curr_x, goal_x, total_horizon=None):
        self._traj_pos = np.linspace(curr_x, goal_x, total_horizon)[1:]
        self.goal_x =  goal_x
        return {
            'T_plan': 0
        }

    @property
    def horizon(self):
        return len(self._traj_pos) + self._end_buffer

    def _call(self, curr_state, t, delta=False):
        t = min(t, len(self._traj_pos)-1)
        desired_state = self._traj_pos[t]
        error = desired_state - curr_state[0]
        dx = self._kp*(error)
        action = np.array([dx])
        return action

