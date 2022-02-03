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
        self._kp = 1

    def _plan(self, curr_x, goal_x, total_horizon=None):
        self._traj_pos = np.linspace(curr_x, goal_x, total_horizon)
        return {
            'T_plan': 0
        }

    @property
    def horizon(self):
        return len(self._traj_pos)

    def _call(self, curr_state, t, delta=False):
        desired_state = self._traj_pos[t]
        dx = self._kp*(desired_state - curr_state[0])
        action = np.array([dx])
        return action

