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
        self._kp = 2 #0.7 was safe
        self._end_buffer = 10

    def _plan(self, curr_pose, goal_pose, total_horizon=None):
        self._traj_pos = np.linspace(curr_pose, goal_pose, total_horizon)[1:]
        self.goal_pose =  goal_pose
        return {
            'T_plan': 0
        }

    @property
    def horizon(self):
        return len(self._traj_pos) + self._end_buffer

    def _call(self, curr_state, t, delta=False):
        t = min(t, len(self._traj_pos)-1)
        desired_state = self._traj_pos[t]
        error = desired_state - curr_state[0:2]
        d_pos = self._kp*(error)
        d_theta = 0
        action = np.hstack([d_pos, [d_theta]])
        return action

class PourController(WaterTransportController):

    def __init__(self):
        super().__init__()
        self._kp_pos = 0.2 #0.7 was safe
        self._kp_theta = 1
        self._end_buffer = 3

    def _plan(self, curr_pos, curr_angle, goal_angle, total_horizon=None):
        self._start_pos = curr_pose
        self._traj = np.linspace(curr_angle, goal_angle, total_horizon)[1:]
        self.goal_angle =  goal_angle
        return {
            'T_plan': 0
        }

    @property
    def horizon(self):
        return len(self._traj) + self._end_buffer

    def _call(self, curr_state, t, delta=False):
        t = min(t, len(self._traj)-1)
        desired_theta = self._traj[t]
        desired_pos = self._start_pos 
        error = desired_pos - curr_state[0:2]
        d_pos = self._kp_pos*(error)
        d_theta = self._kp_theta(desired_theta-curr_state[2])
        action = np.hstack([d_pos, [d_theta]])
        return action
