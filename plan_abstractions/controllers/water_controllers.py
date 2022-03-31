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
        self._kp = 3 #2 was safe
        self._end_buffer = 10

    def _plan(self, curr_pos, goal_pos, total_horizon=None):
        self._traj_pos = np.linspace(curr_pos, goal_pos, total_horizon)[1:]
        self.goal_pos =  goal_pos
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
        self._kp_theta = 0.3
        self._end_buffer = 3

    def _plan(self, curr_pos, curr_angle, goal_angle, max_volume, total_horizon=None):
        self._start_pos = curr_pos
        self.start_angle = curr_angle
        third_traj = np.linspace(curr_angle, goal_angle, int(total_horizon/3))[1:]
        self._traj =  np.hstack([third_traj, np.ones(int(total_horizon/3))*goal_angle, third_traj[::-1]])
        self.goal_angle =  goal_angle
        self._max_volume = max_volume
        self._reverse = False
        return {
            'T_plan': 0
        }

    @property
    def horizon(self):
        return len(self._traj) + self._end_buffer

    def _call(self, curr_state, t, delta=False):
        """
        if not self._reverse and curr_state[-2] > self._max_volume:
            self._reverse = True
            self._time_reversed = t
        if self._reverse:
            t_local = min(t-self._time_reversed, len(self._traj)-1)
            desired_theta = self._traj[-(t_local+1)]
        """
        t = min(t, len(self._traj)-1)
        desired_theta = self._traj[t]
        desired_pos = self._start_pos
        error = desired_pos - curr_state[0:2]
        d_pos = self._kp_pos*(error)
        d_theta = self._kp_theta*(desired_theta-curr_state[2])
        action = np.hstack([d_pos, [d_theta]])
        return action
