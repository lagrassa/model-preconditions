from time import time
import numpy as np
from scipy.linalg import block_diag
from pyquaternion import Quaternion

from isaacgym_utils.math_utils import min_jerk, quat_to_np
import quaternion

from ..utils import ee_yaw_to_np_quat, angle_axis_between_quats
from .base_controller import BaseController


class PositionWaypointController(BaseController):
    ''' Plans a trajectory w/ 4 waypoints to "teleport" an end-effector 
        by first moving to a separate z plane, then going to a point 
        above the goal, then going to the goal.
    '''

    def __init__(self, dt, Kp_pos=500, Kp_angle=5, avoid_obstacle_height=True):
        super().__init__()
        self._Ks = np.diag([Kp_pos] * 3 + [5, 5, Kp_angle])
        self._Ds = np.sqrt(self._Ks) * [10, 10, 10, 0.5, 0.5, 0.5]
        #self._total_horizon = np.floor(20 / dt).astype(int)
        self._total_horizon = np.floor(4 / dt).astype(int)
        self._avoid_obstacle_height = avoid_obstacle_height

    def _plan(self, curr_pose, goal_pose, z, total_horizon=None):
        if total_horizon is not None:
            self._total_horizon = total_horizon
        s = time()

        wp0 = curr_pose

        wp1 = wp0.copy()
        wp1[2] = z

        wp2 = wp1.copy()
        goal_pos = goal_pose[:3]
        goal_quat = goal_pose[3:7]
        wp2[[0, 1]] = goal_pos[[0, 1]]
        wp2[3:] = goal_quat


        wp3 = wp2.copy()
        wp3[2] = goal_pos[2]

        wps = [wp0, wp1, wp2, wp3]
        xyzs = [wp[:3] for wp in wps]
        # maintain vertical axis w/ gripper pointing down
        qs = [quaternion.from_float_array(wp[3:]) for wp in wps]

        self._goal_pose = goal_pos.copy()
        self._curr_pose = curr_pose.copy()
        if not self._avoid_obstacle_height or self._goal_pose[2] >= z and self._curr_pose[2] >= z:
            # No need to go up then down
            self._traj_pos = [min_jerk(xyzs[0], xyzs[3], t, self._total_horizon) for t in range(self._total_horizon)]
            self._traj_qs = [quaternion.slerp(qs[0], qs[1], 0, self._total_horizon, t) for t in range(self._total_horizon)]
        else:
            seg_horizon_up = int(0.25 * self._total_horizon)
            seg_horizon_down = int(0.35 * self._total_horizon)
            seg_horizon_middle = self._total_horizon - (seg_horizon_up + seg_horizon_down)
            self._traj_pos = [
                                 min_jerk(xyzs[0], xyzs[1], t, seg_horizon_up-1)
                                 for t in range(seg_horizon_up)
                             ] + [
                                 min_jerk(xyzs[1], xyzs[2], t, seg_horizon_middle-1)
                                 for t in range(seg_horizon_middle)
                             ] + [
                                 min_jerk(xyzs[2], xyzs[3], t, seg_horizon_down-1)
                                 for t in range(seg_horizon_down)
                             ]
            self._traj_qs = [
                                quaternion.slerp(qs[0], qs[1], 0, seg_horizon_up, t)
                                for t in range(seg_horizon_up)
                            ] + [
                                quaternion.slerp(qs[1], qs[2], 0, seg_horizon_middle, t)
                                for t in range(seg_horizon_middle)
                            ] + [
                                quaternion.slerp(qs[2], qs[3], 0, seg_horizon_down, t)
                                for t in range(seg_horizon_down)
                            ]
        #print("Generated trajectory of length", len(self._traj_pos))
        return {
            'T_plan': time() - s
        }

    @property
    def horizon(self):
        return len(self._traj_pos)

    def _call(self, curr_state, t, delta=False):
        ''' Expect curr_state to be (3 + 4 + 3 + 3) for:
        pos (xyz)
        quat (wxyz)
        lin vel (xyz)
        ang vel (xyz)
        '''

        traj_idx = min(t, self.horizon - 1)
        xd_pos = self._traj_pos[traj_idx]
        xd_quat = self._traj_qs[traj_idx]
        if delta:
            curr_pos = curr_state[:3]
            curr_quat = quaternion.from_float_array(curr_state[3:7])
            delta_quat = xd_quat*curr_quat.inverse()
            action = np.concatenate([xd_pos - curr_pos, quat_to_np(delta_quat, format="wxyz")])

        else:
            action = np.concatenate([xd_pos, quat_to_np(xd_quat, format="wxyz")])
        return action


class LQRController(BaseController):
    ''' Plans a trajectory in 1 part '''

    def __init__(self):
        super().__init__()

        self._n = 12  # size of state space
        self._m = 6  # size of action space
        self._R = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        self._Q = block_diag(
            np.diag([200, 200, 200, 300, 300, 300]),
            np.diag([200, 200, 200, 300, 300, 300])
        )

        self._Ks = np.diag([100] * 3 + [5] * 3)
        self._Ds = 2 * np.sqrt(self._Ks)

        self._mass = 3  # mass of the pusher
        self._block_w = 0.3  # width of block
        self._block_h = 0.3  # height of block

    @property
    def horizon(self):
        return self.K.shape[0]

    def _call(self, internal_state, t):
        ''' Expect internal_state to be (3 + 4 + 3 + 3) for:
            pos (xyz)
            quat (wxyz)
            lin vel (xyz)
            ang vel (xyz)
        '''
        if t < self.horizon:
            xe = np.zeros(self._n)
            xe[:3] = internal_state[:3] - self.x0[:3]
            xe_ang_axis = angle_axis_between_quats(quaternion.from_float_array(internal_state[3:7]),
                                                   quaternion.from_float_array(self.x0[3:7]))
            xe[3:6] = xe_ang_axis
            xe[6:] = internal_state[7:] - self.x0[7:]
            action = self.K[t] @ xe + self.k[t].flatten()
        else:
            x_pos = internal_state[:3]
            x_quat = quaternion.from_float_array(internal_state[3:7])
            x_vel = internal_state[7:]

            xf_quat = ee_yaw_to_np_quat(self._goal_yaw)

            xe_pos = x_pos - self._goal_pos
            xe_ang_axis = angle_axis_between_quats(x_quat, xf_quat)

            xe = np.concatenate([xe_pos, xe_ang_axis])

            action = -self._Ks @ xe - self._Ds @ x_vel

        return action.flatten()

    def _plan(self, internal_state, goal_pos, goal_yaw, z, dt, T_plan_max, T=6):
        ''' Expect internal_state to be (3 + 4 + 3 + 3) for:
            pos (xyz)
            quat (wxyz)
            lin vel (xyz)
            ang vel (xyz)

        Expect goal_pos to be
            (x,y,z) position
        '''

        x0_pos = internal_state[:3]
        x0_quat = quaternion.from_float_array(internal_state[3:7])
        self.x0 = internal_state
        xe = np.zeros((self._n, 1))
        x0 = np.zeros((self._n, 1))
        x0[:3, 0] = x0_pos
        x0[3, 0] = Quaternion(x=x0_quat.x, y=x0_quat.y, z=x0_quat.z, w=x0_quat.w).yaw_pitch_roll[0]
        x0[4, 0] = Quaternion(x=x0_quat.x, y=x0_quat.y, z=x0_quat.z, w=x0_quat.w).yaw_pitch_roll[1]
        x0[5, 0] = Quaternion(x=x0_quat.x, y=x0_quat.y, z=x0_quat.z, w=x0_quat.w).yaw_pitch_roll[2]
        params = {
            'n': self._n,
            'm': self._m,
            'dt': dt,
            'xe': xe,
            'Q': self._Q,
            'R': self._R,
            'T': T,
            'mass': self._mass,
            'block_w': self._block_w,
            'block_h': self._block_h
        }
        # Don't need to do the up down thing because already in obstacle free zone
        params['xe'][:3, 0] = goal_pos[:3] - x0_pos[:3]
        T_plan, K, k = LQR_xyz_ypr(params)
        self.K = K
        self.k = k
        self._goal_pos = goal_pos
        self._goal_yaw = goal_yaw
        return {
            'T_plan': T_plan
        }


class LQRControllerOutOfPlane(LQRController):
    ''' Plans a trajectory in 3 parts: 
        1) move to a separate z plane in _T_z_1 time
        2) goto to a point above the goal
        3) move to the goal in _T_z_2 time.
    '''

    def __init__(self):
        super().__init__()
        self._T_z_1 = 1
        self._T_z_2 = 3

    def _plan(self, internal_state, goal_pos, goal_yaw, z, dt, T_plan_max, T=6):
        ''' Expect internal_state to be (3 + 4 + 3 + 3) for:
            pos (xyz)
            quat (wxyz)
            lin vel (xyz)
            ang vel (xyz)
        Expect goal_pos to be
            (x,y,z) position
        '''

        x0_pos = internal_state[:3]
        x0_quat = quaternion.from_float_array(internal_state[3:7])

        # Move out of plane to desired z-location
        xe = np.zeros((self._n, 1))
        xe[2, 0] = z - x0_pos[2]
        params = {
            'n': self._n,
            'm': self._m,
            'T': self._T_z_1,
            'dt': dt,
            'xe': xe,
            'Q': self._Q,
            'R': self._R,
            'mass': self._mass,
            'block_w': self._block_w,
            'block_h': self._block_h
        }
        T_plan1, K1, k1 = LQR_xyz_ypr(params)

        # Move to goal location while maintaining z-location
        xe[:2, 0] = goal_pos[:2] - x0_pos[:2]
        xe[2, 0] = z - x0_pos[2]
        xf_quat = ee_yaw_to_np_quat(goal_yaw)
        xe_ang_axis = angle_axis_between_quats(xf_quat, x0_quat)
        xe[3:6, 0] = xe_ang_axis
        params['xe'] = xe
        params['T'] = T - self._T_z_1 - self._T_z_2
        T_plan2, K2, k2 = LQR_xyz_ypr(params)

        # # Move out of plane to desired z-location
        xe[2, 0] = goal_pos[2] - x0_pos[2]
        params['xe'] = xe
        params['T'] = self._T_z_2
        T_plan3, K3, k3 = LQR_xyz_ypr(params)

        self.K = np.concatenate((K1, K2, K3), axis=0)
        self.k = np.concatenate((k1, k2, k3), axis=0)
        self.x0 = internal_state
        self._goal_pos = goal_pos
        self._goal_yaw = goal_yaw

        T_plan = T_plan1 + T_plan2 + T_plan3

        self._goal_pos = goal_pos
        self._goal_yaw = goal_yaw
        return {
            'T_plan': T_plan
        }


class LQRWaypointControllerXYZ(BaseController):

    def __init__(self):
        super().__init__()

        self._n = 6  # size of state space
        self._m = 3  # size of action space
        self._R = 0.01 * np.eye(self._m)  # penalty on actions
        self._Q = 1000 * np.eye(self._n)  # penalty on states
        self._Q[2, 2] = 5e4  # putting high penalty in z-direction so that it remains in the plane of rods
        self._mass = 10  # mass of the pusher

    def _call(self, internal_state, t):
        if t < self.horizon:
            x = np.array([[internal_state[0]], [internal_state[1]], [internal_state[2]], [internal_state[7]],
                          [internal_state[8]], [internal_state[9]]])
            action_ = self.K[t] @ x + self.k[t]
            action = np.zeros(6)
            action[:3] = action_.flatten()
        else:
            action = np.zeros(6)
        return action

    @property
    def horizon(self):
        return self.K.shape[0]

    def _plan(self, initial_pos, waypoints, dt, T=4):
        params = {
            'n': self._n,
            'm': self._m,
            'T': T,
            'dt': dt,
            'waypoints': waypoints,
            'mass': self._mass,
            'Q': self._Q,
            'R': self._R
        }

        T_plan, K, k, T_exec = LQR_waypoints_xyz(params)

        # save state for execution time
        self.k = k
        self.K = K

        return {
            'T_plan': T_plan,
            'T_plan_exec': T_exec
        }


class PickController(PositionWaypointController):

    def __init__(self, Kp_pos=500, Kp_angle=5, goal_pose_tol=0.005, goal_yaw_tol=0.07,
                 open_width=0.02,
                 closed_width=0, dt = 0.01, real_robot=True):  # Okay for these to be on the large side, it's just when it starts to close
        super().__init__(Kp_pos=Kp_pos, Kp_angle=Kp_angle, dt=dt)
        self._goal_pose_tol = goal_pose_tol
        self._goal_yaw_tol = goal_yaw_tol
        self._open_width = open_width
        self._gripper_eps = 5e-3
        self._closed_width = closed_width
        self._start_width = None
        self._closing_start_time = None
        self._timesteps_to_open = np.max([1, np.ceil(0.4 /dt)]).astype(int)
        self._timesteps_to_close =np.max([2,np.floor(0.5 / dt)]).astype(int)
        if real_robot:
            self._total_horizon = np.floor(16 / dt).astype(int)
        else:
            self._total_horizon = np.floor(4 / dt).astype(int)
            #print("Total horizon is ", self._total_horizon)
        self._timesteps_to_give_up_and_close = np.ceil(0.5*self._total_horizon).astype(int)

    @property
    def open_width(self):
        return self._open_width

    @property
    def closed_width(self):
        return self._closed_width

    def _plan(self, curr_pose, goal_pose, z, curr_width=None):
        self._goal_position = goal_pose[:3]
        self._goal_quat = quaternion.from_float_array(goal_pose[3:7])
        #if curr_width > self._open_width - self._gripper_eps:
        #    self._timesteps_to_open = 0
        return super()._plan(curr_pose, goal_pose, z,
                             total_horizon=self._total_horizon - self._timesteps_to_open - self._timesteps_to_close)

    def _call(self, curr_state, t, delta=False):
        assert not delta
        if t < self._timesteps_to_open:  # time to open first
            movement_action = super()._call(curr_state[:-1], 0)
            if self._start_width is None:
                self._start_width = curr_state[-1]
            gripper_action = min_jerk(self._start_width, self.open_width, t, self._timesteps_to_open)
        else:
            movement_action = super()._call(curr_state[:-1], t - self._timesteps_to_open)
            curr_quat = quaternion.from_float_array(curr_state[3:7])
            pos_close = np.linalg.norm(
                curr_state[:3] - self._goal_position[:3]) < self._goal_pose_tol
            angle_close = np.linalg.norm(
                angle_axis_between_quats(curr_quat, self._goal_quat)) < self._goal_yaw_tol
            around_object =  pos_close and angle_close
            #print(f"control Pos close {pos_close} angle pos close {angle_close}")
            if around_object or t > self._timesteps_to_open + len(self._traj_pos) + self._timesteps_to_give_up_and_close:
                if self._closing_start_time is None:
                    self._closing_start_time = t
                if t - self._closing_start_time < self._timesteps_to_close:
                    gripper_action = min_jerk(self.open_width, self.closed_width, t - self._closing_start_time,
                                              self._timesteps_to_close)
                else:
                    gripper_action = self._closed_width
            else:
                gripper_action = self.open_width

        action = np.hstack([movement_action, gripper_action])
        return action


class LQRWaypointControllerXYZYaw(BaseController):

    def __init__(self):
        super().__init__()

        self._n = 8  # size of state space
        self._m = 4  # size of action space
        self._R = np.diag([0.01, 0.01, 0.01, 0.1])
        # penalty on states
        # putting high penalty in z-direction so that it remains in the plane of rods
        self._Q = np.diag([100, 100, 5e+3, 100, 100, 100, 100, 100]) 
        self._mass = 6  # mass of the pusher - tunable parameter
        self._block_w = 0.4  # width of block - tunable parameter
        self._block_h = 0.4  # height of block - tunable parameter
        self._Ks = np.diag([100] * 3 + [5] * 3)
        self._Ds = 2 * np.sqrt(self._Ks)

    def _call(self, internal_state, t, delta=False):
        if t < self.horizon:
            action = np.zeros(6)
            xe = np.zeros(self._n)
            xe[:3] = internal_state[:3] - self._x0[:3]  # xyz
            xe[3] = angle_axis_between_quats(quaternion.from_float_array(internal_state[3:7]),
                                             quaternion.from_float_array(self._x0[3:7]))[2]  # yaw
            xe[4:7] = internal_state[7:10] - self._x0[7:10]  # linear velocity
            xe[7] = internal_state[12] - self._x0[12]  # yaw velocity
            _action = self.K[t] @ xe + self.k[t].flatten()
            action[:3] = _action[:3]
            action[5] = _action[3]
            
        else:
            x_pos = internal_state[:3]
            x_quat = quaternion.from_float_array(internal_state[3:7])
            x_vel = internal_state[7:]

            xf_quat = quaternion.from_float_array(self._goal_pose[3:7])

            xe_pos = x_pos - self._goal_pose[:3]
            xe_ang_axis = angle_axis_between_quats(x_quat, xf_quat)
            xe_ang_axis[2] = xe_ang_axis[2] if abs(xe_ang_axis[2]) < np.pi / 2 \
                else xe_ang_axis[2] - np.sign(xe_ang_axis[2]) * np.pi

            xe = np.concatenate([xe_pos, xe_ang_axis])
            action = -self._Ks @ xe - self._Ds @ x_vel
        return action.flatten()

    @property
    def horizon(self):
        return self.K.shape[0]

    def _plan(self, internal_state, waypoints, dt, T=4):

        waypoints_control = np.zeros((self._n, waypoints.shape[1]))
        x0_quat = quaternion.from_float_array(internal_state[3:7])
        for i in range(waypoints.shape[1]):
            waypoints_control[:3, i] = waypoints[:3, i] - internal_state[:3]
            xf_quat = quaternion.from_float_array(waypoints[3:7, i])
            del_yaw = angle_axis_between_quats(xf_quat, x0_quat)[2]
            waypoints_control[3, i] = del_yaw if abs(del_yaw) < np.pi / 2 \
                else del_yaw - np.sign(del_yaw) * np.pi
        
        params = {
            'n': self._n,
            'm': self._m,
            'T': T,
            'dt': dt,
            'waypoints': waypoints_control,
            'mass': self._mass,
            'block_w': self._block_w,
            'block_h': self._block_h,
            'Q': self._Q,
            'R': self._R
        }

        T_plan, K, k, T_exec = LQR_waypoints_xyzyaw(params)

        # save state for execution time
        self._x0 = internal_state
        self.k = k
        self.K = K
        self._goal_pose = waypoints[:,-1]

        return {
            'T_plan': T_plan,
            'T_plan_exec': T_exec
        }


class LiftAndPlaceController(PositionWaypointController):
    def __init__(self, Kp_pos=20, Kp_angle=5, goal_pose_tol=0.008, goal_yaw_tol=0.07,
                 open_width=0.04, closed_width=0,  dt=0.01, real_robot=False):
        super().__init__(dt=dt, Kp_pos=Kp_pos, Kp_angle=Kp_angle)
        self._goal_pose_tol = goal_pose_tol
        self._goal_yaw_tol = goal_yaw_tol
        self._open_width = open_width
        self._closed_width = closed_width
        self._opening_start_time = None
        self._initial_width = None
        self._gripper_eps = 9e-3

        self._timesteps_to_open = np.max([1, np.ceil(0.4 /dt)]).astype(int)
        self._timesteps_to_close =np.max([1,np.floor(0.5 / dt)]).astype(int)
        if real_robot:
            self._total_horizon = np.floor(19 / dt).astype(int)
        else:
            self._total_horizon = np.floor(5 / dt).astype(int)
        self._timesteps_to_give_up_and_open = np.ceil(0.2*self._total_horizon).astype(int)

    @property
    def open_width(self):
        return self._open_width

    @property
    def closed_width(self):
        return self._closed_width

    def _plan(self, curr_pose, goal_pose, z, curr_width=None, total_horizon=None):
        self._goal_position = goal_pose[:3]
        self._goal_quat = quaternion.from_float_array(goal_pose[3:7])
        if curr_width < self._closed_width - self._gripper_eps:
            self._timesteps_to_close = 0
        return super()._plan(curr_pose, goal_pose, z,
                             total_horizon=self._total_horizon - self._timesteps_to_close - self._timesteps_to_open)

    def _call(self, curr_state, t, delta=False):
        if t < self._timesteps_to_close:
            movement_action = super()._call(curr_state[:-1], 0, delta=delta)
            if movement_action[2] > 0.012:
                movement_action[2] = 0.008 #Hack because physx doesn't seem to be working propertly and causes the franka gripper to go up for no reason
            if self._initial_width is None:
                self._initial_width = curr_state[-1]
            gripper_action = min_jerk(self._initial_width, self.closed_width, t, self._timesteps_to_close)
        else:
            movement_action = super()._call(curr_state[:-1], t - self._timesteps_to_close, delta=delta)
            curr_quat = quaternion.from_float_array(curr_state[3:7])
            around_object = np.linalg.norm(
                curr_state[:3] - self._goal_position[:3]) < self._goal_pose_tol and np.linalg.norm(
                angle_axis_between_quats(curr_quat, self._goal_quat)) < self._goal_yaw_tol
            if around_object or t > self._timesteps_to_open + self._timesteps_to_give_up_and_open + len(self._traj_pos):
                if self._opening_start_time is None:
                    self._opening_start_time = t
                if t - self._opening_start_time < self._timesteps_to_open:
                    gripper_action = min_jerk(self._closed_width, self.open_width, t - self._opening_start_time,
                                              self._timesteps_to_open)
                else:
                    gripper_action = self._open_width
            else:
                gripper_action = self.closed_width
        action = np.hstack([movement_action, gripper_action])
        return action

class LiftAndInsertController(LiftAndPlaceController):
    def __init__(self, Kp_pos=20, Kp_angle=5, goal_pose_tol=0.008, goal_yaw_tol=0.03,
                 open_width=0.04, closed_width=0,  dt=0.01, real_robot=False, y_insertion_length = 0.01):
        super().__init__(Kp_pos=Kp_pos, Kp_angle=Kp_angle, goal_pose_tol=goal_pose_tol, goal_yaw_tol = goal_yaw_tol, open_width=open_width, closed_width = closed_width, dt=dt, real_robot=real_robot)
        self._y_insertion_length = y_insertion_length
        self._freespace_impedances = [0,0,0,0,0,0] #indicates to not use impedance control
        self._insertion_impedances = [2000,600,600,100,100,100]
        self._z_slide_down_amount = 0.00
        self._timesteps_to_give_up_and_open = np.ceil(0.2*self._total_horizon).astype(int)


    def _call(self, curr_state, t, delta=False):
        ''' Expect curr_state to be (3 + 4 + 3 + 3) for:
        pos (xyz)
        quat (wxyz)
        lin vel (xyz)
        ang vel (xyz)
        '''

        traj_idx = min(t, self.horizon - 1)
        xd_pos = self._traj_pos[traj_idx]
        xd_quat = self._traj_qs[traj_idx]
        movement_action = np.concatenate([xd_pos, quat_to_np(xd_quat, format="wxyz")])
        #impedance_gains = self._traj_impedances[traj_idx]
        impedance_gains = []
        gripper_control = LiftAndPlaceController._call(self,curr_state, t, delta=delta)[-1]
        return np.hstack([movement_action, impedance_gains, gripper_control])

    def _plan(self, curr_pose, goal_pose, z, curr_width = None, total_horizon=300):
        s = time()
        self._goal_position = goal_pose[:3].copy()
        self._goal_position[1] += self._y_insertion_length
        self._goal_quat = quaternion.from_float_array(goal_pose[3:7])

        if curr_width < self._closed_width + self._gripper_eps:
            self._timesteps_to_close = 0

        wp0 = curr_pose

        wp1 = wp0.copy()
        wp1[2] = z

        wp2 = wp1.copy()
        goal_pos = goal_pose[:3]
        goal_quat = goal_pose[3:7]
        wp2[[0, 1]] = goal_pos[[0, 1]]
        wp2[3:] = goal_quat


        wp3 = wp2.copy()
        wp3[2] = goal_pos[2] - self._z_slide_down_amount
        wp4 = wp3.copy()
        wp4[1] += self._y_insertion_length


        wps = [wp0, wp1, wp2, wp3, wp4]
        xyzs = [wp[:3] for wp in wps]
        # maintain vertical axis w/ gripper pointing down
        qs = [quaternion.from_float_array(wp[3:]) for wp in wps]

        self._goal_pose = goal_pos.copy()
        self._curr_pose = curr_pose.copy()
        if self._goal_pose[2] >= z and self._curr_pose[2] >= z:
            # No need to go up then down
            self._traj_pos = [min_jerk(xyzs[0], xyzs[3], t, self._total_horizon) for t in range(self._total_horizon)]
            self._traj_qs = [quaternion.slerp(qs[0], qs[1], 0, self._total_horizon, t) for t in range(self._total_horizon)]
        else:
            seg_horizon_up = int(0.25 * self._total_horizon)
            seg_horizon_down = int(0.25 * self._total_horizon)
            seg_horizon_insert = int(0.25 * self._total_horizon)
            seg_horizon_middle = self._total_horizon - (seg_horizon_up + seg_horizon_down + seg_horizon_insert)
            self._traj_pos = [
                                 min_jerk(xyzs[0], xyzs[1], t, seg_horizon_up-1)
                                 for t in range(seg_horizon_up)
                             ] + [
                                 min_jerk(xyzs[1], xyzs[2], t, seg_horizon_middle-1)
                                 for t in range(seg_horizon_middle)
                             ] + [
                                 min_jerk(xyzs[2], xyzs[3], t, seg_horizon_down-1)
                                 for t in range(seg_horizon_down)
                             ]+ [
                                 min_jerk(xyzs[3], xyzs[4], t, seg_horizon_insert-1)
                                 for t in range(seg_horizon_insert)
                             ]
            self._traj_qs = [
                                quaternion.slerp(qs[0], qs[1], 0, seg_horizon_up, t)
                                for t in range(seg_horizon_up)
                            ] + [
                                quaternion.slerp(qs[1], qs[2], 0, seg_horizon_middle, t)
                                for t in range(seg_horizon_middle)
                            ] + [
                                quaternion.slerp(qs[2], qs[3], 0, seg_horizon_down, t)
                                for t in range(seg_horizon_down)
                            ]+ [
                                quaternion.slerp(qs[3], qs[4], 0, seg_horizon_insert, t)
                                for t in range(seg_horizon_insert)
                            ]
            self._traj_impedances = np.ones((self._total_horizon, 6)) * self._freespace_impedances
            self._traj_impedances[seg_horizon_up + seg_horizon_middle + seg_horizon_down:] = self._insertion_impedances
        return {
            'T_plan': time() - s
        }


class ReleaseController(BaseController):

    def __init__(self, open_width=0.04, closed_width=0, horizon=100):
        super().__init__()
        self._open_width = open_width
        self._closed_width = closed_width
        self._horizon = horizon

    def _call(self, curr_state, t):
        gripper_traj = np.linspace(self._open_width, self._closed_width, self._horizon - 1)
        gripper_action = gripper_traj[t]
        return [gripper_action]

    def _plan(self):
        return {
            'T_plan': 0
        }


class EEImpedanceTrajController(BaseController):

    def _plan(self, wps_pos_quat_gripper, traj_seg_horizons):
        s = time()

        self._wps_pos_quat_gripper = wps_pos_quat_gripper
        self._traj_seg_horizons = traj_seg_horizons

        traj_pos, traj_quat, traj_gripper = [], [], []
        for wp_idx in range(len(wps_pos_quat_gripper) - 1):
            seg_horizon = traj_seg_horizons[wp_idx]
            traj_pos.extend([
                            min_jerk(
                                wps_pos_quat_gripper[wp_idx][0],
                                wps_pos_quat_gripper[wp_idx + 1][0],
                                t, seg_horizon
                            ) for t in range(seg_horizon)
                        ])
            traj_quat.extend([
                            quaternion.slerp(
                                wps_pos_quat_gripper[wp_idx][1],
                                wps_pos_quat_gripper[wp_idx + 1][1],
                                0, seg_horizon, t
                            ) for t in range(seg_horizon)
                        ])
            traj_gripper.extend([
                            min_jerk(
                                wps_pos_quat_gripper[wp_idx][2],
                                wps_pos_quat_gripper[wp_idx + 1][2],
                                t, seg_horizon
                            ) for t in range(seg_horizon)
                        ])

        self._traj_pos = traj_pos
        self._traj_quat = traj_quat
        self._traj_gripper = traj_gripper

        return {
            'T_plan': time() - s
        }

    def _call(self, curr_state, t_step):
        t_traj = min(t_step, len(self._traj_pos) - 1)
        return {
            'pos': self._traj_pos[t_traj],
            'quat': self._traj_quat[t_traj],
            'gripper_width': self._traj_gripper[t_traj]
        }

    @property
    def horizon(self):
        return len(self._traj_pos)

    @property
    def n_wps(self):
        return len(self._wps_pos_quat_gripper) - 1

    def get_wp(self, t_wp):
        assert t_wp >= 0 and t_wp < self.n_wps
        return self._wps_pos_quat_gripper[t_wp + 1], self._traj_seg_horizons[t_wp]


class EEImpedanceTorqueTrajController(EEImpedanceTrajController):

    def __init__(self, Kp=300, Kr=5):
        super().__init__()
        self._Ks = np.diag([Kp] * 3 + [Kr] * 3)
        self._Ds = np.diag([4 * np.sqrt(Kp)] * 3 + [0 * np.sqrt(Kr)] * 3)

    def _call(self, curr_state, t_step):
        ''' curr_state is (3 + 4 + 3 + 3,) for ee pos, wxyz quat, lin vel, ang vel
        '''

        t_traj = min(t_step, len(self._traj_pos) - 1)
        xd_pos = self._traj_pos[t_traj]
        xd_quat = self._traj_quat[t_traj]

        x_pos = curr_state[:3]
        x_quat = quaternion.from_float_array(curr_state[3:7])
        x_vel = curr_state[7:]

        xe_pos = x_pos - xd_pos
        xe_ang_axis = angle_axis_between_quats(x_quat, xd_quat)
        xe = np.concatenate([xe_pos, xe_ang_axis])

        ee_tau = -self._Ks @ xe - self._Ds @ x_vel
        gripper_width = self._traj_gripper[t_traj]
        action = np.r_[ee_tau, [gripper_width]]
        return action

class OpenDrawerController(LiftAndInsertController):
    def __init__(self, Kp_pos=20, Kp_angle=5, goal_pose_tol=0.008, goal_yaw_tol=0.03,
                 open_width=0.04, closed_width=0,  dt=0.01, real_robot=False):
        super().__init__(Kp_pos=Kp_pos, Kp_angle=Kp_angle, goal_pose_tol=goal_pose_tol, goal_yaw_tol = goal_yaw_tol, open_width=open_width, closed_width = closed_width, dt=dt, real_robot=real_robot)
        self._freespace_impedances = [0,0,0,0,0,0] #indicates to not use impedance control
        self._insertion_impedances = [2000,600,600,100,100,100]


    def _call(self, curr_state, t, delta=False):
        ''' Expect curr_state to be (3 + 4 + 3 + 3) for:
        pos (xyz)
        quat (wxyz)
        lin vel (xyz)
        ang vel (xyz)
        '''

        traj_idx = min(t, self.horizon - 1)
        xd_pos = self._traj_pos[traj_idx]
        xd_quat = self._traj_qs[traj_idx]
        if delta:
            curr_pos = curr_state[:3]
            curr_quat = quaternion.from_float_array(curr_state[3:7])
            delta_quat = xd_quat * curr_quat.inverse()
            movement_action = np.concatenate([xd_pos - curr_pos, quat_to_np(delta_quat, format="wxyz")])
        else:
            movement_action = np.concatenate([xd_pos, quat_to_np(xd_quat, format="wxyz")])
        return movement_action

    def _plan(self, curr_pose, goal_pose, z, end_y, total_horizon=300):
        s = time()
        self._goal_position = goal_pose[:3].copy()
        self._goal_quat = quaternion.from_float_array(goal_pose[3:7])

        wp0 = curr_pose

        wp1 = wp0.copy()
        wp1[2] = z

        wp2 = wp1.copy()
        goal_pos = goal_pose[:3]
        goal_quat = goal_pose[3:7]
        wp2[[0, 1]] = goal_pos[[0, 1]]
        wp2[3:] = goal_quat


        wp3 = wp2.copy()
        wp3[2] = goal_pos[2]
        wp4 = wp3.copy()
        wp4[1] = end_y


        wps = [wp0, wp1, wp2, wp3, wp4]
        xyzs = [wp[:3] for wp in wps]
        # maintain vertical axis w/ gripper pointing down
        qs = [quaternion.from_float_array(wp[3:]) for wp in wps]

        self._goal_pose = goal_pos.copy()
        self._curr_pose = curr_pose.copy()
        if self._goal_pose[2] >= z and self._curr_pose[2] >= z:
            # No need to go up then down
            self._traj_pos = [min_jerk(xyzs[0], xyzs[3], t, self._total_horizon) for t in range(self._total_horizon)]
            self._traj_qs = [quaternion.slerp(qs[0], qs[1], 0, self._total_horizon, t) for t in range(self._total_horizon)]
        else:
            seg_horizon_up = int(0.25 * self._total_horizon)
            seg_horizon_down = int(0.2 * self._total_horizon)
            seg_horizon_insert = int(0.15 * self._total_horizon)
            seg_horizon_middle = self._total_horizon - (seg_horizon_up + seg_horizon_down + seg_horizon_insert)
            self._traj_pos = [
                                 min_jerk(xyzs[0], xyzs[1], t, seg_horizon_up-1)
                                 for t in range(seg_horizon_up)
                             ] + [
                                 min_jerk(xyzs[1], xyzs[2], t, seg_horizon_middle-1)
                                 for t in range(seg_horizon_middle)
                             ] + [
                                 min_jerk(xyzs[2], xyzs[3], t, seg_horizon_down-1)
                                 for t in range(seg_horizon_down)
                             ]+ [
                                 min_jerk(xyzs[3], xyzs[4], t, seg_horizon_insert-1)
                                 for t in range(seg_horizon_insert)
                             ]
            self._traj_qs = [
                                quaternion.slerp(qs[0], qs[1], 0, seg_horizon_up, t)
                                for t in range(seg_horizon_up)
                            ] + [
                                quaternion.slerp(qs[1], qs[2], 0, seg_horizon_middle, t)
                                for t in range(seg_horizon_middle)
                            ] + [
                                quaternion.slerp(qs[2], qs[3], 0, seg_horizon_down, t)
                                for t in range(seg_horizon_down)
                            ]+ [
                                quaternion.slerp(qs[3], qs[4], 0, seg_horizon_insert, t)
                                for t in range(seg_horizon_insert)
                            ]
            self._traj_impedances = np.ones((self._total_horizon, 6)) * self._freespace_impedances
            self._traj_impedances[seg_horizon_up + seg_horizon_middle + seg_horizon_down:] = self._insertion_impedances
        return {
            'T_plan': time() - s
        }

