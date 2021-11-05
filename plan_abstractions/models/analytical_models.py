import numpy as np
from autolab_core import RigidTransform

from plan_abstractions.models.contact_model import gripper_in_contact_with_drawer_vector
from plan_abstractions.utils import yaw_from_np_quat, point_in_box
from isaacgym_utils.math_utils import rpy_to_quat, quat_to_np
DRAWER_BOTTOM_DIMS = [0.25, 0.38, 0.02]
DRAWER_CHASSIS_DIMS = [0.25, 0.38, 0.09]
FREESPACE_PARAM_LENGTH = 4 #TODO find better way to detect incompatible params


class SEMSimpleFreeSpace:
    def __init__(self, num_rods, dim_state = None):
        self._num_rods = num_rods
        if dim_state is None:
            self._dim_state = 4 + (4 * self._num_rods)
        else:
            self._dim_state = dim_state
        self._dim_output = self._dim_state + 3

    @property
    def dim_state(self):
        return self._dim_state

    @property
    def fixed_input_size(self):
        return True

    def sample(self, cond, num_samples, truncate=None, unnormalized=False):
        cond_np = cond.flatten()
        state = cond_np[:self.dim_state]
        params = cond_np[self.dim_state:].copy()
        if len(params) > FREESPACE_PARAM_LENGTH:
            params = np.array([params[0], params[4], params[2], params[3]]) #assume is drawer
        try:
            ee_pos_coords = params[:3] - state[:3]
        except ValueError:
            print("Issue")
        ee_yaw_coords = (-params[3]) - state[3]
        ee_coords = np.hstack([ee_pos_coords, ee_yaw_coords])
        # only supporting diffs..
        rod_coords = np.zeros((self._num_rods * 4))
        cost = np.linalg.norm(state[:2] - params[:2]) + abs(state[2] - params[2])
        exec_time = 10 * cost
        plan_time = 0.001
        state_change = np.hstack([ee_coords, rod_coords ]).flatten()
        if len(state_change) < self._dim_state:
            padding = np.zeros(self._dim_state - len(state_change))
            state_change = np.hstack([state_change, padding])
        sem_state = np.hstack([state_change, [exec_time, cost, plan_time]]).reshape(1, -1)
        return {"x_hats": sem_state}


class SEMAnalyticalRodsAndRobot(SEMSimpleFreeSpace):
    def __init__(self, num_rods, dim_state = None):
        self._num_rods = num_rods
        if dim_state is None:
            self._dim_state = 4 + (4 * self._num_rods)
        else:
            self._dim_state = dim_state
        self._dim_output = self._dim_state + 3


    @property
    def dim_state(self):
        return self._dim_state

    @property
    def fixed_input_size(self):
        return True

    def sample(self, cond, num_samples, truncate=None, unnormalized=False):
        cond_np = cond.flatten()
        state = cond_np[:self.dim_state]
        params = cond_np[self.dim_state:]
        target_ee_pose = params[:3]
        ee_pos_coords = target_ee_pose - state[:3]
        rod_distances = [np.linalg.norm(state[:2] - state[4 + 4 * i:4 + 4 * i + 2]) for i in range(self._num_rods)]
        target_rod = np.argmin(rod_distances)
        if rod_distances[target_rod] > .178/2 and state[2] > 0.06: #TODO fix hardcoded length of rod. better metric to predict if in contact
            return super().sample(cond, num_samples=num_samples)
        current_rod_pose = state[4+4 * target_rod: 4+4 * target_rod + 3]
        current_rod_yaw = state[4+4*target_rod+3]
        ee_yaw_coords = (-params[3]) - state[3]
        ee_coords = np.hstack([ee_pos_coords, ee_yaw_coords])
        target_ee_yaw = state[3] + ee_yaw_coords
        # only supporting diffs..
        rod_coords = np.zeros((self._num_rods * 4))
        # Put the coords of the rod to be the same as the franka, then make it a transform

        init_ee_rot = RigidTransform.rotation_from_quaternion(quat_to_np(rpy_to_quat([0,0,state[3]]),format="wxyz"))
        gripper_to_world_before = RigidTransform(translation=state[:3], rotation = init_ee_rot, from_frame="gripper", to_frame="world")
        rod_to_world_before = RigidTransform(translation = current_rod_pose, rotation=RigidTransform.z_axis_rotation(current_rod_yaw), from_frame="pen", to_frame="world" )
        gripper_to_rod_before = rod_to_world_before.inverse() * gripper_to_world_before
        gripper_to_rod_before.translation[-1] = 0
        gripper_to_rod_vector = current_rod_pose[:2] - state[:2]
        theta = -ee_yaw_coords
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

        target_ee_rot = RigidTransform.z_axis_rotation(params[3])
        target_rod_pose_rt = RigidTransform(translation=target_ee_pose, rotation=target_ee_rot, from_frame="gripper", to_frame="world") * gripper_to_rod_before.inverse()
        rotated_gripper_to_rod  = rotation_matrix @ gripper_to_rod_vector
        target_rod_pose2 = target_ee_pose[:2] + rotated_gripper_to_rod


        target_rod_yaw = target_ee_yaw - state[3] + current_rod_yaw #yaw_from_np_quat(target_rod_pose_rt.quaternion)

        rod_pos_coords = target_rod_pose2 - current_rod_pose[:2]  #No change in z
        rod_yaw_diff = target_rod_yaw - current_rod_yaw
        rod_coords[4 * target_rod:4 * target_rod + 2] = rod_pos_coords
        rod_coords[4 * target_rod + 3] = rod_yaw_diff
        if self.dim_state > 12:
            #there's probably a drawer
            drawer_edge_coords = state[12:15]
            drawer_center_coords = drawer_edge_coords.copy()
            drawer_center_coords[1] += DRAWER_BOTTOM_DIMS[1]/2
            drawer_out = drawer_edge_coords[1] <= 0.08 #hard coded based on relative pose to base
            if point_in_box(target_rod_pose2[:2], drawer_center_coords, DRAWER_BOTTOM_DIMS):
                if drawer_out:
                    rod_coords[4*target_rod+2] = DRAWER_BOTTOM_DIMS[2] + 0.02
                else:
                    #assumed to still in in the case: not the best of assumptions but works
                    #for our distribution of drawer opening
                    rod_coords[4*target_rod+2] = DRAWER_CHASSIS_DIMS[2] + 0.02

                #print("Predicting higher rod pose because in drawer")


        cost = np.linalg.norm(state[:2] - params[:2]) + abs(state[2] - params[2])
        exec_time = 10 * cost
        plan_time = 0.001
        state_change = np.hstack([ee_coords, rod_coords ]).flatten()
        if len(state_change) < self._dim_state:
            padding = np.zeros(self._dim_state - len(state_change))
            state_change = np.hstack([state_change, padding])
        sem_state = np.hstack([state_change, [exec_time, cost, plan_time]]).reshape(1, -1)
        return {"x_hats": sem_state}

class SEMAnalyticalInsert(SEMAnalyticalRodsAndRobot):
    def sample(self, cond, num_samples, truncate=None, unnormalized=False):
        cond_np = cond.flatten()
        state = cond_np[:self.dim_state]
        params = cond_np[self.dim_state:]
        y_insertion_amount = params[-1]
        target_ee_pose = params[:3]
        target_ee_pose[1] += y_insertion_amount
        ee_pos_coords = target_ee_pose - state[:3]
        rod_distances = [np.linalg.norm(state[:2] - state[4 + 4 * i:4 + 4 * i + 2]) for i in range(self._num_rods)]
        target_rod = np.argmin(rod_distances)
        current_rod_pose = state[4+4 * target_rod: 4+4 * target_rod + 3]
        current_rod_yaw = state[4+4*target_rod+3]
        ee_yaw_coords = (-params[3]) - state[3]
        ee_coords = np.hstack([ee_pos_coords, ee_yaw_coords])
        # only supporting diffs..
        rod_coords = np.zeros((self._num_rods * 4))
        # Put the coords of the rod to be the same as the franka, then make it a transform

        init_ee_rot = RigidTransform.rotation_from_quaternion(quat_to_np(rpy_to_quat([0,0,state[3]]),format="wxyz"))
        gripper_to_world_before = RigidTransform(translation=state[:3], rotation = init_ee_rot, from_frame="gripper", to_frame="world")
        rod_to_world_before = RigidTransform(translation = current_rod_pose, rotation=RigidTransform.z_axis_rotation(current_rod_yaw), from_frame="pen", to_frame="world" )
        gripper_to_rod_before = rod_to_world_before.inverse() * gripper_to_world_before
        gripper_to_rod_before.translation[-1] = 0
        gripper_to_rod_vector = current_rod_pose[:2] - state[:2]
        theta = -ee_yaw_coords
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

        target_ee_rot = RigidTransform.z_axis_rotation(params[3])
        target_rod_pose_rt = RigidTransform(translation=target_ee_pose, rotation=target_ee_rot, from_frame="gripper", to_frame="world") * gripper_to_rod_before.inverse()
        rotated_gripper_to_rod  = rotation_matrix @ gripper_to_rod_vector
        target_rod_pose2 = target_ee_pose[:2] + rotated_gripper_to_rod


        target_rod_yaw = yaw_from_np_quat(target_rod_pose_rt.quaternion)

        rod_pos_coords = target_rod_pose2 - current_rod_pose[:2]  #No change in z
        rod_yaw_diff = target_rod_yaw - current_rod_yaw
        rod_coords[4 * target_rod:4 * target_rod + 2] = rod_pos_coords
        rod_coords[4 * target_rod + 3] = rod_yaw_diff

        cost = np.linalg.norm(state[:2] - params[:2]) + abs(state[2] - params[2])
        exec_time = 10 * cost
        plan_time = 0.001
        sem_state = np.hstack([ee_coords, rod_coords, [exec_time, cost, plan_time]]).reshape(1, -1)
        return {"x_hats": sem_state}

class SEMAnalyticalDrawerAndRobot(SEMSimpleFreeSpace):
    def __init__(self, num_rods, dim_state, drawer_edge_dims):
        self._num_rods = num_rods
        self._drawer_edge_dims = drawer_edge_dims
        self._gripper_space = 0.04
        if dim_state is None:
            self._dim_state = 4 + (4 * self._num_rods)
        else:
            self._dim_state = dim_state

    @property
    def dim_state(self):
        return self._dim_state #4 + (4 * self._num_rods) + 3 + 1

    @property
    def fixed_input_size(self):
        return True

    def sample(self, cond, num_samples, truncate=None, unnormalized=False):
        cond_np = cond.flatten()
        state = cond_np[:self.dim_state]
        params = cond_np[self.dim_state:]
        if len(params) == FREESPACE_PARAM_LENGTH:
            return super().sample(cond_np, num_samples)

        finger_width = state[-1]
        # only supporting diffs..
        #if gripper in contact with drawer. We dont need to do any rotation stuff, just move in y
        drawer_coords = np.zeros(3,)
        out_amount = params[-1] - params[1]
        if gripper_in_contact_with_drawer_vector(state,params, self._drawer_edge_dims, contact_distance = 0.02):
            drawer_coords[1] = out_amount + self._gripper_space #might need to be negated
        else:
            print("No opening detected")

        #converting into form for freespace because the format of opendrawer cond is [x, start_y, z, theta, end_y]
        super_cond_before_open = np.hstack([state, [params[0], params[1], params[2], params[3]]])
        super_cond_after_open = np.hstack([state, [params[0], params[4], params[2], params[3]]])
        super_result_before_open = super().sample(super_cond_before_open, num_samples)
        super_result_after_open = super().sample(super_cond_after_open, num_samples)
        ee_and_rod_coords_before_open  = super_result_before_open["x_hats"][:, :12].flatten()
        ee_and_rod_coords_after_open  = super_result_after_open["x_hats"][:, :12].flatten()
        goto_drawer_cost = np.linalg.norm(state[:2] - params[:2]) + abs(state[2] - params[2])
        opening_cost = abs(params[1] - params[4])
        cost = goto_drawer_cost + opening_cost #amount to open by
        exec_time = 10*cost #not actually used
        plan_time = 0.001
        sem_state = np.hstack([ee_and_rod_coords_after_open, drawer_coords, finger_width, [exec_time, cost, plan_time]]).reshape(1, -1)
        return {"x_hats": sem_state}
