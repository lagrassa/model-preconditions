import numpy as np
import quaternion
import time
import logging

from autolab_core import RigidTransform
from isaacgym import gymapi
from pillar_state import State
from isaacgym_utils.math_utils import transform_to_np, RigidTransform_to_transform, quat_to_np, np_to_vec3, \
    np_quat_to_quat, rpy_to_quat
from ..envs import FrankaRodEnv, FrankaDrawerEnv
from ..utils import yaw_from_quat, yaw_from_np_quat, ee_yaw_to_np_quat, angle_axis_between_quats, \
    pillar_state_obj_to_transform, xyz_yaw_to_transform, transform_to_xyz_yaw, r_flip_yz, get_pose_pillar_state, \
    is_pos_B_btw_A_and_C, params_cause_collision_franka, is_obj_in_gripper, get_rod_grasps_transforms, \
    get_object_names_in_pillar_state, yaw_to_np_quat, seven_dim_internal_state_to_pose, rod_intersects_fingers
from ..controllers.franka_controllers import PositionWaypointController, LQRControllerOutOfPlane, \
    LQRWaypointControllerXYZ, ReleaseController, PickController, LQRWaypointControllerXYZYaw, LiftAndPlaceController, \
    LQRController, LiftAndInsertController, OpenDrawerController
from .skills import Skill

GRIPPER_EPS = 5e-3 


logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)


class FreeSpaceMoveFranka(Skill):

    def __init__(self, num_rods=2, **kwargs):
        super().__init__(**kwargs)
        self.num_rods = num_rods
        self.param_shape = (4,)  # x,y,z,theta
        self.position_tol = 1e-2
        self.yaw_tol = np.deg2rad(4)
        self.lin_vel_tol = 1e-2
        self.ang_vel_tol = 5e-2

        self._max_goal_dist = 1
        self._max_goal_ang = np.deg2rad(180)
        self._goal_pos_range = np.array([
            [0.15, 0.75],  # x lims
            [-0.5, 0.5]  # y lims
        ])
        self._goal_yaw_range = np.deg2rad([-150, 150]) # +- 166 is about the franka joint limit
        self._z_height = 0.18  # TODO lagrassa have this automatically match with obstacle_free_height
        self._termination_buffer_time = 100
        self._gripper_eps = 5e-3
        self._terminate_on_timeout = False


    def pillar_state_to_internal_state(self, state):
        return np.array(state.get_values_as_vec([
            'frame:franka:ee:pose/position',
            'frame:franka:ee:pose/quaternion',
            'frame:franka:ee:pose/linear_velocity',
            'frame:franka:ee:pose/angular_velocity'
        ]))

    def state_precondition_satisfied(self, state):
        return True

    def precondition_satisfied(self, state, parameters, check_valid_goal_state=False, env=None):
        if not self.state_precondition_satisfied(state):
            logger.debug(f"Skill: {self.__class__.__name__} State precondition check failed.")
            return False

        if check_valid_goal_state and params_cause_collision_franka(state, parameters, FrankaRodEnv, gripper_eps=self._gripper_eps):
            logger.debug("Params cause collision")
            return False

        curr_pos = np.array(state.get_values_as_vec(['frame:franka:ee:pose/position']))[:2]
        goal_pos = parameters[:2]
        goal_dist = np.linalg.norm(goal_pos - curr_pos)
        close_dist = goal_dist < self._max_goal_dist

        curr_quat = quaternion.from_float_array(state.get_values_as_vec(['frame:franka:ee:pose/quaternion']))
        goal_yaw = parameters[3]

        goal_in_range = np.all(goal_pos[:2] >= self._goal_pos_range[:, 0]) and \
                        np.all(goal_pos[:2] <= self._goal_pos_range[:, 1]) and \
                        goal_yaw >= self._goal_yaw_range[0] and \
                        goal_yaw <= self._goal_yaw_range[1]

        logger.debug(
            f"Skill: {self.__class__.__name__} close dist: {close_dist} goal in range: {goal_in_range}")
        return close_dist and goal_in_range

    def _gen_object_centric_parameters(self, env, state):
        goal_transforms = env.get_goal_transforms_around_objects(state, plot=False)
        while True:
            for goal_transform in goal_transforms:
                x = goal_transform.p.x
                y = goal_transform.p.y
                z = max(goal_transform.p.z, env.grasp_height)
                theta = yaw_from_quat(goal_transform.r)
                parameters = np.array([x, y, z, theta])
                yield parameters

    def _gen_random_parameters(self, env, state):
        while True:
            curr_pos = get_pose_pillar_state(state, "franka:ee")
            close_dist = False
            while not close_dist:
                random_pos = np.random.uniform(low=self._goal_pos_range[:, 0],
                                            high=self._goal_pos_range[:, 1])
                goal_dist = np.linalg.norm(random_pos - curr_pos[:2])
                close_dist = goal_dist < self._max_goal_dist

            random_height = np.random.uniform(low=env.grasp_height, high=env.obstacle_free_height)
            parameters = [random_pos[0], random_pos[1], random_height]
            parameters = np.hstack([parameters, np.random.uniform(low=self._goal_yaw_range[0],
                                                                  high=self._goal_yaw_range[1])])
            yield parameters

    def _gen_relation_centric_parameters(self, env, state):
        # Should be irrelevant, raise an error if sampled
        while True:
            yield
            raise NotImplementedError

    @staticmethod
    def parameters_to_relative_parameters(parameters, ref_pillar_state, anchor_obj_name):
        a_T_w = pillar_state_obj_to_transform(
            ref_pillar_state, anchor_obj_name, align_z=True
        ).inverse()

        w_T_o = xyz_yaw_to_transform(parameters, for_ee=True)
        a_T_o = a_T_w * w_T_o

        relative_parameters = transform_to_xyz_yaw(a_T_o)
        return relative_parameters

    @staticmethod
    def relative_parameters_to_parameters(relative_parameters, ref_pillar_state, anchor_obj_name):
        w_T_a = pillar_state_obj_to_transform(
            ref_pillar_state, anchor_obj_name, align_z=True
        )

        xyz_yaw = relative_parameters.copy()

        anchor_obj_is_ee = anchor_obj_name == 'franka:ee'

        a_T_o = xyz_yaw_to_transform(xyz_yaw)

        if anchor_obj_is_ee:
            a_T_o.r = a_T_o.r * r_flip_yz

        w_T_o = w_T_a * a_T_o

        parameters = transform_to_xyz_yaw(w_T_o)
        return parameters

    def apply_action(self, env, env_idx, action):
        transform = gymapi.Transform(
            p=np_to_vec3(action[:3]),
            r=np_quat_to_quat(quaternion.from_float_array(action[3:7]))
        )
        if self._use_delta_actions:
            env.set_delta_attractor(transform, env_idx)
        else:
            env.set_attractor(transform, env_idx)


    def unpack_parameters(self, parameters):
        goal_pos = parameters[:3]
        goal_yaw = parameters[3]
        return np.array(goal_pos), goal_yaw

    def make_controllers(self, initial_states, parameters, T_plan_max, t, dt, real_robot, avoid_obstacle_height=True):
        info_plans = []
        controllers = []
        for env_idx, initial_state in enumerate(initial_states):
            internal_state = self.pillar_state_to_internal_state(initial_state)
            curr_pose, goal_pose = seven_dim_internal_state_to_pose(internal_state, parameters, env_idx)
            controller = PositionWaypointController(dt, avoid_obstacle_height=avoid_obstacle_height)
            info_plan = controller.plan(curr_pose, goal_pose,  self._z_height)

            controllers.append(controller)
            info_plans.append(info_plan)
        return controllers, info_plans

    def check_termination_condition(self, internal_state, parameters, t, controller=None, env_idx=None):
        goal_pos, goal_yaw = self.unpack_parameters(parameters)
        error_pos = goal_pos - internal_state[:3]

        goal_quat = ee_yaw_to_np_quat(goal_yaw)
        curr_quat = quaternion.from_float_array(internal_state[3:7])
        error_ang = np.linalg.norm(angle_axis_between_quats(curr_quat, goal_quat))

        pos_close = np.isclose(np.linalg.norm(error_pos), 0, atol=self.position_tol)
        ang_close = np.isclose(error_ang, 0, atol=self.yaw_tol)

        lin_vel_zero = np.isclose(np.linalg.norm(internal_state[7:10]), 0, atol=self.lin_vel_tol)
        ang_vel_zero = np.isclose(np.linalg.norm(internal_state[10:13]), 0, atol=self.ang_vel_tol)

        timeout = False
        if self._terminate_on_timeout and controller is not None:
            timeout = t >= controller.horizon + self._termination_buffer_time
        #print(f"Ang close: {ang_close}, Pos close {pos_close}")
        return timeout or pos_close and ang_close and lin_vel_zero and ang_vel_zero

class FreeSpaceMoveToGroundFranka(FreeSpaceMoveFranka):
    def precondition_satisfied(self, state, parameters, check_valid_goal_state=False, env=None):
        return super().precondition_satisfied(state, parameters, check_valid_goal_state = False, env=env)

    def make_controllers(self, initial_states, parameters, T_plan_max, t, dt, real_robot):
        return super().make_controllers(initial_states, parameters, T_plan_max, t, dt, real_robot, avoid_obstacle_height=False)

    def _gen_object_centric_parameters(self, env, state):
        obj_names = [name for name in get_object_names_in_pillar_state(state) if env.object_name in name]
        obj_transforms = [pillar_state_obj_to_transform(state, obj_name) for obj_name in obj_names]
        while True:
            for goal_transform in obj_transforms:
                x = goal_transform.p.x
                y = goal_transform.p.y
                z = 0.01
                theta = 0
                parameters = np.array([x, y, z, theta])
                yield parameters


class FreeSpaceMoveLQRFranka(Skill):

    def __init__(self, num_rods=2, **kwargs):
        super().__init__(**kwargs)
        self.num_rods = num_rods
        self.param_shape = (4,)  # x, y, z, theta

        self.position_tol = 5e-3
        self.yaw_tol = np.deg2rad(2)
        self.lin_vel_tol = 1e-2
        self.ang_vel_tol = 5e-2

        self._max_goal_dist = 0.3
        self._max_goal_ang = np.deg2rad(180)
        self._goal_pos_range = np.array([
            [0.2, 0.7],  # x lims
            [-0.3, 0.3]  # y lims
        ])
        self._goal_yaw_range = np.deg2rad([-150, 150]) # +- 166 is about the franka joint limit
        self._gripper_eps = 5e-3

        self._z_height = 0.1  # TODO lagrassa have this automatically match with obstacle_free_height
        self._termination_buffer_time = 100
        self._terminate_on_timeout = False

    def pillar_state_to_internal_state(self, state):
        return np.array(state.get_values_as_vec([
            'frame:franka:ee:pose/position',
            'frame:franka:ee:pose/quaternion',
            'frame:franka:ee:pose/linear_velocity',
            'frame:franka:ee:pose/angular_velocity'
        ]))

    def state_precondition_satisfied(self, state):
        return True

    def precondition_satisfied(self, state, parameters, check_valid_goal_state=False):
        if not self.state_precondition_satisfied(state):
            logger.debug(f"Skill: {self.__class__.__name__} State precondition check failed.")
            return False

        if check_valid_goal_state and params_cause_collision_franka(state, parameters, FrankaRodEnv, plot=False, gripper_eps=self._gripper_eps):
            return False

        obj_in_gripper, _, _ = is_obj_in_gripper(state, 0.01, 0.005)
        if obj_in_gripper:
            return False

        curr_pos = np.array(state.get_values_as_vec(['frame:franka:ee:pose/position']))[:2]
        goal_pos = parameters[:2]
        goal_dist = np.linalg.norm(goal_pos - curr_pos)
        close_dist = goal_dist < self._max_goal_dist

        if not close_dist:
            return False

        curr_quat = quaternion.from_float_array(state.get_values_as_vec(['frame:franka:ee:pose/quaternion']))
        goal_yaw = parameters[3]
        goal_quat = ee_yaw_to_np_quat(goal_yaw)
        goal_angle_dist = np.linalg.norm(angle_axis_between_quats(curr_quat, goal_quat))
        close_angle = goal_angle_dist < self._max_goal_ang

        if not close_angle:
            return False

        goal_in_range = np.all(goal_pos[:2] >= self._goal_pos_range[:, 0]) and \
                        np.all(goal_pos[:2] <= self._goal_pos_range[:, 1]) and \
                        goal_yaw >= self._goal_yaw_range[0] and \
                        goal_yaw <= self._goal_yaw_range[1]

        logger.debug(f"Skill: {self.__class__.__name__} close dist: {close_dist} close angle: {close_angle} goal_in_range: {goal_in_range}")
        return close_dist and close_angle and goal_in_range

    def _gen_object_centric_parameters(self, env, state):
        goal_transforms = env.get_goal_transforms_around_objects(state, plot=False)
        while True:
            for goal_transform in goal_transforms:
                x = goal_transform.p.x
                y = goal_transform.p.y
                z = goal_transform.p.z
                theta = yaw_from_quat(goal_transform.r)
                parameters = np.array([x, y, z, theta])
                yield parameters

    def _gen_random_parameters(self, env, state):
        while True:
            curr_pos = get_pose_pillar_state(state, "franka:ee")
            curr_yaw = yaw_from_np_quat(curr_pos[3:])
            random_dir = np.random.uniform(low=0, high=2 * np.pi)
            random_dist = np.random.uniform(low=0, high=self._max_goal_dist)
            des_yaw = np.random.uniform(low=curr_yaw - self._max_goal_dist,
                                        high=curr_yaw + self._max_goal_dist)
            random_height = np.random.uniform(low=env.grasp_height, high=env.obstacle_free_height)
            parameters = (
                curr_pos[0] + random_dist * np.cos(random_dir), curr_pos[1] + random_dist * np.sin(random_dir))
            parameters = np.hstack([parameters, random_height, des_yaw])
            yield parameters

    def _gen_relation_centric_parameters(self, env, state):
        while True:
            raise NotImplementedError()
            yield

    @staticmethod
    def parameters_to_relative_parameters(parameters, ref_pillar_state, anchor_obj_name):
        a_T_w = pillar_state_obj_to_transform(
            ref_pillar_state, anchor_obj_name, align_z=True
        ).inverse()

        w_T_o = xyz_yaw_to_transform(parameters, for_ee=True)
        a_T_o = a_T_w * w_T_o

        relative_parameters = transform_to_xyz_yaw(a_T_o)
        return relative_parameters

    @staticmethod
    def relative_parameters_to_parameters(relative_parameters, ref_pillar_state, anchor_obj_name):
        w_T_a = pillar_state_obj_to_transform(
            ref_pillar_state, anchor_obj_name, align_z=True
        )

        xyz_yaw = relative_parameters.copy()

        anchor_obj_is_ee = anchor_obj_name == 'franka:ee'

        a_T_o = xyz_yaw_to_transform(xyz_yaw)

        if anchor_obj_is_ee:
            a_T_o.r = a_T_o.r * r_flip_yz

        w_T_o = w_T_a * a_T_o

        parameters = transform_to_xyz_yaw(w_T_o)
        return parameters

    def apply_action(self, env, env_idx, action):
        env.apply_ee_force_torque(action, env_idx, maintain_elbow=False, gripper_orient=False)

    def unpack_parameters(self, parameters):
        goal_pos = parameters[:3]
        goal_yaw = parameters[3]
        return np.array(goal_pos), goal_yaw

    def make_controllers(self, initial_states, parameters, T_plan_max, t, dt):
        info_plans = []
        controllers = []
        for env_idx, initial_state in enumerate(initial_states):
            internal_state = self.pillar_state_to_internal_state(initial_state)
            goal_pos, goal_yaw = self.unpack_parameters(parameters[env_idx])
            if goal_pos[2] < self._z_height or internal_state[2] < self._z_height:
                controller = LQRControllerOutOfPlane()
            else:
                controller = LQRController()

            dt = initial_state.get_values_as_vec(["constants/dt"])[0]
            info_plan = controller.plan(internal_state, goal_pos, goal_yaw, self._z_height, dt, T_plan_max)
            controllers.append(controller)
            info_plans.append(info_plan)
        return controllers, info_plans

    def check_termination_condition(self, internal_state, parameters, t, controller=None, env_idx=None):
        goal_pos, goal_yaw = self.unpack_parameters(parameters)
        error_pos = goal_pos - internal_state[:3]

        goal_quat = ee_yaw_to_np_quat(goal_yaw)
        curr_quat = quaternion.from_float_array(internal_state[3:7])

        del_yaw = np.linalg.norm(angle_axis_between_quats(curr_quat, goal_quat))
        error_ang = np.linalg.norm(del_yaw if abs(del_yaw) < np.pi / 2 \
                                       else np.pi - del_yaw) #symmetry of the gripper

        pos_close = np.isclose(np.linalg.norm(error_pos), 0, atol=self.position_tol)
        ang_close = np.isclose(error_ang, 0, atol=self.yaw_tol)

        lin_vel_zero = np.isclose(np.linalg.norm(internal_state[7:10]), 0, atol=self.lin_vel_tol)
        ang_vel_zero = np.isclose(np.linalg.norm(internal_state[10:13]), 0, atol=self.ang_vel_tol)

        timeout = self._terminate_on_timeout and (t >= controller.horizon + self._termination_buffer_time)

        return pos_close and ang_close and lin_vel_zero and ang_vel_zero



class Pick(FreeSpaceMoveFranka):  # can instead inherit from LQR

    def __init__(self, *args, real_robot=True, grasp_offsets = [0.01], **kwargs):
        super().__init__(*args,  **kwargs)
        self._default_closed_width = 0.007 if real_robot else 0.01
        self._open_width = 0.06
        if real_robot:
            self.resting_countdown = 1
        else:
            self.resting_countdown = 1
        self._termination_t = None
        self._width_eps = GRIPPER_EPS
        self._check_grasped_tol_z = 0.015
        self._gripper_tol = 0.005
        self._should_reset_to_viewable = False
        self._grasp_offsets = grasp_offsets
        self._real_robot = real_robot

    # self._model = FreeSpaceModel()
    def pillar_state_to_internal_state(self, state):
        super_state = super().pillar_state_to_internal_state(state)
        gripper_width = np.array(state.get_values_as_vec(['frame:franka:gripper/width']))
        return np.hstack([super_state, gripper_width])

    def unpack_parameters(self, parameters):
        goal_pos = parameters[:3]
        goal_yaw = parameters[3]
        return np.array(goal_pos), goal_yaw

    def make_controllers(self, initial_states, parameters, T_plan_max, t, dt, real_robot=False):
        info_plans = []
        controllers = []
        for env_idx, initial_state in enumerate(initial_states):
            internal_state = self.pillar_state_to_internal_state(initial_state)
            rod_width = initial_state.get_values_as_vec(["constants/rod_dims"])[0]
            curr_width = internal_state[-1]
            curr_pose, goal_pose = seven_dim_internal_state_to_pose(internal_state, parameters, env_idx)
            controller = PickController(open_width=self._open_width, closed_width=rod_width - self._width_eps, dt=dt, real_robot=real_robot)  # TODO(lagrassa): assumes rods all same width
            info_plan = controller.plan(curr_pose, goal_pose, self._z_height, curr_width=curr_width)

            controllers.append(controller)
            info_plans.append(info_plan)
        return controllers, info_plans

    def apply_action(self, env, env_idx, action):
        start_time = time.time()
        transform = gymapi.Transform(
            p=np_to_vec3(action[:3]),
            r=np_quat_to_quat(quaternion.from_float_array(action[3:7]))
        )
        env.set_attractor(transform, env_idx)
        env.set_gripper_width_target(action[-1], env_idx)
        end_time = time.time()
        #logger.info((f"Time elapsed in apply_action for Pick: {end_time - start_time}"))
        #print(f"Time elapsed in apply_action for Pick: {end_time - start_time}")

    def check_termination_condition(self, internal_state, parameters, t, controller=None, env_idx=None):
        pos_complete = super().check_termination_condition(internal_state, parameters, t, controller, env_idx)
        if not pos_complete:
            return False # no point checking rest
        if controller is None:
            small_width = self._default_closed_width - 1e-3
        else:
            small_width = controller.closed_width + self._gripper_tol # + 4e-3
        gripper_closed = internal_state[-1] <= small_width
        can_terminate =  pos_complete and gripper_closed
        if not can_terminate:
            return False
        if self._termination_t is None:
            self._termination_t = t
        return t - self._termination_t >= self.resting_countdown


    def _gen_object_centric_parameters(self, env, state):
        dims = state.get_values_as_vec(["constants/rod_dims"])
        rod_grasp_transforms = get_rod_grasps_transforms(dims[1], dims[2], grasp_offset = self._grasp_offsets)
        rod_names = [name for name in get_object_names_in_pillar_state(state) if env.object_name in name]
        rod_transforms = [pillar_state_obj_to_transform(state, obj_name) for obj_name in rod_names]
        min_theta = -1.95
        max_theta = 3.7
        symmetries_angles = [-np.pi, 0, np.pi]
        symmetries_quats = [rpy_to_quat([0,0,yaw]) for yaw in symmetries_angles]
        gripper_curr_quat = quaternion.from_float_array(state.get_values_as_vec(['frame:franka:ee:pose/quaternion']))
        while True:
            for grasp_transform in rod_grasp_transforms:
                for rod_transform in rod_transforms:
                    goal_transform = rod_transform * grasp_transform
                    x = goal_transform.p.x
                    y = goal_transform.p.y
                    z = 0.012 #goal_transform.p.z - 8e-3
                    theta = yaw_from_quat(goal_transform.r)
                    curr_yaw = yaw_from_quat(gripper_curr_quat)
                    dist_from_currents = []
                    for symmetry_yaw in symmetries_angles:
                        angle_dist = np.abs((theta + symmetry_yaw) - curr_yaw) #wraparound will be a problem
                        dist_from_currents.append(angle_dist)
                    theta = theta + symmetries_angles[np.argmin(dist_from_currents)]
                    if theta > max_theta:
                        theta -= np.pi
                    if theta < min_theta:
                        theta += np.pi
                    parameters = np.array([x, y, z, theta])  # TODO do the other way too
                    yield parameters

    def precondition_satisfied(self, state, parameters, check_valid_goal_state=True, env=None):
        potential_state = state.copy()
        potential_state.update_property("frame:franka:gripper/width", self._open_width)
        rod0_dist = np.linalg.norm(np.array(get_pose_pillar_state(state, "rod0")[:2]) - parameters[:2])
        rod1_dist = np.linalg.norm(np.array(get_pose_pillar_state(state, "rod1")[:2]) - parameters[:2])
        if check_valid_goal_state:
            if params_cause_collision_franka(potential_state, parameters, franka_cls=FrankaDrawerEnv, gripper_eps=1e-3):
                return False
        if rod0_dist < rod1_dist:
            if get_pose_pillar_state(state,"rod0")[2] > 0.03:
                return False
        else:
            if get_pose_pillar_state(state,"rod1")[2] > 0.03:
                return False


        if not super().state_precondition_satisfied(state):
            return False

        obj_in_gripper, smallest_obj_width, object_name = is_obj_in_gripper(state, self._check_grasped_tol_z, self._gripper_tol)
        if obj_in_gripper:
            logger.debug(f"Skill: {self.__class__.__name__} already has object in hand")
            return False
        return True


class LQRWaypointsXYZYawFranka(Skill):

    def __init__(self, num_rods=2, **kwargs):
        super().__init__(**kwargs)
        self.num_rods = num_rods
        self.param_shape = (3 + num_rods,)

        self.position_tol = 5e-3
        self.yaw_tol = np.deg2rad(2)
        self.lin_vel_tol = 1e-2
        self.ang_vel_tol = 5e-2

        self._max_goal_dist = 0.3
        self._max_goal_ang = np.deg2rad(180)
        self._goal_pos_range = np.array([
            [0.2, 0.7],  # x lims
            [-0.3, 0.3]  # y lims
        ])
        self._goal_yaw_range = np.deg2rad([-150, 150]) # +- 166 is about the franka joint limit
        self._max_z = 0.05

        self._termination_buffer_time = 100
        self._z_height = 0.015

        self._rod_perp_offset = 0.1
        self._yaw_diff_thresh = np.deg2rad(30)

    def pillar_state_to_internal_state(self, state):
        return np.array(state.get_values_as_vec([
            'frame:franka:ee:pose/position',
            'frame:franka:ee:pose/quaternion',
            'frame:franka:ee:pose/linear_velocity',
            'frame:franka:ee:pose/angular_velocity'
        ]))

    def unpack_parameters(self, parameters):
        goal_pos = parameters[:2]
        goal_yaw = parameters[2]
        rods_to_push = np.where(parameters[3:] == 1)[0]
        return np.array(goal_pos), goal_yaw, rods_to_push

    def state_precondition_satisfied(self, state):
        """Checks if there is any rod between the fingers."""
        internal_state = self.pillar_state_to_internal_state(state)
        is_z_low_enough = abs(internal_state[2]) < self._max_z
        if not is_z_low_enough:
            return False
        all_rods = np.arange(self.num_rods)
        finger_pos = np.array(state.get_values_as_vec([
            'frame:franka:finger_left:pose/position',
            'frame:franka:finger_right:pose/position'
        ]))
        left_finger_pos, right_finger_pos = finger_pos[:2], finger_pos[3:5]

        good_rods = []
        rod_dim_y = state.get_values_as_vec([f'constants/rod_dims'])[1]
        for rod_idx in all_rods:
            rod_pos = state.get_values_as_vec([f'frame:rod{rod_idx}:pose/position'])[:2]
            rod_quat = quaternion.from_float_array(state.get_values_as_vec([f'frame:rod{rod_idx}:pose/quaternion']))
            rod_not_between_fingers = not rod_intersects_fingers(rod_pos, rod_quat, rod_dim_y, left_finger_pos, right_finger_pos)

            good_rods.append(rod_not_between_fingers)
        return np.all(good_rods)

    def precondition_satisfied(self, state, parameters, check_valid_goal_state=False):
        """

        Args:
            state:
            parameters:

        Returns: The evaluation of 4 conditions:
            - if the goals are close enough to the current poses
            - if the rods are between the pusher and each goal, so it can perform the action with sweeping
            - if the gripper is in the plane of the table
            - atleast one rod should be pushed
            - yaw should be a resonable value 

        """
        if not self.state_precondition_satisfied(state):
            logger.debug(f"Skill: {self.__class__.__name__} State precondition check failed.")
            return False

        goal_pos, goal_yaw, rods_to_push = self.unpack_parameters(parameters)
        if abs(goal_yaw) < 0.15:
            #print("Small goal yaw")
            return False
        internal_state = self.pillar_state_to_internal_state(state)
        
        gripper_pos = internal_state[:2]
        close_dist = np.linalg.norm(goal_pos - gripper_pos) < self._max_goal_dist

        if not close_dist:
            return False
        

        finger_pos = np.array(state.get_values_as_vec([
            'frame:franka:finger_left:pose/position',
            'frame:franka:finger_right:pose/position'
        ]))
        left_finger_pos, right_finger_pos = finger_pos[:2], finger_pos[3:5]
        
        gripper_curr_quat = quaternion.from_float_array(state.get_values_as_vec(['frame:franka:ee:pose/quaternion']))
        gripper_goal_quat = yaw_to_np_quat(goal_yaw)
        goal_angle_dist = np.linalg.norm(angle_axis_between_quats(gripper_curr_quat, gripper_goal_quat))
        close_angle = goal_angle_dist < self._max_goal_ang

        goal_in_range = np.all(goal_pos[:2] >= self._goal_pos_range[:, 0]) and \
                        np.all(goal_pos[:2] <= self._goal_pos_range[:, 1]) and \
                        goal_yaw >= self._goal_yaw_range[0] and \
                        goal_yaw <= self._goal_yaw_range[1]
        
        if not goal_in_range:
            return False

        angle_gripper_to_goal = np.arctan2(goal_pos[0] - gripper_pos[0], goal_pos[1] - gripper_pos[1])
        des_rod_quat = quaternion.from_euler_angles([0, 0, angle_gripper_to_goal - np.pi / 2])
        # checking if the rods are between the pusher and each goal
        good_rods = []
        rod_dim_y = state.get_values_as_vec([f'constants/rod_dims'])[1]
        for rod_idx in rods_to_push:
            rod_pos = state.get_values_as_vec([f'frame:rod{rod_idx}:pose/position'])[:2]
            rod_quat = quaternion.from_float_array(state.get_values_as_vec([f'frame:rod{rod_idx}:pose/quaternion']))

            # checking if the rods are between the pusher and each goal
            in_between, projection = is_pos_B_btw_A_and_C(gripper_pos, rod_pos, goal_pos)
            # perpendicular dist. of rod from line joining gripper and goal
            perp_dist = np.linalg.norm(rod_pos - projection)

            _yaw_delta_before_symmetry = np.linalg.norm(angle_axis_between_quats(rod_quat, des_rod_quat))
            rod_yaw_delta_from_desired = _yaw_delta_before_symmetry if abs(_yaw_delta_before_symmetry) < np.pi / 2 else np.pi - _yaw_delta_before_symmetry
            goal_yaw_rod_yaw_dist_before_symmetry = np.linalg.norm(angle_axis_between_quats(rod_quat, gripper_goal_quat))
            goal_yaw_rod_yaw_dist = goal_yaw_rod_yaw_dist_before_symmetry if goal_yaw_rod_yaw_dist_before_symmetry < np.pi / 2 else np.pi - goal_yaw_rod_yaw_dist_before_symmetry
            goal_yaw_not_far_from_rod_yaw  = goal_yaw_rod_yaw_dist < self._yaw_diff_thresh

            rod_not_between_fingers = not rod_intersects_fingers(rod_pos, rod_quat, rod_dim_y, left_finger_pos, right_finger_pos)
            # logger.info(f"   \t            in_between: {in_between}\n" \
            #             f"   \t             perp_dist: {perp_dist:.4f} ({perp_dist < self._rod_perp_offset})\n" \
            #             f"   \t             rod_yaw_delta_from_desired: {rod_yaw_delta_from_desired:.4f} ({rod_yaw_delta_from_desired < self._yaw_diff_thresh})\n" \
            #             f"   \t rod_not_bw_fingers: {rod_not_between_fingers}\n" \
            #             f"  left_finger_pos: ({left_finger_pos[0]:.3f}, {left_finger_pos[1]:.3f}), right_finger: ({right_finger_pos[0]:.3f}, {right_finger_pos[1]:.3f}), rod_pos: ({rod_pos[0]:.3f}, {rod_pos[1]:.3f})\n"  \
            #     )

            good_rods.append(
                in_between and \
                perp_dist < self._rod_perp_offset and \
                rod_yaw_delta_from_desired < self._yaw_diff_thresh and \
                goal_yaw_not_far_from_rod_yaw and \
                rod_not_between_fingers
            )

        logger.debug(
            f"Skill: {self.__class__.__name__}, close dist: {close_dist}, close angle: {close_angle}, rods on way to goal: {good_rods},  goal_in_range: {goal_in_range}")
        return close_dist and close_angle and goal_in_range and np.all(good_rods)  and (rods_to_push.shape[0] > 0)

    def _gen_object_centric_parameters(self, env, state):
        goal_transforms = env.get_goal_transforms_around_objects(state, plot=False)
        while True:
            for goal_transform in goal_transforms:
                x = goal_transform.p.x
                y = goal_transform.p.y
                theta = yaw_from_quat(goal_transform.r)
                while True:
                    rods_to_push = np.random.randint(0, 2, self.num_rods)
                    if np.sum(rods_to_push) > 0:
                        break
                parameters = np.append(np.array([x, y, theta]), rods_to_push)
                yield parameters

    def _gen_random_parameters(self, env, state):
        curr_pos = get_pose_pillar_state(state, "franka:ee")
        curr_yaw = yaw_from_np_quat(curr_pos[3:])
        while True:
            random_dir = np.random.uniform(low=0, high=2 * np.pi)
            random_dist = np.random.uniform(low=0, high=self._max_goal_dist)
            x = curr_pos[0] + random_dist * np.cos(random_dir)
            y = curr_pos[1] + random_dist * np.sin(random_dir)
            while True:
                rods_to_push = np.random.randint(0, 2, self.num_rods)
                if np.sum(rods_to_push) > 0:
                    break
            parameters = np.hstack([
                np.array([x, y]), 
                np.random.uniform(
                    low=curr_yaw - self._max_goal_dist, 
                    high=curr_yaw + self._max_goal_dist),
                rods_to_push
            ])
            yield parameters

    def _gen_relation_centric_parameters(self, env, state):
        curr_pos = get_pose_pillar_state(state, "franka:ee")
        goal_transforms = env.get_goal_transforms_around_objects(state, plot=False)
        while True:
            for goal_transform in goal_transforms:
                x = goal_transform.p.x
                y = goal_transform.p.y
                curr_dist = np.linalg.norm(curr_pos[:2] - np.array([x, y]))
                des_theta = yaw_from_quat(goal_transform.r)
                if curr_dist > self._max_goal_dist:
                    random_dist = 0  # this parameter will fail the precondition check but that shouldn't be handled at this phase
                else:
                    random_dist = np.random.uniform(low=curr_dist, high=self._max_goal_dist - curr_dist)
                # go farther in that direction
                dir = np.arctan2((y - curr_pos[1]), (x - curr_pos[0]))
                while True:
                    rods_to_push = np.random.randint(0, 2, self.num_rods)
                    if np.sum(rods_to_push) > 0:
                        break
                parameters = np.hstack([
                    np.array([x + random_dist * np.cos(dir), y + random_dist * np.sin(dir)]), 
                    des_theta,
                    rods_to_push
                ])
                yield parameters

    @staticmethod
    def parameters_to_relative_parameters(parameters, ref_pillar_state, anchor_obj_name):
        a_T_w = pillar_state_obj_to_transform(
            ref_pillar_state, anchor_obj_name, align_z=True
        ).inverse()

        xyz_yaw = np.array((parameters[0], parameters[1], 0, parameters[2]))
        w_T_o = xyz_yaw_to_transform(xyz_yaw, for_ee=True)
        a_T_o = a_T_w * w_T_o

        out = transform_to_xyz_yaw(a_T_o)
        relative_parameters = np.hstack([out[:2], out[3], parameters[3:]])
        return relative_parameters

    @staticmethod
    def relative_parameters_to_parameters(relative_parameters, ref_pillar_state, anchor_obj_name):
        w_T_a = pillar_state_obj_to_transform(
            ref_pillar_state, anchor_obj_name, align_z=True
        )

        xyz_yaw = np.array((relative_parameters[0], relative_parameters[1], 0, relative_parameters[2]))

        anchor_obj_is_ee = anchor_obj_name == 'franka:ee'

        a_T_o = xyz_yaw_to_transform(xyz_yaw)

        if anchor_obj_is_ee:
            a_T_o.r = a_T_o.r * r_flip_yz

        w_T_o = w_T_a * a_T_o

        out = transform_to_xyz_yaw(w_T_o)
        parameters = np.hstack([out[:2], out[3], relative_parameters[3:]])
        return parameters

    def apply_action(self, env, env_idx, action):
        env.apply_ee_force_torque(action, env_idx, maintain_elbow=True, gripper_orient=True)

    def check_termination_condition(self, internal_state, parameters, t, controller=None, env_idx=None):

        goal_pos, goal_yaw, _ = self.unpack_parameters(parameters)
        error_pos = goal_pos - internal_state[:2]

        goal_quat = ee_yaw_to_np_quat(goal_yaw)
        curr_quat = quaternion.from_float_array(internal_state[3:7])

        del_yaw = angle_axis_between_quats(curr_quat, goal_quat)[2]
        error_ang = np.linalg.norm(del_yaw if abs(del_yaw) < np.pi / 2 \
                                       else del_yaw - np.sign(del_yaw) * np.pi)

        pos_close = np.isclose(np.linalg.norm(error_pos), 0, atol=self.position_tol)
        ang_close = np.isclose(error_ang, 0, atol=self.yaw_tol)

        lin_vel_zero = np.isclose(np.linalg.norm(internal_state[7:10]), 0, atol=self.lin_vel_tol)
        ang_vel_zero = np.isclose(np.linalg.norm(internal_state[10:13]), 0, atol=self.ang_vel_tol)

        return pos_close and ang_close and lin_vel_zero and ang_vel_zero

    def make_controllers(self, initial_states, parameters, T_plan_max, t, dt):
        info_plans = []
        controllers = []

        for env_idx, initial_state in enumerate(initial_states):
            internal_state = self.pillar_state_to_internal_state(initial_state)
            goal_pos, goal_yaw, rods_to_push = self.unpack_parameters(parameters[env_idx])

            controller = LQRWaypointControllerXYZYaw()
            waypoints = np.zeros((7, rods_to_push.shape[0] + 1))
            waypoints[:2, -1] = goal_pos
            waypoints[2, -1] = self._z_height
            goal_quat = ee_yaw_to_np_quat(goal_yaw)
            waypoints[3:, -1] = quaternion.as_float_array(goal_quat)
            for i in range(rods_to_push.shape[0]):
                waypoints[:2, i] = initial_state.get_values_as_vec([f'frame:rod{rods_to_push[i]}:pose/position'])[
                                   :2]
                waypoints[2, i] = self._z_height
                yaw_rod = yaw_from_np_quat(
                    initial_state.get_values_as_vec([f'frame:rod{rods_to_push[i]}:pose/quaternion']))
                waypoints[3:, i] = quaternion.as_float_array(ee_yaw_to_np_quat(yaw_rod))

            # sort waypoints
            dist_vec = np.linalg.norm(waypoints[:2, :-1] - internal_state[:2].reshape(2, 1), axis=0)
            indx = np.append(np.argsort(dist_vec), -1)
            waypoints = waypoints[:, indx]

            dt = initial_state.get_values_as_vec(["constants/dt"])[0]

            info_plan = controller.plan(internal_state, waypoints, dt)

            controllers.append(controller)
            info_plans.append(info_plan)
        return controllers, info_plans


class LiftAndPlace(FreeSpaceMoveFranka):  # can instead inherit from LQR
    def __init__(self, *args, drop=False,  **kwargs):
        super().__init__(*args, **kwargs)
        self._gripper_tol = 0.005
        self._check_grasped_tol_z = 0.03
        self._open_width = 0.035
        self._width_eps = GRIPPER_EPS
        self._random_height_noise = 0.01
        self._drop = drop

    # self._model = FreeSpaceModel()
    def pillar_state_to_internal_state(self, state):
        super_state = super().pillar_state_to_internal_state(state)
        gripper_width = np.array(state.get_values_as_vec(['frame:franka:gripper/width']))
        return np.hstack([super_state, gripper_width])

    def make_controllers(self, initial_states, parameters, T_plan_max, t, dt, real_robot=False):
        controllers = []
        info_plans = []
        for env_idx, initial_state in enumerate(initial_states):
            internal_state = self.pillar_state_to_internal_state(initial_state)
            rod_width = initial_state.get_values_as_vec(["constants/rod_dims"])[0]
            curr_pose, goal_pose = seven_dim_internal_state_to_pose(internal_state, parameters, env_idx)
            controller = LiftAndPlaceController(open_width=self._open_width, closed_width=rod_width-self._width_eps, dt=dt, real_robot = real_robot)  # TODO lagrassa assumes rods all same width
            info_plan = controller.plan(curr_pose, goal_pose, self._z_height, curr_width = internal_state[-1])

            controllers.append(controller)
            info_plans.append(info_plan)
        return controllers, info_plans

    def apply_action(self, env, env_idx, action):
        start_time = time.time()
        super().apply_action(env, env_idx, action)
        env.set_gripper_width_target(action[-1], env_idx)
        end_time = time.time()
        #logger.info(f"Time elapsed drop apply_action {end_time- start_time}")
        #print(f"Time elapsed drop apply_action {end_time- start_time}")

    def check_termination_condition(self, internal_state, parameters, t, controller=None, env_idx=None):
        pos_complete = super().check_termination_condition(internal_state, parameters, t, controller=controller, env_idx=env_idx)
        gripper_open = internal_state[-1] > self._open_width - self._width_eps
        return pos_complete and gripper_open

    def _gen_random_parameters(self, env, state):
        while True:
            curr_pos = get_pose_pillar_state(state, "franka:ee")
            close_dist = False
            while not close_dist:
                random_pos = np.random.uniform(low=self._goal_pos_range[:, 0],
                                            high=self._goal_pos_range[:, 1])
                goal_dist = np.linalg.norm(random_pos - curr_pos[:2])
                close_dist = goal_dist < self._max_goal_dist

            if self._drop:
                random_height = np.random.uniform(low=env.obstacle_free_height, high=env.obstacle_free_height + self._random_height_noise)
            else:
                random_height = np.random.uniform(low=env.grasp_height, high = env.grasp_height + self._random_height_noise)

            parameters = [random_pos[0], random_pos[1], random_height]
            parameters = np.hstack([parameters, np.random.uniform(low=self._goal_yaw_range[0],
                                                                  high=self._goal_yaw_range[1])])
            yield parameters

    def _gen_object_centric_parameters(self, env, state):
        goal_transforms = env.get_goal_transforms_around_objects(state, plot=False)
        while True:
            for goal_transform in goal_transforms:
                x = goal_transform.p.x
                y = goal_transform.p.y
                if self._drop:
                    z = env.obstacle_free_height
                else:
                    z = goal_transform.p.z
                theta = yaw_from_quat(goal_transform.r)
                parameters = np.array([x, y, z, theta - np.pi / 2])  # TODO do the other way too
                yield parameters

    def _gen_relation_centric_parameters(self, env, state):
        while True:
            raise NotImplementedError()
            yield

    def state_precondition_satisfied(self, state):
        if not super().state_precondition_satisfied(state):
            return False

        obj_in_gripper, smallest_obj_width, object_name = is_obj_in_gripper(state, self._check_grasped_tol_z, self._gripper_tol)
        if not obj_in_gripper:
            logger.debug(f"Skill: {self.__class__.__name__} no object in hand")
            return False
        return True

    def precondition_satisfied(self, state, parameters, env=None, check_valid_goal_state=True):
        if not super().precondition_satisfied(state, parameters, check_valid_goal_state = check_valid_goal_state):
            return False

        if not self.state_precondition_satisfied(state):
            logger.debug(f"Skill: {self.__class__.__name__} State precondition check failed.")
            return False

        if check_valid_goal_state:
            obj_in_gripper, smallest_obj_width, object_name = is_obj_in_gripper(state, self._check_grasped_tol_z, self._gripper_tol)
            potential_state = State.create_from_serialized_string(state.get_serialized_string())
            potential_state.update_property("frame:franka:gripper/width", self._open_width)
            ee_pose = get_pose_pillar_state(state, "franka:ee")
            obj_pose = get_pose_pillar_state(state, object_name)
            T_ee_to_world = RigidTransform(translation=ee_pose[:3],
                                           rotation=RigidTransform.rotation_from_quaternion(ee_pose[3:]),
                                           to_frame="world", from_frame="ee")
            T_obj_to_world = RigidTransform(translation=obj_pose[:3],
                                            rotation=RigidTransform.rotation_from_quaternion(obj_pose[3:]),
                                            to_frame="world", from_frame="obj")
            T_obj_to_ee = T_ee_to_world.inverse() * T_obj_to_world
            T_obj_to_ee_transform = RigidTransform_to_transform(T_obj_to_ee)

            commanded_parameters = parameters.copy()
            commanded_parameters[3] *= -1
            T_ee_to_world_new_transform = xyz_yaw_to_transform(commanded_parameters[:4])
            T_ee_to_world_new_transform.r = T_ee_to_world_new_transform.r * r_flip_yz
            new_T_obj_to_world_transform  = T_ee_to_world_new_transform * T_obj_to_ee_transform
            potential_state.update_property(f"frame:{object_name}:pose/position", transform_to_np(new_T_obj_to_world_transform, format="wxyz")[:3])
            potential_state.update_property(f"frame:{object_name}:pose/quaternion", transform_to_np(new_T_obj_to_world_transform, format="wxyz")[3:])
            cause_coll, potential_pillar_state =  params_cause_collision_franka(potential_state, parameters, FrankaRodEnv, ignore_in_hand = object_name, return_state=True)
            if cause_coll:
                logging.debug("Parameters cause collision")
                return False

        return True

class LiftAndDrop(LiftAndPlace):  # can instead inherit from LQR
    def __init__(self, *args, real_robot=True,**kwargs):
        super().__init__(*args, drop=True,  **kwargs)
        self._super_termination_t = None
        self._real_robot = real_robot
        if real_robot:
            self.resting_countdown = 0
        else:
            self.resting_countdown = 40 #40 #timesteps to wait until object has settled, because IG seems to have horrible contact dynamics...
        self._should_reset_to_viewable=True

    def precondition_satisfied(self, state, parameters, check_valid_goal_state=False, env=None):
        return super().precondition_satisfied(state, parameters, check_valid_goal_state=False, env=env)

    def check_termination_condition(self, internal_state, parameters, t, controller=None, env_idx=None):
        super_terminated =  super().check_termination_condition(internal_state, parameters, t, controller=controller, env_idx=env_idx)
        if not super_terminated:
            return False
        if self._super_termination_t is None:
            self._super_termination_t = t
        return t - self._super_termination_t >= self.resting_countdown

class LiftAndInsert(LiftAndPlace):  # can instead inherit from LQR
    def __init__(self, *args, real_robot=True, **kwargs):
        super().__init__(*args, drop=True, **kwargs)
        self._super_termination_t = None
        self.param_shape = (5,)  # x,y,z,theta, y insert length
        self._width_eps = 0.008
        self._should_reset_to_viewable=True
        if real_robot:
            self.resting_countdown = 0
        else:
            self.resting_countdown = 20  # timesteps to wait until object has settled, because IG seems to have horrible contact dynamics...
        self._should_reset_to_viewable = True

    def precondition_satisfied(self, state, parameters, check_valid_goal_state=False, env=None):
        potential_state = state.copy()
        potential_state.update_property("frame:franka:gripper/width", self._open_width)
        if not super().state_precondition_satisfied(state):
            return False
        return super().precondition_satisfied(state, parameters, check_valid_goal_state=False, env=env)
    def check_termination_condition(self, internal_state, parameters, t, controller=None, env_idx=None):
        super_terminated = super().check_termination_condition(internal_state, parameters, t, controller=controller,
                                                               env_idx=env_idx)
        if not super_terminated:
            return False
        if self._super_termination_t is None:
            self._super_termination_t = t
            print("Set countdown")
        return t - self._super_termination_t >= self.resting_countdown

    def make_controllers(self, initial_states, parameters, T_plan_max, t, dt, real_robot=False):
        controllers = []
        info_plans = []
        for env_idx, initial_state in enumerate(initial_states):
            internal_state = self.pillar_state_to_internal_state(initial_state)
            rod_width = initial_state.get_values_as_vec(["constants/rod_dims"])[0]
            curr_pose, goal_pose = seven_dim_internal_state_to_pose(internal_state, parameters, env_idx)
            y_insertion_length = parameters[0][-1]
            controller = LiftAndInsertController(closed_width=rod_width-self._width_eps, dt=dt, real_robot = real_robot, y_insertion_length=y_insertion_length)  # TODO lagrassa assumes rods all same width
            info_plan = controller.plan(curr_pose, goal_pose, self._z_height, curr_width = internal_state[-1])

            controllers.append(controller)
            info_plans.append(info_plan)
        return controllers, info_plans

    def apply_action(self, env, env_idx, action):
        transform = gymapi.Transform(
            p=np_to_vec3(action[:3]),
            r = np_quat_to_quat(quaternion.from_float_array(action[3:7]))
        )
        env.set_attractor(transform, env_idx)#, impedances = action[7:13])
        env.set_gripper_width_target(action[-1], env_idx)


class Release(Skill):
    """
    Just releases the gripper. Not much to learn here
    """

    def __init__(self):
        super().__init__()
        self._default_open_width = 0.04
        self._default_closed_width = 0.01
        self._gripper_tol = 0.005
        self._check_grasped_tol_z = 0.01
        self._horizon = 150

    def _gen_random_parameters(self, env, state):
        # there are no parameters for this skill
        yield []

    def _gen_object_centric_parameters(self, env, state):
        # there are no parameters for this skill
        yield []

    def _gen_relation_centric_parameters(self, env, state):
        # there are no parameters for this skill
        yield []

    def state_precondition_satisfied(self, state):
        obj_in_gripper, smallest_obj_width, _ = is_obj_in_gripper(state, self._check_grasped_tol_z, self._gripper_tol)
        if not obj_in_gripper:
            logger.debug(f"Skill: {self.__class__.__name__} no object in hand")
            return False
        else:
            return True

    def precondition_satisfied(self, state, parameters, check_valid_goal_state=False):
        """
        Args:
            state: state
            parameters: []

        Returns: whether an object is between the grippers

        """
        if not self.state_precondition_satisfied(state):
            logger.debug(f"Skill: {self.__class__.__name__} State precondition check failed.")
            return False
        return True

    def make_controllers(self, initial_states, parameters, T_plan_max, t, dt):
        info_plans = []
        controllers = []
        for _, _ in enumerate(initial_states):
            controller = ReleaseController(open_width=self._default_open_width, closed_width=self._default_closed_width,
                                           horizon=self._horizon)
            controller.plan()
            controllers.append(controller)
            info_plans.append({})
        return controllers, info_plans

    def apply_action(self, env, env_idx, action):
        env.set_gripper_width_target(action[0], env_idx)

    def pillar_state_to_internal_state(self, state):
        """

        Args:
            state:

        Returns: [gripper_width]

        """
        return np.array(state.get_values_as_vec(["frame:franka:gripper/width"]))

    def check_termination_condition(self, internal_state, parameters, t, controller=None, env_idx=None):
        return np.abs(internal_state - self._default_open_width) < self._gripper_tol

    def unpack_parameters(self, parameters):
        return parameters

class OpenDrawer(FreeSpaceMoveFranka):  # can instead inherit from LQR

    def __init__(self, *args, real_robot=True, **kwargs):
        super().__init__(*args, real_robot=True, **kwargs)
        self._should_reset_to_viewable = 0 #True
        self.param_shape = (5,) #amount the drawer should be opened + the start pose of the opening action
        self._z_height = 0.2 #needs to be high enough to avoid the drawer
        self._check_grasped_tol_z = 0.02
        self._gripper_tol = 0.01

    # self._model = FreeSpaceModel()
    def pillar_state_to_internal_state(self, state):
        super_state = super().pillar_state_to_internal_state(state)
        drawer_front_pose = get_pose_pillar_state(state, "drawer")[:3]
        return np.hstack([super_state, drawer_front_pose])

    def unpack_parameters(self, parameters):
        goal_pos = parameters[:3]
        goal_yaw = parameters[3]
        return np.array(goal_pos), goal_yaw

    def make_controllers(self, initial_states, parameters, T_plan_max, t, dt, real_robot=False):
        info_plans = []
        controllers = []
        for env_idx, initial_state in enumerate(initial_states):
            internal_state = self.pillar_state_to_internal_state(initial_state)
            curr_pose, goal_pose = seven_dim_internal_state_to_pose(internal_state, parameters, env_idx)
            end_y = parameters[env_idx][-1]
            controller = OpenDrawerController(dt=dt, real_robot=real_robot)  # TODO(lagrassa): assumes rods all same width
            info_plan = controller.plan(curr_pose, goal_pose, self._z_height, end_y)

            controllers.append(controller)
            info_plans.append(info_plan)
        return controllers, info_plans

    def apply_action(self, env, env_idx, action):
        super().apply_action(env, env_idx, action)

    def check_termination_condition(self, internal_state, parameters, t, controller=None, env_idx=None):
        pos_complete = super().check_termination_condition(internal_state, parameters, t, controller, env_idx)
        drawer_pos = internal_state[-3:]
        if not pos_complete:
            return False # no point checking rest
        if np.abs(drawer_pos[1]) > parameters[-1]: #assumes open is in the negative y direction
            return False
        return True


    def _gen_object_centric_parameters(self, env, state):
        #find the np.pi position for the franka, make it some offset in -y from the front. The open amount is a random number between 0.03 and 0.2
        theta = np.pi/2
        drawer_pose = get_pose_pillar_state(state, "drawer")
        x = drawer_pose[0] + 0.01
        z = drawer_pose[2] + 0.03 #formerly z but something is going on w/ joint limits??
        start_y = drawer_pose[1] + 0.02
        while True:
            end_y = drawer_pose[1] - np.random.uniform(low=0.14, high = 0.17) #originally 12 to 13
            parameters = np.array([x, start_y, z, theta, end_y])  # TODO do the other way too
            yield parameters

    def precondition_satisfied(self, state, parameters, check_valid_goal_state=False, env=None):
        if not super().state_precondition_satisfied(state):
            return False
        obj_in_gripper, smallest_obj_width, object_name = is_obj_in_gripper(state, self._check_grasped_tol_z, self._gripper_tol)
        if obj_in_gripper:
            logger.debug(f"Skill: {self.__class__.__name__} already has object in hand")
            return False

        if get_pose_pillar_state(state, "drawer")[1] < 0.08:
            #print("Drawer already too far open", get_pose_pillar_state(state, "drawer")[1])
            return False
        return True
