import numpy as np
import copy
from itertools import combinations
import logging

import quaternion

from isaacgym_utils.math_utils import np_to_quat, transform_to_RigidTransform, np_to_transform
from .base_task import BaseTask
from ..skills.franka_skills import LiftAndInsert, Pick
from ..utils import get_pose_pillar_state, states_similar_within_tol, yaw_from_np_quat, yaw_from_quat, point_in_box, \
    is_obj_in_gripper, angle_axis_between_quats, get_rod_grasps_transforms, get_object_names_in_pillar_state, \
    pillar_state_obj_to_transform, min_distance_between_angles
from ..skills import FreeSpaceMoveLQRFranka, FreeSpaceMoveFranka, LQRWaypointsXYZYawFranka, LiftAndPlace, LiftAndDrop

from ..utils.utils import pretty_print_state_with_params, pretty_print_array, min_distance_between_angles
try:
    from ..utils.ar_perception import ObjectDetector
except ImportError:
    print("Not able to import ObjectDetector")
from autolab_core import YamlConfig

logger = logging.getLogger(__name__)


class MoveGripperToPose(BaseTask):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._pos_same_tol = cfg['goal']['position_tol']
        self._position_tol = self._pos_same_tol
        self._position_tol_frac = 0.8
        self._yaw_same_tol = cfg['goal']['yaw_tol']  # 0.05
        self._yaw_tol = self._yaw_same_tol
        self._goal_xyz_yaw = list(cfg['goal']['goal_xyz_yaw'])
        self._goal_xyz_yaw[-1] = np.deg2rad(self._goal_xyz_yaw[-1]) #account for upside down
        if cfg['goal']['randomize']:
            self._goal_xyz_yaw = self._get_random_goal(cfg)

        def free_space_xyz_yaw_gen(env, state):
            center = self._goal_xyz_yaw
            while True:
                random_distance = self._position_tol_frac * np.random.random() * self._position_tol
                random_dir = np.random.uniform(low=0, high=2 * np.pi)
                random_yaw = np.random.uniform(low=center[-1]-self._yaw_tol, high=center[-1]+self._yaw_tol)
                yield np.array([center[0] + random_distance * np.cos(random_dir),
                                center[1] + random_distance * np.sin(random_dir),
                                center[2],
                                -random_yaw]) #same z, why not, this is already better than just returning xyzyaw

        self._skill_specific_param_generators[FreeSpaceMoveLQRFranka.__name__] = free_space_xyz_yaw_gen
        self._skill_specific_param_generators[FreeSpaceMoveFranka.__name__] = free_space_xyz_yaw_gen
    
    @property
    def goal_pos(self):
        return self._goal_xyz_yaw
    
    def resample_goal(self, env=None):
        old_goal = np.copy(self._goal_xyz_yaw)
        self._goal_xyz_yaw = self._get_random_goal(self._cfg)
        return old_goal, self._goal_xyz_yaw
    
    def resample_goal_from_range(self, low, high, env=None):
        old_goal = np.copy(self._goal_xyz_yaw)
        self._goal_xyz_yaw = np.random.uniform(low=low, high=high)
        return old_goal, self._goal_xyz_yaw

    def _get_random_goal(self, cfg):
        goal_xyz_yaw = np.random.uniform(low=cfg["goal"]["goal_pose_ranges"]["low"][:3],
                                         high=cfg["goal"]["goal_pose_ranges"]["high"][:3])
        random_yaw = np.random.uniform(low=np.deg2rad(cfg["goal"]["goal_pose_ranges"]["low"][-1]),
                                        high=np.deg2rad(cfg["goal"]["goal_pose_ranges"]["high"][-1]))
        return np.hstack([goal_xyz_yaw, random_yaw])

    def pillar_state_to_internal_state(self, pillar_state):
        # x,y,z, yaw
        ee_pose = get_pose_pillar_state(pillar_state, "franka:ee")
        return np.array(ee_pose)

    def is_goal_state(self, pillar_state):
        franka_ee_pose = self.pillar_state_to_internal_state(pillar_state)
        yaw_close = abs(yaw_from_np_quat(franka_ee_pose[3:]) - self._goal_xyz_yaw[3]) < self._yaw_tol
        pos_close = np.linalg.norm(franka_ee_pose[:3] - self._goal_xyz_yaw[:3]) < self._position_tol
        return pos_close and yaw_close

    def states_similar(self, pillar_state_1, pillar_state_2):
        return states_similar_within_tol(pillar_state_1, pillar_state_2, self._pos_same_tol, self._yaw_same_tol)

    def distance_to_goal_state(self, pillar_state, use_angle = False):
        franka_ee_pose = self.pillar_state_to_internal_state(pillar_state)
        position_distance = np.linalg.norm(franka_ee_pose[:3] - self._goal_xyz_yaw[:3])
        if use_angle:
            angle_distance = yaw_from_np_quat(franka_ee_pose[3:7]) - self._goal_xyz_yaw[-1]
            return position_distance + angle_distance
        return position_distance

    def is_valid_state(self, pillar_state, skills):
        return True

    def evaluate(self, pillar_state):
        return self.distance_to_goal_state(pillar_state)

    def pretty_print_goal_params(self):
        return pretty_print_state_with_params(self._goal_xyz_yaw[:3], self._goal_xyz_yaw[-1])
    
    def pretty_print_with_reference_to_pillar_state(self, pillar_state):
        franka_ee_pose = self.pillar_state_to_internal_state(pillar_state)
        franka_pos = franka_ee_pose[:3]
        yaw = yaw_from_np_quat(franka_ee_pose[3:])
        goal_str = self.pretty_print_goal_params()
        curr_state_str = pretty_print_state_with_params(franka_pos, yaw)
        return f'  Goal: {goal_str}\n' \
               f' State: {curr_state_str}\n'



class PushRodsInBoxFranka(BaseTask):

    def __init__(self, cfg, real_robot=False):
        super().__init__(cfg)
        self._pos_same_tol = cfg['goal']['position_tol']
        self._yaw_same_tol = cfg['yaw_same_tol']
        self._is_real = real_robot
        if self._is_real:
            goal_detect_cfg =  YamlConfig("/home/lagrassa/git/plan-abstractions/data/calibration/shoebox_detect.yaml") 
            #self.goal_detector = ObjectDetector(goal_detect_cfg) hardcode for now
        self._goal_dims = np.array(list(cfg['goal']['dims'].values()))
        self._setup_callbacks.append(lambda env, scene, env_idx: self.add_goal_box_to_env_cb(env, scene, env_idx))
        self._rods_to_push = list(cfg["goal"]["target_rods"])
        self._random_goal_dist_frac = 0.1
        self.num_rods = len(self._rods_to_push)
        if not self._is_real:
            if cfg['goal']['randomize']:
                self._goal_pose = np.random.uniform(low=cfg["goal"]["goal_pose_ranges"]["low"],
                                                    high=cfg["goal"]["goal_pose_ranges"]["high"])
                self._rods_to_push = np.zeros((self.num_rods,))  # np.random.randint(0, 2, self.num_rods)
                self._rods_to_push[np.random.randint(0, 2)] = 1
                #self._rods_to_push = np.random.randint(0, 2, self.num_rods)

            else:
                self._goal_pose = np.array(cfg['goal']['pose'] + [0, ])
        else:
            self._goal_pose = self.detect_goal_pose()
        

        def free_space_xyz_yaw_gen(env, state):
            center = self._goal_pose
            while True:
                random_distance = np.random.random() * self._random_goal_dist_frac*min(self._goal_dims[:2])
                random_dir = np.random.uniform(low=0, high=2 * np.pi)
                random_yaw = np.random.uniform(low=center[-1]-2*self._yaw_same_tol, high=center[-1]+2*self._yaw_same_tol)
                yield np.array([center[0] + random_distance * np.cos(random_dir),
                                center[1] + random_distance * np.sin(random_dir),
                                env.grasp_height,
                                -random_yaw]) #same z, why not, this is already better than just returning xyzyaw

        def drop_gen(env, state):
            center = self._goal_pose
            while True:
                random_distance = np.random.random() * self._random_goal_dist_frac*min(self._goal_dims[:2])
                random_dir = np.random.uniform(low=0, high=2 * np.pi)
                random_yaw = np.random.uniform(low=center[-1]-2*self._yaw_same_tol, high=center[-1]+2*self._yaw_same_tol)
                yield np.array([center[0] + random_distance * np.cos(random_dir),
                                center[1] + random_distance * np.sin(random_dir),
                                env.obstacle_free_height,#env.obstacle_free_height,
                                -random_yaw]) #same z, why not, this is already better than just returning xyzyaw


        def lqr_waypoint_xyyaw_gen(env, state):
            curr_quat = np_to_quat(state.get_values_as_vec(['frame:franka:ee:pose/quaternion']), format="wxyz")
            curr_yaw = yaw_from_quat(curr_quat)
            num_rods_to_push = np.sum(self._rods_to_push)
            rods_to_push_org = [0 for _ in range(len(self._rods_to_push))]
            rod_idxs_to_push = [i for i in range(len(self._rods_to_push)) if self._rods_to_push[i] == 1]
            while True:
                pos = self.get_goal_positions_around_goal_region()
                x, y = pos
                parameters = np.array([x, y, curr_yaw] + self._rods_to_push.tolist())
                yield parameters

        self._skill_specific_param_generators[FreeSpaceMoveLQRFranka.__name__] = free_space_xyz_yaw_gen
        self._skill_specific_param_generators[FreeSpaceMoveFranka.__name__] = free_space_xyz_yaw_gen
        self._skill_specific_param_generators[LiftAndPlace.__name__] = free_space_xyz_yaw_gen
        self._skill_specific_param_generators[LiftAndDrop.__name__] = drop_gen
        self._skill_specific_param_generators[LQRWaypointsXYZYawFranka.__name__] = lqr_waypoint_xyyaw_gen

    def detect_goal_pose(self):
        #detections = self.goal_detector.detect() #hardcoded to have fewer potential problems
        return np.array([0.43119273, -0.12742051])

    def resample_goal(self, env=None):
        return self.resample_goal_from_range(self._cfg["goal"]["goal_pose_ranges"]["low"],
                                             self._cfg["goal"]["goal_pose_ranges"]["high"],
                                             env=env)

    def resample_goal_from_range(self, low, high, env=None):
        old_goal_pose = self._goal_pose
        if self._is_real:
            input("Move goal box")
            new_goal_pose = self.detect_goal_pose()
            self._goal_pose = new_goal_pose
        else:
            new_goal_pose = np.random.uniform(low=low, high=high)
        self._goal_pose = new_goal_pose
        #self._rods_to_push = np.random.randint(0, 2, self.num_rods)
        self._rods_to_push = np.zeros((self.num_rods,)) #np.random.randint(0, 2, self.num_rods)
        self._rods_to_push[np.random.randint(0,2)] = 1
        #self._rods_to_push[0] = 1

        # Update the visual rendering of the goal in the env.
        if env is not None:
            env.reset_visual_box(self._goal_pose, self._goal_dims)
        return old_goal_pose, new_goal_pose

    @property
    def goal_pos(self):
        return self._goal_pose

    @property
    def goal_dims(self):
        return self._goal_dims

    def add_goal_box_to_env_cb(self, env, scene, env_idx):
        """
        Callback that adds a visual box to the environment.
        useful for debugging
        """
        env.add_visual_box_cb(self.goal_pos, self.goal_dims)

    def pillar_state_to_internal_state(self, pillar_state):
        ee_pos = get_pose_pillar_state(pillar_state, "franka:ee")[:3]
        rod0_pos = np.array(
            pillar_state.get_values_as_vec(["frame:rod0:pose/position"])
        )
        rod1_pos = np.array(
            pillar_state.get_values_as_vec(["frame:rod1:pose/position"])
        )
        return np.array([ee_pos, rod0_pos, rod1_pos])

    def is_goal_state(self, pillar_state):
        _, rod0_pos, rod1_pos = self.pillar_state_to_internal_state(pillar_state)
        rod_poses = [rod0_pos, rod1_pos]
        for rod_idx, to_push in enumerate(self._rods_to_push):
            if not to_push:
                continue
            rod_pos = rod_poses[rod_idx]
            # Rod should be inside the box?
            if not point_in_box(rod_pos[:2], self.goal_pos, self.goal_dims[:2]):
                return False
            # Rod should not be in air.
            if rod_pos[2] > 0.09:
                return False
            # TODO(Mohit): Maybe adding orientation information could be useful for planning.
        return True

    def states_similar(self, pillar_state_1, pillar_state_2):
        return states_similar_within_tol(pillar_state_1, pillar_state_2, self._pos_same_tol, self._yaw_same_tol)

    def distance_to_goal_state(self, pillar_state):
        ee_pos, rod0_pos, rod1_pos = self.pillar_state_to_internal_state(pillar_state)
        rod_poses = [rod0_pos, rod1_pos]
        # desired pos in 3D, 0.004 since rods have a height of 0.01 so half of it.
        # Although, z-variable might not be required, since all skills should end up on ground only.
        desired_pos = np.r_[self._goal_pose[:2], np.array([0.008])] #0.008 was okay hard to close
        # We don't really care about EE being close to goal.
        # total_dist_to_goal = (np.linalg.norm(ee_pos[:2]-desired_pos))**2
        total_dist_to_goal = 0.0
        for rod_idx, to_push in enumerate(self._rods_to_push):
            if not to_push:
                continue
            rod_pos = rod_poses[rod_idx]
            total_dist_to_goal += np.linalg.norm(rod_pos - desired_pos)
        return total_dist_to_goal

    def get_goal_positions_around_goal_region(self):
        return np.random.uniform(low=self.goal_pos[:2] - self.goal_dims[:2] / 2,
                                 high=self.goal_pos[:2] + self.goal_dims[:2] / 2)

    def is_valid_state(self, pillar_state, skills):
        # TODO
        # Check for the preconditions of all the skills
        return True

    def evaluate_admissible(self, pillar_state):
        return self.distance_to_goal_state(pillar_state)

    def evaluate_inadmissible(self, pillar_state):
        return self.distance_to_goal_state(pillar_state)

    def evaluate(self, pillar_state, heur_id=0):
        if heur_id == 0:
            return self.evaluate_admissible(pillar_state)
        else:
            return self.evaluate_inadmissible(pillar_state)

    def pretty_print_goal_params(self):
        return pretty_print_array(self.goal_pos, prefix='Goal position')
    
    def pretty_print_with_reference_to_pillar_state(self, pillar_state):
        goal_str = self.pretty_print_goal_params()
        ee_pos, rod0_pos, rod1_pos = self.pillar_state_to_internal_state(pillar_state)
        rod0_str = pretty_print_array(ee_pos, 'EE:')
        rod0_str = pretty_print_array(rod0_pos, 'Rod 0:')
        rod1_str = pretty_print_array(rod1_pos, 'Rod 1:')

        return f'     Goal: {goal_str}\n' \
               f'       EE: {ee_pos}\n' \
               f' Rod 0: {rod0_str}\n' \
               f' Rod 1: {rod1_str}\n'
    


class PickRod(BaseTask):
    def __init__(self, cfg, real_robot=False):
        super().__init__(cfg)
        self._pos_same_tol = cfg['position_same_tol']
        self._yaw_same_tol = cfg['yaw_same_tol']
        self._rod_to_pick = np.argmax(cfg["goal"]["target_rod"])
        self._num_rods = len(cfg["goal"]["target_rod"])
        self._gripper_err = 5e-3
        self._max_height_diff = 0.04


    def pillar_state_to_internal_state(self, pillar_state):
        ee_pos = get_pose_pillar_state(pillar_state, "franka:ee")[:3]
        init_state_kwargs = {}
        rod0_pos = np.array(
            pillar_state.get_values_as_vec(["frame:rod0:pose/position"])
        )[:2]
        rod1_pos = np.array(
            pillar_state.get_values_as_vec(["frame:rod1:pose/position"])
        )[:2]
        return np.array([ee_pos, rod0_pos, rod1_pos])

    def is_goal_state(self, pillar_state):
        obj_in_gripper, width, obj_name = is_obj_in_gripper(pillar_state, self._max_height_diff, self._gripper_err)
        if not obj_in_gripper:
            return False
        return str(self._rod_to_pick) in obj_name

    def states_similar(self, pillar_state_1, pillar_state_2):
        return states_similar_within_tol(pillar_state_1, pillar_state_2, self._pos_same_tol, self._yaw_same_tol)

    def resample_goal(self, env=None):
        old_goal = np.copy(self._rod_to_pick)
        self._rod_to_pick = np.random.randint(self._num_rods)
        return old_goal, None

    def distance_to_goal_state(self, pillar_state):
        ee_pos, rod0_pos, rod1_pos = self.pillar_state_to_internal_state(pillar_state)
        rod_poses = [rod0_pos, rod1_pos]
        rod_xy = rod_poses[self._rod_to_pick]
        height = pillar_state.get_values_as_vec(["constants/rod_dims"])[2]
        rod_pose = np.hstack([rod_xy, height])
        distance =  np.linalg.norm(ee_pos - rod_pose)
        return distance

    def is_valid_state(self, pillar_state, skills):
        # TODO
        # Check for the preconditions of all the skills
        return True

    def evaluate(self, pillar_state):
        return self.distance_to_goal_state(pillar_state)

    def pretty_print_goal_params(self):
        return f"Rod to pick: {self._rod_to_pick}"

    def pretty_print_with_reference_to_pillar_state(self, pillar_state):
        franka_ee_pose = self.pillar_state_to_internal_state(pillar_state)
        franka_pos = franka_ee_pose[:3]
        yaw = yaw_from_np_quat(franka_ee_pose[3:])
        goal_str = self.pretty_print_goal_params()
        curr_state_str = pretty_print_state_with_params(franka_pos, yaw)
        return f'  Goal: {goal_str}\n' \
               f' State: {curr_state_str}\n'

class PickRodsInBoxFranka(PushRodsInBoxFranka):
    def __init__(self, cfg, real_robot=False):
        super().__init__(cfg)
        self._setup_callbacks = [lambda env, scene, env_idx: self.add_real_goal_box_to_env_cb(env, scene, env_idx)]
        self._is_real = real_robot
        self._grasp_offsets = [0.01]
        if self._is_real:
            goal_detect_cfg =  YamlConfig("/home/lagrassa/git/plan-abstractions/cfg/perception/shoebox_detect.yaml")
            #self.goal_detector = ObjectDetector(goal_detect_cfg) hardcode for now
        def pick_gen(env, state):
            dims = state.get_values_as_vec(["constants/rod_dims"])
            rod_grasp_transforms = get_rod_grasps_transforms(dims[1], dims[2], grasp_offset=self._grasp_offsets)
            rod_names = [name for name in get_object_names_in_pillar_state(state) if env.object_name in name]
            rod_names.sort() #I foolishly assumed these would be ordered....
            rod_transforms = [pillar_state_obj_to_transform(state, obj_name) for obj_name in rod_names]
            min_theta = -2
            max_theta = 3.7
            symmetries_angles = [-np.pi, 0, np.pi]
            gripper_curr_quat = quaternion.from_float_array(
                state.get_values_as_vec(['frame:franka:ee:pose/quaternion']))
            while True:
                for grasp_transform in rod_grasp_transforms:
                    for i, rod_transform in enumerate(rod_transforms):
                        if not self._rods_to_push[i]:
                            continue
                        goal_transform = rod_transform * grasp_transform
                        x = goal_transform.p.x
                        y = goal_transform.p.y
                        if self._is_real:
                            z = 0.014 ##.012 0.008 works on real robot, 0.004 in sim  # goal_transform.p.z - 8e-3
                            z = rod_transform.p.z
                        else:
                            z = rod_transform.p.z
                        theta = yaw_from_quat(goal_transform.r)
                        curr_yaw = yaw_from_quat(gripper_curr_quat)
                        dist_from_currents = []
                        for symmetry_yaw in symmetries_angles:
                            angle_dist = np.abs((theta + symmetry_yaw) - curr_yaw)  # wraparound will be a problem
                            #angle_dist = min_distance_between_angles(theta+symmetry_yaw, curr_yaw)
                            dist_from_currents.append(angle_dist)
                        theta = theta + symmetries_angles[np.argmin(dist_from_currents)]
                        # if theta < min_theta:
                        #     theta += np.pi
                        # if theta > max_theta:
                        #     theta -= np.pi
                        parameters = np.array([x, y, z, theta])  # TODO do the other way too
                        yield parameters
        self._skill_specific_param_generators[Pick.__name__] = pick_gen

    def add_real_goal_box_to_env_cb(self, env, scene, env_idx):
        """
        Callback that adds a visual box to the environment.
        useful for debugging
        """
        env.add_real_box_cb(self.goal_pos, self.goal_dims)

    def resample_goal_from_range(self, low, high, env=None):
        old_goal_pose = self._goal_pose
        if self._is_real:
            input("Move goal box")
            new_goal_pose = self.detect_goal_pose()
            self._goal_pose = new_goal_pose
        else:
            new_goal_pose = np.random.uniform(low=low, high=high)
        self._goal_pose = new_goal_pose
        self._rods_to_push = np.zeros((self.num_rods,))
        self._rods_to_push[np.random.randint(0,self.num_rods)] = 1
        print("rods to push", self._rods_to_push)

        # Update the visual rendering of the goal in the env.
        if env is not None:
            env.reset_real_box(self._goal_pose)
        return old_goal_pose, new_goal_pose



class RodInDrawer(PickRodsInBoxFranka):
    def __init__(self, cfg, real_robot=False):
        super().__init__(cfg)
        self._setup_callbacks = [] # already in env self.add_real_drawer_to_env_cb]
        self._drawer_length = .38
        self._random_goal_dist_frac = 0.001
        def drop_gen(env, state):
            center = self._get_center_from_pillar_state(pillar_state=state)
            center[1] -= 0.06 # shift it a bit more
            while True:
                random_distance = np.random.random() * self._random_goal_dist_frac*min(self._goal_dims[:2])
                random_dir = np.random.uniform(low=0, high=2 * np.pi)
                yield np.array([center[0] + random_distance * np.cos(random_dir),
                                center[1] + random_distance * np.sin(random_dir),
                                env.obstacle_free_height,#env.obstacle_free_height,
                                -np.pi/2]) #same z, why not, this is already better than just returning xyzyaw
        self._skill_specific_param_generators[LiftAndDrop.__name__] = drop_gen

    def add_real_drawer_to_env_cb(self, env, scene, env_idx):
        drawer_pose = [0.4, 0.34, 0.005] #This gets updated by the env
        env.add_real_drawer_cb(env, scene, env_idx, drawer_pose)

    #init goal pose is somewhere in the bottom drawer
    def _get_center_from_pillar_state(self, pillar_state):
        front_pose = get_pose_pillar_state(pillar_state, "drawer")
        center = np.array([front_pose[0], front_pose[1] +  self._drawer_length/2])
        return center

    def states_similar(self, pillar_state_1, pillar_state_2):
        if not super().states_similar(pillar_state_1, pillar_state_2):
            return False
        drawer_pose_1 = get_pose_pillar_state(pillar_state_1, "drawer")
        drawer_pose_2 = get_pose_pillar_state(pillar_state_2, "drawer")
        if np.linalg.norm(np.array(drawer_pose_1)-np.array(drawer_pose_2)) > 10*self._pos_same_tol: #want this to be more tolerant:
            return False

        return True

    def is_goal_state(self, pillar_state):
        _, rod0_pos, rod1_pos = self.pillar_state_to_internal_state(pillar_state)
        rod_poses = [rod0_pos, rod1_pos]
        lower_z = 0.03
        high_z = 0.09
        center = self._get_center_from_pillar_state(pillar_state)
        for rod_idx, to_push in enumerate(self._rods_to_push):
            if not to_push:
                continue
            rod_pos = rod_poses[rod_idx]
            # Rod should be inside the box?
            if not point_in_box(rod_pos[:2], center, self.goal_dims[:2]):
                return False
            # Rod should not be in air.
            if rod_pos[2] > high_z or rod_pos[2] < lower_z:
                return False
        return True


    def distance_to_goal_state(self, pillar_state):
        ee_pos, rod0_pos, rod1_pos = self.pillar_state_to_internal_state(pillar_state)
        rod_poses = [rod0_pos, rod1_pos]
        # desired pos in 3D, 0.004 since rods have a height of 0.01 so half of it.
        # Although, z-variable might not be required, since all skills should end up on ground only.
        center = self._get_center_from_pillar_state(pillar_state)
        # We don't really care about EE being close to goal.
        # total_dist_to_goal = (np.linalg.norm(ee_pos[:2]-desired_pos))**2
        total_dist_to_goal = 0.0
        for rod_idx, to_push in enumerate(self._rods_to_push):
            if not to_push:
                continue
            rod_pos = rod_poses[rod_idx]
            total_dist_to_goal += np.linalg.norm(rod_pos[:2] - center[:2])
        desired_drawer_center = 0
        drawer_dist = np.abs(center[1]-desired_drawer_center)
        return total_dist_to_goal + drawer_dist


class DrawerOpen(RodInDrawer):
    def is_goal_state(self, pillar_state):
        drawer_pose = get_pose_pillar_state(pillar_state, "drawer")
        print(drawer_pose[1])
        open_thresh = 0.08
        return drawer_pose[1] < open_thresh

class RodInSlot(PushRodsInBoxFranka):
    def __init__(self, cfg):
        self.initialized_with_real_robot = False;
        super().__init__(cfg)
        self._setup_callbacks = [] #[lambda env, scene, env_idx: self.add_real_slot_to_env_cb(env, scene, env_idx)]
        self._insertion_buffer_inital = 0.10
        self._insertion_buffer_after = 0.05
        self.initialized_with_real_robot = False

        def insert_gen(env, state):
            center = self._goal_pose
            ee_pose = get_pose_pillar_state(state, "franka:ee")
            tol_z = 0.03
            gripper_err = 0.01
            z_hole = 0.123 #Knwn, I dont want ot add yet another camera
            obj_in_gripper, smallest_obj_width, obj_name = is_obj_in_gripper(state, tol_z,gripper_err)
            rod_pose = get_pose_pillar_state(state, obj_name)
            ee_rt = transform_to_RigidTransform(np_to_transform(ee_pose, format="wxyz"), from_frame="gripper", to_frame="world")
            rod_rt = transform_to_RigidTransform(np_to_transform(rod_pose, format="wxyz"), from_frame="gripper", to_frame="world")
            #take dot product of current ee_rt axis and vector from ee pose to rod pose
            T_obj_to_ee_rt = ee_rt.inverse() * rod_rt
            #axis = angle_axis_between_quats(quaternion.from_float_array(T_obj_to_ee_rt.quaternion),
            #                                     quaternion.from_float_array([1,0,0,0]))
            axis = angle_axis_between_quats(quaternion.from_float_array(T_obj_to_ee_rt.quaternion),
                                                 quaternion.from_float_array(ee_rt.quaternion))
            ee_to_obj_vector = rod_rt.translation[:3] - ee_rt.translation[:3]
            ee_rt_x_axis = ee_rt.rotation[:,0]
            dot_prod = np.dot(ee_to_obj_vector, ee_rt_x_axis)

            y_insert_length = state.get_values_as_vec([f"constants/rod_dims"])[1]/2 + np.linalg.norm(T_obj_to_ee_rt.translation)
            print("Dot prod", dot_prod)
            if dot_prod < 0:
                desired_angle = -np.pi/2
            elif dot_prod > 0:
                desired_angle = np.pi/2
            else:
                raise ValueError("Invalid grasp")
            while True:
                yield np.hstack([center[0], center[1] - y_insert_length - self._insertion_buffer_inital,z_hole, desired_angle, y_insert_length-self._insertion_buffer_after + self._insertion_buffer_inital]).flatten() #same z, why not, this is already better than just returning xyzyaw
        self._skill_specific_param_generators[LiftAndInsert.__name__] = insert_gen

    def set_detector(self):
        self.initialized_with_real_robot = True
        side_cfg = YamlConfig("/home/lagrassa/git/plan-abstractions/cfg/perception/slot_detect.yaml")
        self.slot_detector = ObjectDetector(side_cfg)

    def add_real_slot_to_env_cb(self, env, scene, env_idx):
        """
        Callback that adds a visual box to the environment.
        useful for debugging
        """
        env.add_real_slot_cb(self.goal_pos, self.goal_dims)

    def resample_goal_from_range(self, low, high, env=None):
        old_goal_pose = self._goal_pose
        if self.initialized_with_real_robot:
            input("Move slotr")
            new_goal_pose = self.detect_goal_pose()
            self._goal_pose = new_goal_pose
        else:
            new_goal_pose = np.random.uniform(low=low, high=high)
        self._goal_pose = new_goal_pose
        self._rods_to_push = np.zeros((self.num_rods,)) #np.random.randint(0, 2, self.num_rods)
        #self._rods_to_push[np.random.randint(0, 2, self.num_rods)] = 1
        self._rods_to_push[np.random.randint(0,2)] = 1
        if np.sum(self._rods_to_push) == 2 or np.sum(self._rods_to_push) == 0:
            import ipdb; ipdb.set_trace()
        print("rods to push", self._rods_to_push)

        # Update the visual rendering of the goal in the env.
        if env is not None:
            env.reset_real_slot(self._goal_pose)
        return old_goal_pose, new_goal_pose

    def detect_goal_pose(self):
        if self.initialized_with_real_robot:
            detections = self.slot_detector.detect()
            rod_num = 0
            rt = self.slot_detector.get_rod_rt_from_detections(rod_num, detections)
            return rt.translation
        else:
            print("Warning: fake pose because perception node can't be initialized")
            return np.array([0.49278037,0.31954214,0.24099996])
            #return np.array([0.49, 0.327])

    def is_goal_state(self, pillar_state):
        _, rod0_pos, rod1_pos = self.pillar_state_to_internal_state(pillar_state)
        rod_poses = [rod0_pos, rod1_pos]
        for rod_idx, to_push in enumerate(self._rods_to_push):
            if not to_push:
                continue
            rod_pos = rod_poses[rod_idx]
            if not rod_pos[0] < self._goal_pose[0] + self.goal_dims[0]/2 and rod_pos[0] > self._goal_pose[0] - self._goal_dims[0]/2:
                print("Not in x range")
                return False
            if not rod_pos[1] > (self._goal_pose[1] - 0.008):
                print(f"not through the slot, distance is {rod_pos[1] - (self._goal_pose[1]-0.008)}")
                return False
        return True

    def distance_to_goal_state(self, pillar_state, use_angle = False):
        franka_ee_pose, rod0_pos, rod1_pos = self.pillar_state_to_internal_state(pillar_state)
        rod_poses = [rod0_pos, rod1_pos]
        total_distance = 0
        for rod_idx, to_push in enumerate(self._rods_to_push):
            if not to_push:
                continue
            rod_pose = rod_poses[rod_idx]
            x_dist = (rod_pose[0] - self._goal_pose[0])**2
            y_dist = np.max(self._goal_pose[1] - rod_pose[1], 0)**2
            total_distance += np.sqrt(x_dist + y_dist)
        return total_distance
