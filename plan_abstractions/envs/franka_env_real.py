from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import copy
import rospy
import seaborn as sns
import logging
from collections import OrderedDict

from autolab_core import RigidTransform, YamlConfig
from pillar_state import State
import seaborn as sns
OFF_ROBOT = 0
if OFF_ROBOT:
    print("Not using robot. Only use this global variable for debugging")
else:
    from frankapy import FrankaArm
import quaternion
from isaacgym import gymapi
from isaacgym_utils.draw import draw_contacts, draw_transforms
from isaacgym_utils.math_utils import rpy_to_quat, RigidTransform_to_transform, np_to_transform, quat_to_np, transform_to_RigidTransform, \
    transform_to_np, vec3_to_np, quat_to_np, compute_task_space_impedance_control, np_quat_to_quat, project_to_line, \
    quat_to_rot, angle_axis_between_axes, np_to_vec3
from . import FrankaRodEnv

from .base_env import BaseEnv
from .utils import is_pose_of_object_close, get_pose_pillar_state, set_pose, get_joint_position_pillar_state, \
    get_joint_velocity_pillar_state, get_gripper_width_pillar_state, is_state_of_robot_close, make_env_with_init_states
from ..utils import yaw_from_quat, yaw_from_np_quat, get_rod_rel_goal_RigidTransforms_x_in, pillar_state_to_shapes, \
    object_name_to_asset_name, pillar_state_obj_to_transform, xyz_yaw_to_transform, r_flip_yz, \
    get_object_names_in_pillar_state, set_franka_pillar_properties_from_init_states_arr, is_obj_in_gripper, extract_xy_dims_and_height
from ..utils.ar_perception import ObjectDetector, InHandRodDetector


class RealFrankaRodEnv(FrankaRodEnv):
    franka_name = "franka"
    object_name = "rod"
    goal_name = "goal"

    @staticmethod
    def get_rod_colors(n_rods, only_ig_colors=False, only_sem_colors=False):
        colors = sns.color_palette("Paired", n_colors=2 * n_rods)
        if only_ig_colors:
            return colors[1::2]
        elif only_sem_colors:
            return colors[0::2]
        else:
            return colors

    def __init__(self, cfg, setup_callbacks=[], make_drawer=False, make_box=False):
        #FrankaRodEnv.__init__(self, cfg)
        self._cfg = cfg
        self._gripper_mass = 1
        original_cwd = Path(self._cfg['original_cwd'])
        self._pillar_states = [State()]
        self._num_rods = cfg["env_props"]["num_rods"]
        self._real_robot = True
        if OFF_ROBOT:
            rospy.init_node("franka")
            self._franka = None
        else:
            self._franka = FrankaArm()
            self.reset_to_viewable()
        self.franka_name = FrankaRodEnv.franka_name  # needs to match what's used in pillar_state
        self._body_dims = list(cfg[FrankaRodEnv.object_name]['dims'].values())
        self._finger_dims = [0.01, 0.01, 0.01]
        rod_cfg = YamlConfig("/home/lagrassa/git/plan-abstractions/cfg/perception/rod_detect.yaml")
        intel_cfg = YamlConfig("/home/lagrassa/git/plan-abstractions/cfg/perception/intel_detector.yaml")
        self.rod_detector = ObjectDetector(rod_cfg)
        side_camera = False
        if side_camera:
            side_cfg = YamlConfig("/home/lagrassa/git/plan-abstractions/cfg/perception/side_rod_detect.yaml")
            self.side_detector = ObjectDetector(side_cfg)
        if make_drawer:
            drawer_cfg = YamlConfig("/home/lagrassa/git/plan-abstractions/cfg/perception/drawer_detect.yaml")
            self.drawer_detector = ObjectDetector(drawer_cfg)
        else:
            self.drawer_detector = None

        self.inhand_detector = InHandRodDetector(intel_cfg)
        #self.shoebox_detector = ObjectDetector(rod_cfg)
        self._n_envs = 1
        self._rod_dims = cfg[FrankaRodEnv.object_name]['dims']
        self._rod_mass = cfg[FrankaRodEnv.object_name]['asset_options']['density'] * (
                self._rod_dims['sx'] * self._rod_dims['sy'] * self._rod_dims['sz'])
        self._dt = 1.3
        self._g =  -9.8
        self._scaling_speed = 1.5
        self._collision_eps = cfg["env_props"]["collision_eps"]
        self._grasp_height = 0.005  # height where franka can grasp
        self._obstacle_free_height = 0.18  # height above which there are guaranteed to be no obstacles #TODO dont hardcode
        self._asset_name_to_eps_arr = {
            "finger_left": [2 * self._collision_eps, 2 * self._collision_eps],
            "finger_right": [2 * self._collision_eps, 2 * self._collision_eps],
            "rod": [self._collision_eps, self._collision_eps]
        }
        self._conservative_asset_name_to_eps_arr = {
            "finger_left": [2 * self._collision_eps, 2 * self._collision_eps],
            "finger_right": [2 * self._collision_eps, 2 * self._collision_eps],
            "rod": [3*self._collision_eps, 3*self._collision_eps]
        }

        self._skill_params = [None]

        sx_franka = 0.01
        sx_rod = self.rod_dims['sx']
        sy_rod = self.rod_dims['sy']
        sz_rod = self.rod_dims['sz']
        th = 5e-2 #4 more reliable
        x_th = (sx_franka + sx_rod) / 2 + th
        y_th = (sx_franka + sy_rod) / 2 + th
        rod_rel_goal_RigidTransforms_x_in = get_rod_rel_goal_RigidTransforms_x_in(x_th, y_th, sy_rod)
        rod_rel_goal_RigidTransforms_x_out = [
            T * RigidTransform(rotation=np.array([
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ]), from_frame=T.from_frame, to_frame=T.from_frame)
            for T in rod_rel_goal_RigidTransforms_x_in
        ]
        self._rod_rel_goal_transforms = [
            RigidTransform_to_transform(T)
            for T in \
            rod_rel_goal_RigidTransforms_x_in + rod_rel_goal_RigidTransforms_x_out
        ]
        if not OFF_ROBOT:
            self._current_ee_poses = self._compute_ee_pose()
        for env_idx in range(self.n_envs):
            self._update_pillar_state(env_idx=env_idx, update_rod_poses=True)


    @property
    def n_envs(self):
        return self._n_envs

    @property
    def num_rods(self):
        return self._num_rods

    def reset_to_viewable(self):
        joints = np.array([ 2.55112696e-04, -6.54575045e-01,  3.46233133e-04, -2.96203170e+00,
   -2.82695993e-05,  2.30062563e+00,  7.85685771e-01])
        self._franka.goto_joints(joints)

    def set_attractor(self, transform, env_idx, impedances = None):
        des_rt = transform_to_RigidTransform(transform)
        des_rt.from_frame = "franka_tool"
        des_rt.to_frame = "world"
        min_z = 0.01
        des_rt.translation[2] = max(des_rt.translation[2], min_z)
        #print("Des rt", des_rt.translation.round(2), "actual rt", self._franka.get_pose().translation.round(2))
        #self._franka.goto_pose(des_rt, duration = self._scaling_speed * self.dt, cartesian_impedances=[2000,2000,2000,40,40,40])
        if False and impedances is not None and sum(impedances) != 0:
            if not isinstance(impedances, list):
                impedances = impedances.tolist()
                print(impedances)
            #self._franka.goto_pose(des_rt, duration = self._scaling_speed * self.dt, use_impedance=False, cartesian_impedances=impedances)
            self._franka.goto_pose(des_rt, use_impedance=False, cartesian_impedances=impedances)
        else:
            #self._franka.goto_pose(des_rt, duration=self._scaling_speed * self.dt, use_impedance=False)
            start_time = time.time()
            self._franka.goto_pose(des_rt, use_impedance=False, duration=2)
            end_time = time.time()
            #print(f"Time to do goto_pose for {des_rt.translation.round(3)} was {end_time - start_time}")
            #print(f"curr pos {self._franka.get_pose().translation.round(3)}")
            distance = np.linalg.norm(self._franka.get_pose().translation - des_rt.translation)
            if distance > 0.013:
                print("Using impedance controller to see if it will be more accurate")
                self._franka.goto_pose(des_rt)
        pose = self._franka.get_pose()
        distance = np.linalg.norm(pose.translation - des_rt.translation)
        if distance > 0.02:
            print("Distance", distance)
            import ipdb; ipdb.set_trace()
        #if np.linalg.norm(self._franka.get_pose().translation - des_rt.translation) > 0.01 or self._franka.get_pose().translation[2] < 0.04:
        #    import ipdb; ipdb.set_trace()


    @property
    def rod_dims(self):
        return self._rod_dims

    @property
    def rod_mass(self):
        return self._rod_mass

    @property
    def gripper_mass(self):
        return self._gripper_mass

    @property
    def dt(self):
        return self._dt

    @property
    def g(self):
        return self._g

    @property
    def grasp_height(self):
        return self._grasp_height

    @property
    def obstacle_free_height(self):
        return self._obstacle_free_height

    def get_object_names(self):
        return [f"{FrankaRodEnv.franka_name}:ee"] + self._rod_names

    def step(self):
        for env_idx in range(self.n_envs):
            self._update_pillar_state(env_idx)
        return self._compute_costs()

    def generate_init_states(self, cfg, min_samples=10, max_samples=1000, init_state_flag=None, 
                             choose_from_multiple_rod_configs=False, return_init_state_info=False):
        input("OK when env is setup")
        potential_pillar_state = self._pillar_states[0]
        yield potential_pillar_state


    def get_rod_transforms(self):
        detections = self.rod_detector.detect()
        ids = [0,1]
        rod_transforms = []
        for id_idx in ids:
            rod_rt = self.rod_detector.get_rod_rt_from_detections(id_idx, detections)

            #Hack, turned off for insert in slot because nothing interesting is in hand
            if rod_rt is None:
                side_detections = self.side_detector.detect()
                rod_rt = self.side_detector.get_rod_rt_from_detections(id_idx, side_detections)
                #inhand_detections = self.inhand_detector.detect()
                #if len(inhand_detections) > 0:
                #    print("Detected object in hand, though was not able to detect it using the overhead")
                #    rod_tf = self._rod_poses[id_idx]
            rod_tf = RigidTransform_to_transform(rod_rt)
            rod_transforms.append(rod_tf)
        return rod_transforms

    def get_franka_ee_transform(self):
        ee_pose = self._franka.get_pose()
        return RigidTransform_to_transform(ee_pose)

    def get_finger_transforms(self):
        #Might need to deal with offset
        finger_poses = [self._franka.get_pose()]*2#self._franka.get_finger_poses()
        return [RigidTransform_to_transform(pose) for pose in finger_poses]


    def _update_pillar_state(self, env_idx, update_rod_poses=False, update_franka_pose=True):
        pillar_state = self._pillar_states[env_idx]

        # Update rods
        pillar_state.update_property(f"constants/{FrankaRodEnv.object_name}_mass", self.rod_mass)
        pillar_state.update_property("constants/dt", self.dt)
        if update_rod_poses:
            #also updates drawer....
            if self.drawer_detector is not None:
                detections = self.drawer_detector.detect(3)
                drawer_rt = self.drawer_detector.get_rod_rt_from_detections(0,detections)
                if drawer_rt is None:
                    pass
                pillar_state.update_property(f"frame:drawer:pose/position", drawer_rt.translation)
                pillar_state.update_property(f"frame:drawer:pose/quaternion", drawer_rt.quaternion)
            print("UPdating rod poses")
            self._rod_poses = self.get_rod_transforms()
            for i in range(self.num_rods):
                object_name = f"rod{i}"
                object_pos = transform_to_np(self._rod_poses[i], format="wxyz")
                pillar_state.update_property(f"frame:{object_name}:pose/position", object_pos[:3])
                pillar_state.update_property(f"frame:{object_name}:pose/quaternion", object_pos[3:])
                pillar_state.update_property(f"frame:{object_name}:pose/linear_velocity", np.zeros((3,)))
                pillar_state.update_property(f"frame:{object_name}:pose/angular_velocity", np.zeros((3,)))

        # Update Franka
        # Joints
        if OFF_ROBOT:
            pillar_state.update_property(
                f"frame:{self.franka_name}:joint/position",
                [0]*7,
            )
            pillar_state.update_property(
                f"frame:{self.franka_name}:ee:pose/position",
                [0,0,0.03]
            )
            pillar_state.update_property(
                f"frame:{self.franka_name}:ee:pose/quaternion",
                [1,0,0,0]
            )
        else:
            if update_franka_pose:
                pillar_state.update_property(
                    f"frame:{self.franka_name}:joint/position",
                    self._franka.get_joints()[:7]
                )
                qdot = self._franka.get_joint_velocities()
                pillar_state.update_property(
                    f"frame:{self.franka_name}:joint/velocity",
                    qdot
                )

                # Gripper/fingers
                pillar_state.update_property(
                    f"frame:{self.franka_name}:gripper/width",
                    self._franka.get_gripper_width()/2,
                )
                finger_transforms = self.get_finger_transforms()
                finger_transform_np_left = transform_to_np(finger_transforms[0], format='wxyz')
                finger_transform_np_right = transform_to_np(finger_transforms[1], format='wxyz')
                pillar_state.update_property(
                    f"frame:{self.franka_name}:finger_left:pose/position",
                    finger_transform_np_left[:3]
                )
                pillar_state.update_property(
                    f"frame:{self.franka_name}:finger_left:pose/quaternion",
                    finger_transform_np_left[3:]
                )
                pillar_state.update_property(
                    f"frame:{self.franka_name}:finger_right:pose/position",
                    finger_transform_np_right[:3]
                )
                pillar_state.update_property(
                    f"frame:{self.franka_name}:finger_right:pose/quaternion",
                    finger_transform_np_right[3:]
                )

                # EE
                ee_transform_np = transform_to_np(self.get_franka_ee_transform(), format='wxyz')
                pillar_state.update_property(
                    f"frame:{self.franka_name}:ee:pose/position",
                    ee_transform_np[:3]
                )
                pillar_state.update_property(
                    f"frame:{self.franka_name}:ee:pose/quaternion",
                    ee_transform_np[3:]
                )
        if update_franka_pose:
            pillar_state.update_property(
                f"frame:{self.franka_name}:ee:pose/linear_velocity",
                np.zeros((3,))
            )
            pillar_state.update_property(
                f"frame:{self.franka_name}:ee:pose/angular_velocity",
                np.zeros((3,))
            )

        # Constants
        pillar_state.update_property(
            f"constants/gripper_mass",
            self.gripper_mass
        )
        pillar_state.update_property(
            f"constants/{FrankaRodEnv.object_name}_dims",
            self._body_dims
        )
        pillar_state.update_property(
            f"constants/finger_left_dims",
            self._finger_dims
        )
        pillar_state.update_property(
            f"constants/finger_right_dims",
            self._finger_dims
        )

        self._pillar_states[env_idx] = State.create_from_serialized_string(
            pillar_state.get_serialized_string())  # defensive copy, please avoid bugs


    def set_gripper_width_target(self, width_target, env_idx, use_moveit=False):
        if abs(self._franka.get_gripper_width() - width_target) < 0.005:
            return
        start_time = time.time()
        self._franka.goto_gripper(width_target, grasp=False, speed=0.08)#, force = 1)
        end_time = time.time()
        print(f"Gripper time: {end_time-start_time} for width {width_target}")
        time.sleep(0.1)
        #print(width_target, "actual width", self._franka.get_gripper_width())
        if abs(self._franka.get_gripper_width() - 2*width_target) > 0.01:
            time.sleep(0.1)
            self._franka.stop_gripper()
            self._franka.wait_for_gripper()
            self._franka.goto_gripper(width_target)
            if  abs(self._franka.get_gripper_width() - width_target) > 0.01:
                print("Gripper controller problem")


    def _compute_ee_pose(self):
        return [self.get_franka_ee_transform()]

    def _compute_affected_distance_change(self):
        """
        Returns: joint angle distance difference between current ee pose and last pose
        to be similar to previous version, hence why it's a little weird

        warning: mutates self._current_ee_poses! similar to energy version
        """
        old_ee_poses = self._current_ee_poses
        new_ee_poses = self._compute_ee_pose()
        distance_changes = np.zeros(self.n_envs)
        for env_idx in range(self.n_envs):
            distance_changes[env_idx] = np.linalg.norm(
                transform_to_np(old_ee_poses[env_idx], format="wxyz")[:3] \
                - transform_to_np(new_ee_poses[env_idx], format="wxyz")[:3]
            )  # only xyz, rotation is free just because it's hard to compare the two and then add them together, different scales
        self._current_ee_poses = new_ee_poses
        return distance_changes

    def set_state(self, state, env_idx, n_steps=False):
        print("Reminder, set state does not do anything in the real world")

    def set_all_states(self, states, env_idxs=None, n_steps=0):
        pass

    def get_all_states(self, env_idxs=None):
        return [self.get_state(0) ]

    def get_state(self, env_idx, update_rod_poses=False):
        if update_rod_poses:
            self._update_pillar_state(0, update_rod_poses=True)
        return State.create_from_serialized_string(self._pillar_states[env_idx].get_serialized_string())

    def get_end_state(self, should_reset_to_viewable=False):
        if should_reset_to_viewable:
            self.reset_to_viewable()
        self._update_pillar_state(0, update_rod_poses=should_reset_to_viewable, update_franka_pose = False)
        return self.get_state(0)


    def _compute_costs(self):
        distance_change = self._compute_affected_distance_change()
        return distance_change


