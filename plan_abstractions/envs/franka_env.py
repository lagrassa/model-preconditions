from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import copy
import seaborn as sns
import logging
from collections import OrderedDict

from autolab_core import RigidTransform
from pillar_state import State
import seaborn as sns
import quaternion

from isaacgym import gymapi
from isaacgym_utils.assets import GymBoxAsset, GymFranka, GymURDFAsset, GymCapsuleAsset
from isaacgym_utils.draw import draw_contacts, draw_transforms
from isaacgym_utils.math_utils import rpy_to_quat, RigidTransform_to_transform, np_to_transform, quat_to_np, \
    transform_to_np, vec3_to_np, quat_to_np, compute_task_space_impedance_control, np_quat_to_quat, project_to_line, \
    quat_to_rot, angle_axis_between_axes, np_to_vec3

from .base_env import BaseEnv
from .utils import is_pose_of_object_close, get_pose_pillar_state, set_pose, get_joint_position_pillar_state, \
    get_joint_velocity_pillar_state, get_gripper_width_pillar_state, is_state_of_robot_close, make_env_with_init_states
from ..utils import yaw_from_quat, yaw_from_np_quat, get_rod_rel_goal_RigidTransforms_x_in, pillar_state_to_shapes, \
    object_name_to_asset_name, pillar_state_obj_to_transform, xyz_yaw_to_transform, r_flip_yz, \
    get_object_names_in_pillar_state, set_franka_pillar_properties_from_init_states_arr, is_obj_in_gripper, \
    extract_xy_dims_and_height, params_cause_collision_franka, transform_to_xy_yaw, \
    set_fingers_and_visualize_pillar_state, place_grippers_in_pillar_state


class FrankaRodEnv(BaseEnv):
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

    @classmethod
    def visualize_pillar_state(cls, pillar_state):
        # self._asset_name_to_eps_arr = {
        #     "finger_left": [2 * self._collision_eps, 2 * self._collision_eps],
        #     "finger_right": [2 * self._collision_eps, 2 * self._collision_eps],
        #     "pencil": [self._collision_eps, self._collision_eps]
        # }
        set_fingers_and_visualize_pillar_state(pillar_state, cls)

    def __init__(self, cfg, setup_callbacks=[], for_mde_training=False, baseboard = True):
        super().__init__(cfg, for_mde_training=for_mde_training)
        self._ground_height = 0
        if baseboard:
            baseboard_dims = {'sx':.6, 'sy':1.2, 'sz': 0.006}
            shape_props = {'restitution': 0.001,  'friction': 0.02, 'rolling_friction': 0.0001, 'thickness': 1e-3}
            asset_options= {'density': 10000}
            self._baseboard = GymBoxAsset(self._scene, **baseboard_dims, shape_props=shape_props, rb_props=None)
            self._ground_height = baseboard_dims['sz']/2
            self._baseboard_xform = gymapi.Transform(p=gymapi.Vec3(0.5,0,baseboard_dims["sz"]/2))
            self._baseboard_dims = baseboard_dims



        self._franka = GymFranka(cfg['franka'], self._scene, actuation_mode='attractors')
        if 'rod' in cfg.keys():
            if cfg['rod'].get('capsule', False):
                self._rod = GymCapsuleAsset(self._scene, **cfg[FrankaRodEnv.object_name]['dims'],
                                           shape_props=cfg[FrankaRodEnv.object_name]['shape_props'],
                                           rb_props=cfg[FrankaRodEnv.object_name]['rb_props'],
                                           asset_options=cfg[FrankaRodEnv.object_name]['asset_options']
                                           )
                self._rod_dims = {} #make it like a box. It's approximately correct
                self._rod_dims['sx'] = cfg[FrankaRodEnv.object_name]['dims']['radius']
                self._rod_dims['sy'] = cfg[FrankaRodEnv.object_name]['dims']['width']*2
                self._rod_dims['sz'] = cfg[FrankaRodEnv.object_name]['dims']['radius']
            else:
                self._rod = GymBoxAsset(self._scene, **cfg[FrankaRodEnv.object_name]['dims'],
                                           shape_props=cfg[FrankaRodEnv.object_name]['shape_props'],
                                           rb_props=cfg[FrankaRodEnv.object_name]['rb_props'],
                                           asset_options=cfg[FrankaRodEnv.object_name]['asset_options']
                                           )
                self._rod_dims = cfg[FrankaRodEnv.object_name]['dims']
            self._body_dims = list(cfg[FrankaRodEnv.object_name]['dims'].values())
            self._rod_mass = cfg[FrankaRodEnv.object_name]['asset_options']['density'] * (self._rod_dims['sx'] * self._rod_dims['sy'] * self._rod_dims['sz'])

        franka_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, 1e-2))
        self.franka_name = FrankaRodEnv.franka_name  # needs to match what's used in pillar_state
        if "num_rods" in cfg["env_props"].keys():
            self._num_rods = cfg["env_props"]["num_rods"]
            spacing = 0.05
            poses = ([spacing * i, 0, 0] for i in
                     range(self.num_rods))  # really meant to be used with generate_init_states
        else:
            self._num_rods = len(cfg['env_props']['initial_states']['rod_poses'])
            poses = cfg['env_props']['initial_states']['rod_poses']
        rod_transforms = [
            gymapi.Transform(p=gymapi.Vec3(pose[0], pose[1], self._rod_dims['sz'] / 2 + self._ground_height + 2e-3),
                             r=rpy_to_quat([0, 0, np.deg2rad(pose[2])]))
            for pose in poses]

        self._rod_names = [f'rod{i}' for i in range(len(rod_transforms))]

        self._finger_dims = [0.01, 0.01, 0.01]

        self._n_envs = cfg['scene']['n_envs']
        self._dt = cfg['scene']['gym']['dt']
        self._g = self._scene.gym.get_sim_params(self._scene.sim).gravity.z
        self._collision_eps = cfg["env_props"]["collision_eps"]

        # for maintaining a straight elbow
        self._elbow_joint = 3
        self._elbow_Ks = np.diag([500] * 3 + [1] * 3)
        self._elbow_Ds = 2 * np.sqrt(self._elbow_Ks)

        # for maintaining vertical gripper
        # For PD skill
        # self._gripper_orient_Ks = np.diag([0] * 3 + [1, 1, 1])
        # For SweepXYZYaw skill
        self._gripper_orient_Ks = np.diag([0] * 3 + [10, 10, 0])
        self._gripper_orient_Ds = 2 * np.sqrt(self._gripper_orient_Ks)

        self._grasp_height = 0.005  # height where franka can grasp
        self._obstacle_free_height = 0.18  # height above which there are guaranteed to be no obstacles #TODO dont hardcode
        self._asset_name_to_eps_arr = {
            "finger_left": [2 * self._collision_eps, 2 * self._collision_eps],
            "finger_right": [2 * self._collision_eps, 2 * self._collision_eps],
            "rod": [self._collision_eps, self._collision_eps]
        }
        self._conservative_asset_name_to_eps_arr = {
            "finger_left": [1 * self._collision_eps, 1 * self._collision_eps],
            "finger_right": [1 * self._collision_eps, 1 * self._collision_eps],
            "rod": [5*self._collision_eps, 5*self._collision_eps]
        }

        def setup(scene, env_idx):
            collision_filter = 1
            self._scene.add_asset(self.franka_name, self._franka, franka_transform, collision_filter=collision_filter)
            if baseboard:
                self._scene.add_asset("baseboard", self._baseboard, self._baseboard_xform, collision_filter=collision_filter)

            collision_filter *= 2
            for rod_name, rod_transform in zip(self._rod_names, rod_transforms):
                self._scene.add_asset(rod_name, self._rod, rod_transform, collision_filter=collision_filter)
                collision_filter *= 2
            self._next_collision_filter = collision_filter

            for setup_callback in setup_callbacks:
                setup_callback(self, scene, env_idx)

        self._scene.setup_all_envs(setup)
        self._update_gt_masses()

        self._action_norms = np.zeros(self.n_envs)

        sx_franka = 0.01
        if len(self._rod_names):
            self.setup_rod_tfs(sx_franka)
        #self._energies = self._compute_current_energies()
        self._current_ee_poses = self._compute_ee_pose()
        for env_idx in range(self.n_envs):
            self._update_pillar_state(env_idx=env_idx)

        # Update colors (just for debugging, so we can do it at the end)
        rod_colors = FrankaRodEnv.get_rod_colors(self._num_rods, only_ig_colors=True)
        for env_idx in range(self.n_envs):
            for rod_idx in range(self._num_rods):
                self._rod.set_rb_props(env_idx, self._rod_names[rod_idx],
                                          rb_props={'color': rod_colors[rod_idx]})

        # for sampling initial states
        self._franka_init_states = np.load(Path(cfg['original_cwd']) / cfg['env_props']['franka_init_states_path'])
        self._skill_params = [None for _ in range(self._scene.n_envs)]

    def setup_rod_tfs(self, sx_franka):
        sx_rod = self.rod_dims['sx']
        sy_rod = self.rod_dims['sy']
        th = 5e-2  # 4 more reliable
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

    def set_skill_params(self, env_idx, params):
        self._skill_params[env_idx] = params

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def num_rods(self):
        return self._num_rods

    def set_attractor(self, transform, env_idx):
        self._franka.set_ee_transform(env_idx, self.franka_name, transform)
        distance = np.linalg.norm(transform_to_np(transform)[:3] - transform_to_np(self.get_franka_ee_transform(env_idx))[:3])

    def set_delta_attractor(self, delta_transform, env_idx):
        current_transform = self.get_franka_ee_transform(env_idx)
        transform = delta_transform * current_transform
        self.set_attractor(transform, env_idx)

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

    @staticmethod
    def get_pusher_color():
        return np.array([0.3, 0.3, 0.3])

    @staticmethod
    def get_rod_colors(n_rods, only_ig_colors=False, only_sem_colors=False):
        colors = sns.color_palette("Paired", n_colors=2 * n_rods)
        if only_ig_colors:
            return colors[1::2]
        elif only_sem_colors:
            return colors[0::2]
        else:
            return colors

    def get_rigid_body_props_for_objects(self, env_idxs=None):
        if env_idxs is None:
            env_idxs = self._scene.env_idxs
        rb_props_by_name_dict = dict()
        for name, asset in [(FrankaRodEnv.object_name, self._rod)]:
            rb_props_by_name_dict[name] = []
            for env_idx in env_idxs:
                rb_props = asset.get_rb_props(env_idx, self._rod_names[0])[0]

                rb_props_dict = dict(
                    mass=rb_props.mass,
                    inertia=np.c_[
                        vec3_to_np(rb_props.inertia.x),
                        vec3_to_np(rb_props.inertia.y),
                        vec3_to_np(rb_props.inertia.z)
                    ],
                )
                rb_props_by_name_dict[name].append(rb_props_dict)
        return rb_props_by_name_dict

    def get_shape_props_for_objects(self, env_idxs=None):
        if env_idxs is None:
            env_idxs = self._scene.env_idxs

        shape_props_by_name_dict = dict()
        for name, asset in [(FrankaRodEnv.object_name, self._rod)]:
            shape_props_by_name_dict[name] = []
            for env_idx in env_idxs:
                for rod_name in self._rod_names:
                    shape_props = asset.get_shape_props(env_idx, rod_name)[0]

                shape_props_dict = dict(
                    friction=shape_props.friction,
                    rolling_friction=shape_props.rolling_friction,
                    torsion_friction=shape_props.torsion_friction,
                    restitution=shape_props.restitution,
                    thickness=shape_props.thickness,
                )
                shape_props_by_name_dict[name].append(shape_props_dict)

        return shape_props_by_name_dict
    
    @staticmethod
    def get_dict_representation_for_sem_data(initial_sem_state, end_sem_state, end_states_diff, object_masks, sem_state_obj_names):
        state_indexes = FrankaRodEnv.get_state_indexes_for_sem_objects(sem_state_obj_names)
        object_info_dict = OrderedDict()
        for object_idx, object_name in enumerate(sem_state_obj_names):
            init_state = initial_sem_state[state_indexes[object_idx]]
            end_state = end_sem_state[state_indexes[object_idx]]
            end_state_diff = end_states_diff[state_indexes[object_idx]]
            object_mask = object_masks[object_idx]
            is_robot = 'franka' in object_name

            object_dict = {
                'name': object_name,
                'initial_state': init_state,
                'end_state': end_state,
                'end_state_diff': end_state_diff,
                'mask': object_mask,
                'is_robot': is_robot,
            }
            object_info_dict[f'node_{object_idx}'] = object_dict
        return object_info_dict

    
    @staticmethod
    def object_state_size_in_sem_state(sem_object_name):
        return 4
    
    @staticmethod
    def get_state_indexes_for_sem_objects(sem_object_names):
        indexes = []
        start_idx = 0
        for name in sem_object_names:
            size = FrankaRodEnv.object_state_size_in_sem_state(name)
            indexes.append(list(range(start_idx, start_idx + size)))
            start_idx += size

        return indexes
    
    @staticmethod
    def pillar_state_to_sem_state_masks(end_pillar_state, start_pillar_state, sem_state_obj_names, **kwargs):
        '''Find a binary mask if the start pillar state and end pillar state differ beyond a threshold.'''
        masks = []
        pos_threshold = kwargs.get('position_th', 0.008)
        yaw_threshold = kwargs.get('yaw_th', 5.0)
        for i, name in enumerate(sem_state_obj_names):
            w_T_o = pillar_state_obj_to_transform(start_pillar_state, name)
            start_pos = vec3_to_np(w_T_o.p)
            start_yaw = yaw_from_np_quat(quat_to_np(w_T_o.r, 'wxyz'))

            w_T_o_end = pillar_state_obj_to_transform(end_pillar_state, name)
            end_pos = vec3_to_np(w_T_o_end.p)
            end_yaw = yaw_from_np_quat(quat_to_np(w_T_o_end.r, 'wxyz'))

            pos_unchanged = np.all(np.abs(start_pos-end_pos) < pos_threshold)
            yaw_unchanged = np.all(np.abs(start_yaw-end_yaw) < yaw_threshold)

            mask = 0 if pos_unchanged and yaw_unchanged else 1
            masks.append(mask)
        
        return np.array(masks, dtype=np.int32)


    @staticmethod
    def pillar_state_to_sem_state(pillar_state, sem_state_obj_names, anchor_obj_name=None, ref_pillar_state=None):
        sem_state = np.zeros(len(sem_state_obj_names) * 4)  # xyz yaw

        if anchor_obj_name is None:
            a_T_w = gymapi.Transform()
        else:
            a_T_w = pillar_state_obj_to_transform(
                ref_pillar_state, anchor_obj_name, align_z=True
            ).inverse()

        for i, name in enumerate(sem_state_obj_names):
            w_T_o = pillar_state_obj_to_transform(pillar_state, name)
            a_T_o = a_T_w * w_T_o

            sem_state[i * 4: i * 4 + 3] = vec3_to_np(a_T_o.p)
            sem_state[i * 4 + 3] = yaw_from_np_quat(quat_to_np(a_T_o.r, 'wxyz'))

        return sem_state

    @staticmethod
    def sem_state_to_pillar_state(sem_state, ref_pillar_state, sem_state_obj_names, anchor_obj_name=None):
        state = State.create_from_serialized_string(ref_pillar_state.get_serialized_string())

        if anchor_obj_name is None:
            w_T_a = gymapi.Transform()
        else:
            w_T_a = pillar_state_obj_to_transform(
                ref_pillar_state, anchor_obj_name, align_z=True
            )

        anchor_obj_is_ee = anchor_obj_name == 'franka:ee'
        w_T_ee = None
        for i, name in enumerate(sem_state_obj_names):
            xyz_yaw = sem_state[i * 4: (i + 1) * 4].copy()
            obj_is_ee = name == 'franka:ee'

            if anchor_obj_name is None:
                if obj_is_ee:
                    xyz_yaw[3] *= -1

                a_T_o = xyz_yaw_to_transform(xyz_yaw)

                if obj_is_ee:
                    a_T_o.r = a_T_o.r * r_flip_yz
            else:
                if anchor_obj_is_ee and not obj_is_ee:
                    xyz_yaw[3] *= -1

                a_T_o = xyz_yaw_to_transform(xyz_yaw)

                if anchor_obj_is_ee and not obj_is_ee:
                    a_T_o.r = a_T_o.r * r_flip_yz

            w_T_o = w_T_a * a_T_o

            state.set_values_from_vec([f'frame:{name}:pose/position'], vec3_to_np(w_T_o.p))
            state.set_values_from_vec([f'frame:{name}:pose/quaternion'], quat_to_np(w_T_o.r, 'wxyz'))

            if obj_is_ee:
                w_T_ee = w_T_o
        
        # Update finger transforms, assuming gripper width is the same between the ref state and the sem state
        if w_T_ee is None:
            w_T_ee = pillar_state_obj_to_transform(state, 'franka:ee')
            
        if anchor_obj_is_ee:
            w_T_ee_ref = w_T_a
        else:
            w_T_ee_ref = pillar_state_obj_to_transform(ref_pillar_state, 'franka:ee')
        ee_ref_T_w = w_T_ee_ref.inverse()

        w_T_finger_left_ref = pillar_state_obj_to_transform(ref_pillar_state, 'franka:finger_left')
        w_T_finger_right_ref = pillar_state_obj_to_transform(ref_pillar_state, 'franka:finger_right')

        ee_ref_T_finger_left_ref = ee_ref_T_w * w_T_finger_left_ref
        ee_ref_T_finger_right_ref = ee_ref_T_w * w_T_finger_right_ref

        w_T_finger_left = w_T_ee * ee_ref_T_finger_left_ref
        w_T_finger_right = w_T_ee * ee_ref_T_finger_right_ref

        state.set_values_from_vec([f'frame:franka:finger_left:pose/position'], vec3_to_np(w_T_finger_left.p))
        state.set_values_from_vec([f'frame:franka:finger_left:pose/quaternion'], quat_to_np(w_T_finger_left.r, 'wxyz'))
        state.set_values_from_vec([f'frame:franka:finger_right:pose/position'], vec3_to_np(w_T_finger_right.p))
        state.set_values_from_vec([f'frame:franka:finger_right:pose/quaternion'], quat_to_np(w_T_finger_right.r, 'wxyz'))

        return state

    def planner_state_to_viz_string(self, planner_state):
        pillar_state = planner_state.pillar_state
        action_in = planner_state.action_in
        body_names = self.get_object_names()
        poses = []
        for body_name in body_names:
            body_tf = pillar_state_obj_to_transform(pillar_state, body_name)
            pose = np.array([body_tf.p.x, body_tf.p.y, body_tf.p.z, yaw_from_np_quat(quat_to_np(body_tf.r, 'wxyz'))])
            poses.append(pose)

        state_str = ''
        for body_name, pose in zip(body_names, poses):
            state_str += f"{body_name}: {np.around(pose, 2)}\n"
        g = 1e6 if planner_state.g is None else planner_state.g
        state_str += f"h: {np.around(planner_state.h, 3)}\n" \
                     f"g: {np.around(g, 3)}\n" \
                     f"id: {planner_state.debug_id}"
        if action_in:
            action_str = f"{action_in.skill_idx}, {np.around(action_in.params, 2)}\n" \
                         f"cost: {action_in.cost:.2f}"
        else:
            action_str = ''
        return state_str, action_str
    
    @staticmethod
    def sem_state_to_separate_object_states(sem_state):
        ''' Convert SEM states to object states. 
        
        This conversion is needed in many places, so creating a central place for it.

        sem_state: Numpy array of (N, S) where N is the number of states and S is the state size
        '''
        assert len(sem_state.shape) == 2 and (sem_state.shape[1] - 4 - 3) % 4 == 0
        num_rods = (sem_state.shape[1] - 4 - 3) // 4
        per_object_state_dict = dict()
        per_object_state_dict['pusher'] = sem_state[:, :4]
        for i in range(num_rods):
            rod_state = sem_state[4+4*i:4+4*(i+1)]
            per_object_state_dict[f'rod{i}'] = rod_state
        return  per_object_state_dict

    def add_visual_box_cb(self, goal_pose, goal_dims):
        """
        Returns a callback function that creates a box of specified dims that does not collide with anything
        """
        rb_props = {'color': [0., 0.7, 0.]}
        shape_props = {'thickness': 1e-3}
        dims = {'sx': goal_dims[0], 'sy': goal_dims[1], 'sz': goal_dims[2]}
        self._goal = GymBoxAsset(self._scene, **dims,
                                 rb_props=rb_props,
                                 shape_props=shape_props)
        eps = 2e-3
        goal_transform = gymapi.Transform(p=gymapi.Vec3(goal_pose[0], goal_pose[1], (dims['sz'] / 2.) + eps))
        self._goal_name = FrankaRodEnv.goal_name
        self._scene.add_asset(self._goal_name, self._goal, goal_transform, collision_filter= 2*self._next_collision_filter - 1)

    def add_real_drawer_cb(self, env, scene, env_idx, drawer_pose=None):
        """
        This needed to be back in FrankaEnv for some of the data collection to work...
        """
        if drawer_pose is None:
            if 'drawer' in self._cfg.keys():
                drawer_pose = self._cfg["drawer"]["pose"]

        self._drawer_name = 'drawer'
        env._drawer_name = self._drawer_name
        original_cwd = Path(self._cfg['original_cwd'])
        self._drawer = GymURDFAsset(
            self._cfg['drawer']['urdf_path'],
            self._scene,
            assets_root=original_cwd / self._cfg['drawer']['asset_root'],
            shape_props=self._cfg['drawer']['shape_props'],
            rb_props=self._cfg['drawer']['rb_props'],
            asset_options=self._cfg['drawer']['asset_options'],
            dof_props=self._cfg['drawer']['dof_props']
        )
        env._drawer = self._drawer
        env._drawer_base_transform = gymapi.Transform(p=np_to_vec3(drawer_pose))

        env._next_collision_filter *= 2
        env._scene.add_asset(env._drawer_name, env._drawer, env._drawer_base_transform,
                             collision_filter=env._next_collision_filter)  # collision_filter)

    def reset_real_drawer(self, drawer_pose_xy):
        self._drawer_base_transform.p.x = drawer_pose_xy[0]
        self._drawer_base_transform.p.y = drawer_pose_xy[1]
        joint_val = self._drawer_base_transform.p.y - 0.19 - drawer_pose_xy[1]  # magic number is the length of the drawer
        for env_idx in range(self.n_envs):
            self._drawer.set_joints(env_idx, "drawer", joint_val)
            self._drawer.set_rb_transforms(env_idx=env_idx, name=self._drawer_name, transforms=[self._drawer_base_transform])


    def get_end_state(self, should_reset_to_viewable=False):
        self._update_pillar_state(0)
        return self.get_state(0)


    def add_real_box_cb(self, goal_pose, goal_dims):
        """
        Returns a callback function that creates a box of specified dims that does not collide with anything
        """

        self._box_name = "box"
        width = goal_dims[0] #0.172
        length = goal_dims[1] # 0.193
        depth = goal_dims[2] #0.026
        thickness = goal_dims[3] #0.002
        collision_filter = 1
        box_rb_props = {'flags': "none", "color": [0.1, 0.7, 0.1]}
        bottom_dims = {'sx': length, 'sy': width, 'sz': thickness}
        long_side_dims = {'sx': length, 'sy': thickness, 'sz': depth - thickness}
        short_side_dims = {'sx': thickness, 'sy': width - 2 * thickness,
                           'sz': depth - thickness}  # compensate for long sides taking up more space
        shape_props = {'restitution': 0.05, 'friction':0.7, 'rolling_friction':0.001, 'thickness':1.5e-2}
        asset_options = {"fix_base_link":True}
        self._box_bottom = GymBoxAsset(self._scene, **bottom_dims, shape_props=shape_props, rb_props=box_rb_props,asset_options=asset_options)
        self._box_long_side = GymBoxAsset(self._scene, **long_side_dims, shape_props=shape_props, rb_props=box_rb_props, asset_options=asset_options)
        self._box_short_side = GymBoxAsset(self._scene, **short_side_dims, shape_props=shape_props, rb_props=box_rb_props, asset_options=asset_options)
        self._box_center = goal_pose
        self._box_bottom_transform = gymapi.Transform(
            p=gymapi.Vec3(self._box_center[0], self._box_center[1], thickness / 2. + 1e-3))
        self._box_long_side_right_transform = gymapi.Transform(
            p=gymapi.Vec3(self._box_center[0], self._box_center[1] + width / 2 - thickness / 2, depth / 2. + thickness / 2 + 1e-3))
        self._box_long_side_left_transform = gymapi.Transform(
            p=gymapi.Vec3(self._box_center[0], self._box_center[1] - width / 2 + thickness / 2, depth / 2. + thickness / 2 + 1e-3))
        self._box_short_side_near_transform = gymapi.Transform(
            p=gymapi.Vec3(self._box_center[0] - length / 2 + thickness / 2, self._box_center[1], depth / 2. + thickness / 2 + 1e-3))
        self._box_short_side_far_transform = gymapi.Transform(
            p=gymapi.Vec3(self._box_center[0] + length / 2 - thickness / 2, self._box_center[1], depth / 2. + thickness / 2 + 1e-3))

        box_part_names_postfix = ["_long_side_right", "_long_side_left", "_short_side_near", "_short_side_far"]
        self._box_part_xforms = [self._box_long_side_right_transform, self._box_long_side_left_transform, self._box_short_side_near_transform, self._box_short_side_far_transform]
        self._box_part_assets = [self._box_long_side, self._box_long_side, self._box_short_side, self._box_short_side]
        self._box_part_names = []
        for box_part_postfix,box_part_asset, box_part_xform in zip(box_part_names_postfix, self._box_part_assets, self._box_part_xforms):
            self._scene.add_asset(self._box_name + box_part_postfix, box_part_asset, box_part_xform,
                                  collision_filter=collision_filter)
            self._box_part_names.append(self._box_name + box_part_postfix)

    def add_real_slot_cb(self, goal_pos, goal_dims):
        self._slot_name = "slot"
        width = goal_dims[0] #0.172
        thickness = goal_dims[1] # 0.193
        height = goal_dims[2] #0.026
        hole_size = goal_dims[3] #0.026
        collision_filter = 1
        slot_rb_props = {'flags': "none", "color": [0.7, 0.7, 0.1]}
        middle_thin_dims = {'sx': hole_size, 'sy': thickness, 'sz': (height- hole_size) / 2}
        side_dims = {'sx': (width-hole_size)/2, 'sy':thickness, 'sz': height}
        shape_props = {'restitution': 0.001, 'friction':1, 'rolling_friction':0.6, 'thickness':5e-3}
        asset_options = {"fix_base_link":True}
        self._middle_thin = GymBoxAsset(self._scene, **middle_thin_dims, shape_props=shape_props, rb_props=slot_rb_props,asset_options=asset_options)
        self._side = GymBoxAsset(self._scene, **side_dims, shape_props=shape_props, rb_props=slot_rb_props,asset_options=asset_options)
        self._slot_center = goal_pos
        self._middle_bottom_xform = gymapi.Transform(
            p=gymapi.Vec3(self._slot_center[0], self._slot_center[1], middle_thin_dims['sz']/2 + 1e-3))
        self._middle_top_xform = gymapi.Transform(
            p=gymapi.Vec3(self._slot_center[0], self._slot_center[1], 1.5*middle_thin_dims['sz'] + hole_size + 1e-3))
        self._side_close_xform = gymapi.Transform(
            p=gymapi.Vec3(self._slot_center[0] + (side_dims['sx'] + hole_size) / 2, self._slot_center[1], side_dims['sz']/2 + 1e-3))
        self._side_far_xform = gymapi.Transform(
            p=gymapi.Vec3(self._slot_center[0] - (side_dims['sx'] + hole_size) / 2, self._slot_center[1], side_dims['sz']/2 + 1e-3))

        slot_part_names_postfix = ["_middle_bottom", "_middle_top", "_side_close", "_side_far"]
        self._slot_part_xforms = [self._middle_bottom_xform, self._middle_top_xform, self._side_close_xform, self._side_far_xform]
        self._slot_part_assets = [self._middle_thin, self._middle_thin, self._side, self._side]
        self._slot_part_names = []
        for slot_part_postfix,slot_part_asset, slot_part_xform in zip(slot_part_names_postfix, self._slot_part_assets, self._slot_part_xforms):
            self._scene.add_asset(self._slot_name + slot_part_postfix, slot_part_asset, slot_part_xform,
                                  collision_filter=collision_filter)
            self._slot_part_names.append(self._slot_name + slot_part_postfix)

    def reset_real_slot(self, goal_pose):
        if not hasattr(self, '_box_name'):
            return
        for slot_part_name, slot_part_asset, slot_part_xform in zip(self._slot_part_names, self._slot_part_assets, self._slot_part_xforms):
            slot_part_xform.p.x -= self._slot_center[0]
            slot_part_xform.p.x += goal_pose[0]
            slot_part_xform.p.y -= self._slot_center[1]
            slot_part_xform.p.y += goal_pose[1]
            for env_idx in range(self.n_envs):
                slot_part_asset.set_rb_transforms(env_idx, slot_part_name, [slot_part_xform])
        self._slot_center = [goal_pose[0], goal_pose[1]]

    def reset_real_box(self, goal_pose):
        if not hasattr(self, '_box_name'):
            return
        for box_part_name, box_part_asset, box_part_xform in zip(self._box_part_names, self._box_part_assets, self._box_part_xforms):
            box_part_xform.p.x -= self._box_center[0]
            box_part_xform.p.x += goal_pose[0]
            box_part_xform.p.y -= self._box_center[1]
            box_part_xform.p.y += goal_pose[1]
            for env_idx in range(self.n_envs):
                box_part_asset.set_rb_transforms(env_idx, box_part_name, [box_part_xform])
        self._box_center = [goal_pose[0], goal_pose[1]]

    def reset_visual_box(self, goal_pose, goal_dims):
        if not hasattr(self, '_goal'):
            return
        dims = {'sx': goal_dims[0], 'sy': goal_dims[1], 'sz': goal_dims[2]}
        eps = 2e-3
        goal_transform = gymapi.Transform(p=gymapi.Vec3(goal_pose[0], goal_pose[1], (dims['sz'] / 2.) + eps))
        for env_idx in range(self.n_envs):
            self._goal.set_rb_transforms(env_idx, self._goal_name, [goal_transform])

    def generate_init_states(self, cfg, min_samples=10, max_samples=1000, init_state_flag=None, 
                             choose_from_multiple_rod_configs=False, return_init_state_info=False):
        ''' Generator for initial env states.

        choose_from_multiple_rod_configs:  If True use `rod_configs` Flag from the env config to select
            range
        return_init_state_info:  Use only if `choose_from_multiple_rod_configs: True`. If True, returns info
            dict regarding which config was selected.
        '''

        franka_holding = init_state_flag == "franka_holding"
        franka_near_ground = init_state_flag == "franka_near_ground"
        near_ground = franka_holding or franka_near_ground

        pose_ranges = cfg['env_props']['initial_states']['pose_ranges']
        object_names = ["finger_left", "finger_right"] + [f"rod{i}" for i in range(self.num_rods)]
        init_state_info = dict()


        low, high = pose_ranges['low'], pose_ranges['high']
        low_list = [low[:] for _ in range(self.num_rods)]
        high_list = [high[:] for _ in range(self.num_rods)]
        # Not using any config
        self._last_config_to_use = None
        
        assert len(low_list) == len(high_list) and len(low_list) == self.num_rods

        # Which start states to sample? For the general case just set `sample_start_states_general` to True.
        # Other cases are for specialized tasks.
        # Only one of the following should be true.
        sample_start_states_general = True

        for _ in range(min_samples):
            potential_pillar_state = State()
            for _ in range(max_samples):  # ensures picks one at table height
                init_state_idx = np.random.randint(0, len(self._franka_init_states['joints']))
                franka_ee_pose = self._franka_init_states['ee_poses'][init_state_idx]

                if sample_start_states_general:
                    if near_ground and franka_ee_pose[2] <= self._grasp_height + 0.005:
                        break
                    if not near_ground and franka_ee_pose[2] >= self._obstacle_free_height:
                        break

            set_franka_pillar_properties_from_init_states_arr(self.franka_name, self._franka_init_states, init_state_idx, potential_pillar_state)

            # initialize pillar_state with only relevant information that is not supposed to be randomized
            # assuming all positions are randomized now
            body_names_to_check = []  # already set
            if len(self._rod_names):
                rod_to_hold = np.random.choice(self._rod_names)

            #add fingers first
            for object_name in object_names:
                if "finger" in object_name:
                    self._add_obj_dims_and_vel_to_pillar_state_and_body_names(cfg, object_name, potential_pillar_state, body_names_to_check)

            #now add the rod to hold
            if franka_holding:
                asset_name = object_name_to_asset_name(rod_to_hold)
                _, object_height = extract_xy_dims_and_height(asset_name, cfg, self._finger_dims)
                self._add_obj_dims_and_vel_to_pillar_state_and_body_names(cfg, rod_to_hold, potential_pillar_state,
                                                                          body_names_to_check)
                self._set_object_loc_and_gripper_width(potential_pillar_state, rod_to_hold,
                                                       object_height, asset_name, body_names_to_check)
            #now add all the other rods
            for object_name in object_names:
                if "finger" in object_name:
                    continue
                if franka_holding:
                    if object_name == rod_to_hold:
                        continue #already set
                if "rod" in object_name:
                    # NOTE: Assumes less than 10 rods. 
                    rod_idx = int(object_name[-1])
                asset_name = object_name_to_asset_name(rod_to_hold)
                _, object_height = extract_xy_dims_and_height(asset_name, cfg, self._finger_dims)
                self._add_obj_dims_and_vel_to_pillar_state_and_body_names(cfg, object_name, potential_pillar_state,
                                                                          body_names_to_check)
                self._sample_and_set_collision_free_pose(potential_pillar_state, object_name, object_height,
                                                         body_names_to_check, high_list[rod_idx], low_list[rod_idx], max_samples)
            if return_init_state_info:
                yield potential_pillar_state, init_state_info
            else:
                yield potential_pillar_state

    def _add_obj_dims_and_vel_to_pillar_state_and_body_names(self, cfg, object_name, potential_pillar_state, body_names_to_check):
        if 'finger' in object_name:
            body_names_to_check.append(f"franka:{object_name}")  # now we need to check it with the others
        else:
            body_names_to_check.append(object_name)  # now we need to check it with the others
        asset_name = object_name_to_asset_name(object_name)
        xy_dims, object_height = extract_xy_dims_and_height(asset_name, cfg, self._finger_dims)
        potential_pillar_state.update_property(f"constants/{asset_name}_dims", xy_dims + [object_height])
        potential_pillar_state.update_property(f"frame:{object_name}:pose/angular_velocity", [0] * 3)
        potential_pillar_state.update_property(f"frame:{object_name}:pose/linear_velocity", [0] * 3)

    def _set_object_loc_and_gripper_width(self, potential_pillar_state, object_name, object_height, asset_name,
                                          body_names_to_check):
        # first object goes between grippers, wherever that is
        franka_ee_pose = get_pose_pillar_state(potential_pillar_state, "franka:ee")
        rod_xy_pose = franka_ee_pose[:2]
        pen_yaw = (yaw_from_np_quat(franka_ee_pose[3:])) + np.pi/2
        grasp_offset = self._cfg["rod"]["dims"]["sy"]/2 - 0.01
        offset = False #np.random.randint(0,2)
        if offset:
            rod_xy_pose[0] += grasp_offset * np.sin(pen_yaw)
            rod_xy_pose[1] += grasp_offset * np.cos(pen_yaw)
        potential_pillar_state.update_property(f"frame:{object_name}:pose/position",
                                               rod_xy_pose + [self._ground_height + object_height])
        rotate = RigidTransform.z_axis_rotation(np.pi / 2)
        franka_rotation = RigidTransform.rotation_from_quaternion(franka_ee_pose[3:])
        new_rotation = franka_rotation @ rotate  # reverse?
        obj_quat_np = RigidTransform(rotation=new_rotation).quaternion

        potential_pillar_state.update_property(f"frame:{object_name}:pose/quaternion",
                                               obj_quat_np)  # TODO might need rotation
        asset_name_to_eps_arr_grasp, potential_pillar_state = self._update_gripper_width_around_obj(asset_name,
                                                                                                    potential_pillar_state)
        assert not self.is_in_collision(potential_pillar_state, body_names=body_names_to_check, asset_name_to_eps_arr=asset_name_to_eps_arr_grasp, plot=False)

    def _update_gripper_width_around_obj(self, asset_name, potential_pillar_state):
        smallest_object_dim = min(potential_pillar_state.get_values_as_vec([f"constants/{asset_name}_dims"]))
        obj_collision_eps = self._conservative_asset_name_to_eps_arr["rod"][0]
        potential_pillar_state.update_property(f"frame:{self.franka_name}:gripper/width",
                                               2 * obj_collision_eps + smallest_object_dim + 1e-3)
        asset_name_to_eps_arr_grasp = self.asset_name_to_eps_arr
        asset_name_to_eps_arr_grasp["franka:finger_left"] = -self._collision_eps
        asset_name_to_eps_arr_grasp[
            "franka:finger_right"] = -self._collision_eps  # should be less conservative when already holding
        #_, _, _, potential_pillar_state = place_grippers_in_pillar_state(potential_pillar_state, 0, franka_xyzyaw=None)
        #Not working
        return asset_name_to_eps_arr_grasp, potential_pillar_state

    def _sample_and_set_collision_free_pose(self, potential_pillar_state, object_name, object_height,
                                            body_names_to_check, high, low, max_samples, max_height_diff = 0.02):
        for j in range(max_samples):
            random_vec = np.random.uniform(low=low, high=high)
            random_yaw = np.deg2rad(random_vec[2])
            random_quat = quat_to_np(rpy_to_quat((0, 0, random_yaw)), format="wxyz")
            random_pose = np.hstack([random_vec[:2], [object_height / 2 + self._ground_height + self._collision_eps]])
            potential_pillar_state.update_property(f"frame:{object_name}:pose/position", random_pose)
            potential_pillar_state.update_property(f"frame:{object_name}:pose/quaternion", random_quat)
            # eps needs to be smaller, especially for franka_holding
            asset_name_to_eps_arr_grasp, potential_pillar_state = self._update_gripper_width_around_obj('rod',
                                                                                                        potential_pillar_state)

            if not self.is_in_collision(potential_pillar_state, body_names=body_names_to_check, asset_name_to_eps_arr=self._conservative_asset_name_to_eps_arr, plot=False):
                break
            #else:
            #    self.is_in_collision(potential_pillar_state, body_names=body_names_to_check,
            #                                asset_name_to_eps_arr=self._conservative_asset_name_to_eps_arr, plot=1)

        if j == max_samples - 1:
            raise RuntimeError(
                f"Was not able to find a valid configuration for {object_name} in {max_samples} attempts")

    @classmethod
    def _make_collision_shapes(cls, pillar_state, body_names=None, plot=False, asset_name_to_eps_arr=None):
        if body_names is None:
            body_names = get_object_names_in_pillar_state(pillar_state)
        if "drawer" in body_names:
            new_body_names = body_names.copy()
            new_body_names.remove("drawer")
            assert "drawer" in body_names
        in_air = .18 #HACK to check if franka_gripper is in the z_free zone, which is technically a skill hyperparameter
        if get_pose_pillar_state(pillar_state, "franka:ee")[2] > in_air:
            body_names_new = []
            for body_name in body_names:
                if "franka" not in body_name:
                    body_names_new.append(body_name)
            body_names = body_names_new
        return pillar_state_to_shapes(pillar_state, body_names, asset_name_to_eps_arr=asset_name_to_eps_arr, plot=plot)

    def randomize_dynamics_params(self, env_idxs=None):
        shape_props_by_name_dict, rb_props_by_name_dict = dict(), dict()
        if env_idxs is None:
            env_idxs = self._scene.env_idxs

        all_shape_props = {}
        for key, (lo, hi) in self._cfg['env_props']['dynamics'][FrankaRodEnv.object_name]['shape_props'].items():
            all_shape_props[key] = np.random.uniform(lo, hi, self.n_envs)

        for env_idx in env_idxs:
            shape_props = {
                key: vals[env_idx]
                for key, vals in all_shape_props.items()
            }
            for rod_name in self._rod_names:
                self._rod.set_shape_props(env_idx, rod_name, shape_props)
        for rod_name in self._rod_names:
            shape_props_by_name_dict[rod_name] = copy.deepcopy(all_shape_props)

        all_rb_props = {}
        for key, (lo, hi) in self._cfg['env_props']['dynamics'][FrankaRodEnv.object_name]['rb_props'].items():
            all_rb_props[key] = np.random.uniform(lo, hi, self.n_envs)

        for env_idx in env_idxs:
            rb_props = {
                key: vals[env_idx]
                for key, vals in all_rb_props.items()
            }
            for rod_name in self._rod_names:
                self._rod.set_rb_props(env_idx, rod_name, rb_props)
        for rod_name in self._rod_names:
            rb_props_by_name_dict[rod_name] = copy.deepcopy(all_rb_props)

        self._update_gt_masses()
        return shape_props_by_name_dict, rb_props_by_name_dict

    def _update_gt_masses(self):
        if len(self._rod_names):
            self._gt_rod_masses = [
                self._rod.get_rb_props(env_idx, self._rod_names[0])[0].mass
                for env_idx in self._scene.env_idxs
            ]
            self._gt_rod_Is = [
                1 / 12 * mass * (self._rod_dims['sx'] ** 2 + self._rod_dims['sy'] ** 2)
                for mass in self._gt_rod_masses
            ]
            self._franka_masses = np.array([
                [rb_props.mass for rb_props in self._rod.get_rb_props(env_idx, self.franka_name)]
                for env_idx in self._scene.env_idxs
            ])
        self._gripper_mass = self._franka.get_rb_props(0, self.franka_name) \
                                 [self._franka.rb_names_map['panda_leftfinger']].mass * 2

    @staticmethod
    def get_pose_pillar_state(pillar_state, object_name):
        # pillar state only makes sense in the context of an env so makes sense to evaluate based on the env
        prop_names = [f"frame:{object_name}:pose/position", f"frame:{object_name}:pose/quaternion"]
        pose = pillar_state.get_values_as_vec(prop_names)
        return pose

    def states_similar_for_env(self, state1, state2, yaw_only=True, check_joints=True, position_tol=5e-3, angle_tol= 5e-3, velocity_tol=0.01):
        """
        For testing only: used to test if state1 == state2 in the context of this environment.
        underdefined for an arbitrary pillar state, would have to make context-dependent
        """
        ee_close = is_pose_of_object_close(state1, state2, f"{self.franka_name}:ee", position_tol=position_tol,angle_tol=angle_tol,
                                           yaw_only=yaw_only)
        if check_joints:
            robot_state_close = is_state_of_robot_close(state1, state2, self.franka_name, velocity_tol=velocity_tol)
        else:
            robot_state_close = True

        rods_close_results = []
        for rod_name in self._rod_names:
            # Rods should be quite close
            rods_close = is_pose_of_object_close(state1, state2, rod_name, 
                                                    yaw_only=yaw_only, position_tol=position_tol)
            rods_close_results.append(rods_close)
        
        return np.all(rods_close_results + [ee_close, robot_state_close])

    def get_rod_transforms(self, env_idx):
        rod_transforms = [self._rod.get_rb_transforms(env_idx, rod_name)[0]
                             for rod_name in self._rod_names]
        return rod_transforms

    def get_franka_ee_transform(self, env_idx):
        return self._franka.get_ee_transform(env_idx, self.franka_name)

    def get_goal_transforms_around_objects(self, state, rel_goal_transforms=None, plot=False):
        if rel_goal_transforms is None:
            rel_goal_transforms = self._rod_rel_goal_transforms

        rod_transforms = []
        for i in range(self.num_rods):
            rod_transform = np_to_transform(
                state.get_values_as_vec([f"frame:rod{i}:pose/position", f"frame:rod{i}:pose/quaternion"]),
                format='wxyz'
            )
            rod_transforms.append(rod_transform)

        goal_transforms = []
        for rod_transform in rod_transforms:
            goal_transforms += [rod_transform * rel_goal_transform
                                for rel_goal_transform in rel_goal_transforms]

        if plot:
            points = np.zeros((len(goal_transforms), 2))
            colors = np.zeros(len(goal_transforms))
            for i, transform in enumerate(goal_transforms):
                points[i, 0] = transform.p.x
                points[i, 1] = transform.p.y
                colors[i] = np.rad2deg(yaw_from_quat(transform.r) + np.pi)
            plt.figure()
            plt.scatter(points[:, 0], points[:, 1], c=colors)
            for rod_transform in rod_transforms:
                plt.scatter([rod_transform.p.x], [rod_transform.p.y], c='red')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()

        return goal_transforms

    def set_state(self, pillar_state, env_idx, n_steps=0):
        franka_joints = get_joint_position_pillar_state(pillar_state, self.franka_name)
        franka_joints_vels = get_joint_velocity_pillar_state(pillar_state, self.franka_name)
        gripper_width = max(get_gripper_width_pillar_state(pillar_state, self.franka_name)[0], 0.011) #hack to prevent collision
        ee_pose = get_pose_pillar_state(pillar_state, "franka:ee")
        transform = gymapi.Transform(p=np_to_vec3(ee_pose[:3]), r=np_quat_to_quat(quaternion.from_float_array(ee_pose[3:])))
        self._franka.set_ee_transform(env_idx, self.franka_name, transform)
        self._franka.set_joints(env_idx, self.franka_name, franka_joints + [gripper_width, gripper_width])
        self._franka.set_joints_velocity(env_idx, self.franka_name, franka_joints_vels + [0, 0])
        self._franka.set_gripper_width_target(env_idx, self.franka_name, gripper_width)

        for rod_name in self._rod_names:
            rod_pose = get_pose_pillar_state(pillar_state, rod_name)
            if rod_pose[2] < self._rod_dims["sz"]/2 + self._collision_eps + self._ground_height:
                rod_pose[2] =  self._rod_dims["sz"]/2 + self._collision_eps + self._ground_height + 0.01
            set_pose(rod_pose, rod_name, self._rod, env_idx)
        #else:
        #    pillar_state.update_property("frame:drawer:pose/position",[.43,.1,0.05])
        #    pillar_state.update_property("frame:drawer:pose/quaternion",[1,0,0,0])
        if self._for_mde_training and "frame:drawer:pose/position" in pillar_state.get_prop_names():
            drawer_pose = get_pose_pillar_state(pillar_state, "drawer")
            drawer_base_tform = self._drawer_base_transform
            joint_val = drawer_base_tform.p.y - 0.19 - drawer_pose[1]  # magic number is the length of the drawer
            curr_pose = self._drawer.get_rb_poses_as_np_array(env_idx, self._drawer_name)[0]
            curr_pose[0] = drawer_pose[0]
            new_pose = curr_pose.copy()
            set_pose(new_pose, self._drawer_name, self._drawer, env_idx)
            self._drawer.set_joints(env_idx, "drawer", joint_val)

        for _ in range(n_steps):
            self._scene.step()
            self._scene.render(custom_draws=self._custom_draws)


        self._update_pillar_state(env_idx)
        #self._energies = self._compute_current_energies()

    def _custom_draws(self, scene):
        for env_idx in scene.env_idxs:
            franka_ee_transform = self.get_franka_ee_transform(env_idx)

            rod_transforms = self.get_rod_transforms(env_idx)
            origin_tf = gymapi.Transform(p=gymapi.Vec3(0, 0, 1e-2))
            transforms = rod_transforms + [franka_ee_transform] + [origin_tf]

            draw_transforms(scene, [env_idx], transforms)

            # The below code helps in debugging, but it is very skill specific, hence, commented out
            # in general. In case you want to visualize if the skills reach their goals, uncomment the below
            # code, parse the skill parameters correctly and then visualize the result. There is a better
            # way to do this, but leave it for later.

            if self._skill_params[env_idx] is not None:
                # Maybe we should call the skill function some delegate method to return
                # some transform to display.

                # Parsing FreeSpacePD Skill
                # x, y, z, yaw = self._skill_params[env_idx][:4]
                # Paring LQRWaypointsXYZFranka Skill

                # Parsing LQRWaypointXYYaw Skill (uncomment for debugging)
                # (x, y, yaw), rods_to_push = self._skill_params[env_idx][:3],  self._skill_params[env_idx][3:]
                # z = 0.05
                # rod_transforms = [rod_transforms[p] for p in range(self._num_rods) if rods_to_push[p]]

                # R = [[np.cos(yaw), -np.sin(yaw), 0],
                #      [np.sin(yaw), np.cos(yaw), 0],
                #      [0, 0, 1]]
                # rigid_tf = RigidTransform(rotation=np.array(R), translation=np.array([x, y, z]))
                # tf = RigidTransform_to_transform(rigid_tf)
                # transforms = [tf] + rod_transforms
                # draw_transforms(scene, [env_idx], transforms, length=0.05)

                pass

        #draw_contacts(scene, scene.env_idxs)

    def _update_pillar_state(self, env_idx):
        pillar_state = self._pillar_states[env_idx]

        # Update rods
        names_of_objects_to_update = self._rod_names
        if len(self._rod_names):
            assets_of_objects_to_update = [self._rod] * len(self._rod_names)
            pillar_state.update_property(f"constants/{FrankaRodEnv.object_name}_mass", self.rod_mass)
        else:
            assets_of_objects_to_update = []
        pillar_state.update_property("constants/dt", self.dt)
        for object_name, object_asset in zip(names_of_objects_to_update, assets_of_objects_to_update):
            object_pos = object_asset.get_rb_poses_as_np_array(env_idx, object_name)[0]
            object_vel = object_asset.get_rb_vels_as_np_array(env_idx, object_name)[0]
            pillar_state.update_property(f"frame:{object_name}:pose/position", object_pos[:3])
            pillar_state.update_property(f"frame:{object_name}:pose/quaternion", object_pos[3:])
            pillar_state.update_property(f"frame:{object_name}:pose/linear_velocity", object_vel[0])
            pillar_state.update_property(f"frame:{object_name}:pose/angular_velocity", object_vel[1])
        #else:
        #    pillar_state.update_property(f"frame:drawer:pose/position", [0.5, 0.1, 0.05])
        #    pillar_state.update_property(f"frame:drawer:pose/quaternion", [1,0,0,0])


        # Update Franka
        # Joints
        pillar_state.update_property(
            f"frame:{self.franka_name}:joint/position",
            self._franka.get_joints(env_idx, self.franka_name)[:7]
        )
        qdot = self._franka.get_joints_velocity(env_idx, self.franka_name)[:7]
        pillar_state.update_property(
            f"frame:{self.franka_name}:joint/velocity",
            qdot
        )

        # Gripper/fingers
        pillar_state.update_property(
            f"frame:{self.franka_name}:gripper/width",
            [self._franka.get_gripper_width(env_idx, self.franka_name)]
        )

        finger_transforms = self._franka.get_finger_transforms(env_idx, self.franka_name)
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
        ee_transform_np = transform_to_np(self._franka.get_ee_transform(env_idx, self.franka_name), format='wxyz')
        pillar_state.update_property(
            f"frame:{self.franka_name}:ee:pose/position",
            ee_transform_np[:3]
        )
        pillar_state.update_property(
            f"frame:{self.franka_name}:ee:pose/quaternion",
            ee_transform_np[3:]
        )
        ee_vels = self._franka.get_jacobian(env_idx, self.franka_name) @ qdot
        pillar_state.update_property(
            f"frame:{self.franka_name}:ee:pose/linear_velocity",
            ee_vels[:3]
        )
        pillar_state.update_property(
            f"frame:{self.franka_name}:ee:pose/angular_velocity",
            ee_vels[3:]
        )

        # Constants
        pillar_state.update_property(
            f"constants/gripper_mass",
            self.gripper_mass
        )
        if len(self._rod_names):
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
        if self._for_mde_training and "frame:drawer:pose/position" in pillar_state.get_prop_names():
            front_tf = self.get_drawer_transform(env_idx)
            drawer_pose_np = transform_to_np(front_tf, format="wxyz")
            pillar_state.update_property(f"frame:drawer:pose/position", drawer_pose_np[:3])
            pillar_state.update_property(f"frame:drawer:pose/quaternion", drawer_pose_np[3:])
            pillar_state.update_property("constants/drawer_dims", self._drawer_bottom_dims)
            self._pillar_states[env_idx] = State.create_from_serialized_string(
                pillar_state.get_serialized_string())  # defensive copy, please avoid bugs

        self._pillar_states[env_idx] = State.create_from_serialized_string(
            pillar_state.get_serialized_string())  # defensive copy, please avoid bugs

    def get_drawer_transform(self, env_idx):
        env_ptr = self._scene.env_ptrs[env_idx]
        drawer_rh = self._scene.gym.get_rigid_handle(env_ptr, "drawer", "front")
        front_tf = self._scene.gym.get_rigid_transform(env_ptr, drawer_rh)
        return front_tf

    def apply_ee_force_torque(self, action, env_idx, maintain_elbow=True, gripper_orient=False):
        ''' Expects action to be a (6,) ndarray - 3D forces and 3D torques
        '''
        J = self._franka.get_jacobian(env_idx, self.franka_name)
        tau = J.T @ action

        if maintain_elbow or gripper_orient:
            q_dot = self._franka.get_joints_velocity(env_idx, self.franka_name)[:7]
            JT_inv = np.linalg.pinv(J.T)

        Null = np.eye(7)
        if gripper_orient:
            ee_transform = self._franka.get_ee_transform(env_idx, self.franka_name)

            # find rotation that would bring ee_transform's z_axis to point vertically downwards
            R = quat_to_rot(ee_transform.r)
            q = quaternion.from_float_array(quat_to_np(ee_transform.r, format='wxyz'))
            ee_z = R[:, 2]
            ee_target_z = np.array([0, 0, -1])

            r_proj = angle_axis_between_axes(ee_z, ee_target_z)
            q_proj = quaternion.from_rotation_vector(r_proj)
            target_ee_quat = q_proj * q

            ee_target_transform = gymapi.Transform(
                p=ee_transform.p,
                r=np_quat_to_quat(target_ee_quat)
            )
            
            x_vel_ee = J @ q_dot

            tau_1 = compute_task_space_impedance_control(J, ee_transform, ee_target_transform, x_vel_ee, 
                                                        self._gripper_orient_Ks, self._gripper_orient_Ds)

            # Null = Null @ (np.eye(7) - J.T @ JT_inv)
            tau += Null @ tau_1

        if maintain_elbow:
            link_transforms = self._franka.get_links_transforms(env_idx, self.franka_name)
            elbow_transform = link_transforms[self._elbow_joint]

            u0 = vec3_to_np(link_transforms[0].p)[:2]
            u1 = vec3_to_np(link_transforms[-1].p)[:2]
            curr_elbow_xy = vec3_to_np(elbow_transform.p)[:2]
            goal_elbow_xy = project_to_line(curr_elbow_xy, u0, u1)
            elbow_target_transform = gymapi.Transform(
                p=gymapi.Vec3(goal_elbow_xy[0], goal_elbow_xy[1], elbow_transform.p.z + 0.1),
                r=elbow_transform.r
            )

            J_elb = self._franka.get_jacobian(env_idx, self.franka_name, target_joint=self._elbow_joint)
            x_vel_elb = J_elb @ q_dot

            tau_2 = compute_task_space_impedance_control(J_elb, elbow_transform, elbow_target_transform, x_vel_elb,
                                                         self._elbow_Ks, self._elbow_Ds)

            Null = Null @ (np.eye(7) - J.T @ JT_inv)
            tau += Null @ tau_2

        self._franka.apply_torque(env_idx, self.franka_name, tau)
        self._action_norms[env_idx] = np.linalg.norm(action)

    def set_gripper_width_target(self, width_target, env_idx):
        width_target = min(width_target, 0.0399)
        if not (width_target >= 0 and width_target <= 0.04):
            raise ValueError(f"Width error: width_target is {width_target} width_target must be between 0 and 0.04")
        self._franka.set_gripper_width_target(env_idx, self.franka_name, width_target/2) #2 needed in new version of IG for some reason


    def _compute_ee_pose(self):
        new_ee_poses = [None] * self.n_envs
        for env_idx in self._scene.env_idxs:
            ee_pose = self._franka.get_ee_transform(env_idx, self.franka_name, offset=True)
            new_ee_poses[env_idx] = ee_pose
        return new_ee_poses

    def _compute_affected_distance_change(self):
        """
        Returns: joint angle distance difference between current ee pose and last pose
        to be similar to previous version, hence why it's a little weird

        warning: mutates self._current_ee_poses! similar to energy version
        """
        old_ee_poses = self._current_ee_poses
        new_ee_poses = self._compute_ee_pose()
        distance_changes = np.zeros(self.n_envs)
        for env_idx in self._scene.env_idxs:
            distance_changes[env_idx] = np.linalg.norm(
                transform_to_np(old_ee_poses[env_idx], format="wxyz")[:3] \
                - transform_to_np(new_ee_poses[env_idx], format="wxyz")[:3]
            )  # only xyz, rotation is free just because it's hard to compare the two and then add them together, different scales
        self._current_ee_poses = new_ee_poses
        return distance_changes


    def _compute_costs(self):
        distance_change = self._compute_affected_distance_change()
        self._action_norms[:] = 0
        return distance_change


def make_franka_rod_test_env(cfg, setup_callbacks=[], init_state_kwargs = {}):
    return make_env_with_init_states(FrankaRodEnv, cfg, setup_callbacks=setup_callbacks, init_state_kwargs=init_state_kwargs)


def make_franka_test_env_already_holding(cfg, setup_callbacks=[]):
    return make_env_with_init_states(FrankaRodEnv, cfg, {'init_state_flag': "franka_holding"}, setup_callbacks=setup_callbacks)
