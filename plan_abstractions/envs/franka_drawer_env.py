import numpy as np
from isaacgym import gymapi
from pathlib import Path
from pillar_state import State

from isaacgym_utils.assets import GymURDFAsset
from isaacgym_utils.math_utils import transform_to_np, vec3_to_np, quat_to_np, np_to_vec3
from .franka_env import FrankaRodEnv
from .utils import get_pose_pillar_state, set_pose
from ..utils import yaw_from_np_quat, pillar_state_obj_to_transform, xyz_yaw_to_transform, r_flip_yz, make_shape, \
    plot_shapes

FIXED_ROTATION = [1,0,0,0]

class FrankaDrawerEnv(FrankaRodEnv):
    franka_name = "franka"
    object_name = "rod"
    goal_name = "goal"

    def __init__(self, cfg, setup_callbacks=[], for_mde_training=False, baseboard=False):
        self._drawer_bottom_dims = cfg["env_props"]["drawer_bottom_dims"]
        super().__init__(cfg, setup_callbacks=setup_callbacks + [self.add_real_drawer_cb], for_mde_training=for_mde_training)
        self._asset_name_to_eps_arr["drawer"] =  [2*self._collision_eps, 2*self._collision_eps]

    @staticmethod
    def pillar_state_to_sem_state(pillar_state, sem_state_obj_names, anchor_obj_name=None, ref_pillar_state=None):
        super_state = FrankaRodEnv.pillar_state_to_sem_state(pillar_state, sem_state_obj_names, anchor_obj_name=anchor_obj_name, ref_pillar_state=ref_pillar_state)
        if len(sem_state_obj_names) == 4: #includes drawer already. get rid of it.
            super_state = super_state[:4*(len(sem_state_obj_names)-1)] #treat drawer separately
        gripper_width = pillar_state.get_values_as_vec(["frame:franka:gripper/width"])[0]
        drawer_sem_state = np.zeros(4,)  # xyz yaw
        drawer_sem_state[-1] = gripper_width
        drawer_sem_state[:3] = get_pose_pillar_state(pillar_state, "drawer")[:3]
        return np.hstack([super_state, drawer_sem_state])

    @staticmethod
    def sem_state_to_pillar_state(sem_state, ref_pillar_state, sem_state_obj_names, anchor_obj_name=None):
        sem_state_obj_names_minus_drawer = sem_state_obj_names.copy()
        if "drawer" in sem_state_obj_names_minus_drawer:
            sem_state_obj_names_minus_drawer.remove("drawer")
        state = FrankaRodEnv.sem_state_to_pillar_state(sem_state, ref_pillar_state, sem_state_obj_names_minus_drawer, anchor_obj_name=anchor_obj_name)
        name = "drawer"
        state.set_values_from_vec([f'frame:{name}:pose/position'], sem_state[12:15])
        state.set_values_from_vec([f'frame:{name}:pose/quaternion'], FIXED_ROTATION)
        state.set_values_from_vec([f'frame:franka:gripper/width'], [sem_state[-1]])
        return state


    def generate_init_states(self, cfg, min_samples=10, max_samples=1000, init_state_flag=None,
                             choose_from_multiple_rod_configs=False, return_init_state_info=False):
        super_gen = super().generate_init_states(cfg, min_samples=min_samples,
                                                 max_samples=max_samples,
                                                 init_state_flag=init_state_flag)
        for potential_pillar_state in super_gen:
            fixed_y_val = 0.11 #formerly 0.14
            potential_pillar_state.update_property("frame:drawer:pose/position",
                                                   [self._drawer_base_transform.p.x, fixed_y_val, 0.03])
            potential_pillar_state.update_property("frame:drawer:pose/quaternion", [1, 0, 0, 0])
            potential_pillar_state.update_property("constants/drawer_dims", self._drawer_bottom_dims)
            yield potential_pillar_state

    def set_state(self, pillar_state, env_idx, n_steps=0):
        super().set_state(pillar_state, env_idx, n_steps=n_steps)
        if "frame:drawer:pose/position" in pillar_state.get_prop_names():
            drawer_pose = get_pose_pillar_state(pillar_state, "drawer")
            drawer_base_tform = self._drawer_base_transform
            joint_val = drawer_base_tform.p.y - 0.19 - drawer_pose[1]  # magic number is the length of the drawer
            curr_pose = self._drawer.get_rb_poses_as_np_array(env_idx, self._drawer_name)[0]
            curr_pose[0] = drawer_pose[0]
            new_pose = curr_pose.copy()
            set_pose(new_pose, self._drawer_name, self._drawer, env_idx)
            self._drawer.set_joints(env_idx, "drawer", joint_val)

    def _update_pillar_state(self, env_idx):
        super()._update_pillar_state(env_idx)
        pillar_state = self._pillar_states[env_idx]

        front_tf = self.get_drawer_transform(env_idx)
        drawer_pose_np = transform_to_np(front_tf, format="wxyz")
        pillar_state.update_property(f"frame:drawer:pose/position", drawer_pose_np[:3])
        pillar_state.update_property(f"frame:drawer:pose/quaternion", drawer_pose_np[3:])
        pillar_state.update_property("constants/drawer_dims", self._drawer_bottom_dims)
        self._pillar_states[env_idx] = State.create_from_serialized_string(
            pillar_state.get_serialized_string())  # defensive copy, please avoid bugs

    def get_drawer_transform(self, env_idx):
        env_ptr = self._scene.env_ptrs[env_idx]
        drawer_rh = self._scene.gym.get_rigid_handle(env_ptr, "drawer", "front")
        front_tf = self._scene.gym.get_rigid_transform(env_ptr, drawer_rh)
        return front_tf

    def states_similar_for_env(self, state1, state2, yaw_only=True, check_joints=True, position_tol=5e-3):
        super_result = super().states_similar_for_env(state1, state2, yaw_only=yaw_only, check_joints=check_joints, position_tol=position_tol)
        if not super_result:
            return False
        drawer1 = get_pose_pillar_state(state1, "drawer")
        drawer2 = get_pose_pillar_state(state2, "drawer")
        return np.allclose(drawer1[:3], drawer2[:3], atol=position_tol)

    @classmethod
    def _make_collision_shapes(cls, pillar_state, body_names=None, plot=False, asset_name_to_eps_arr=None):
        super_shapes = FrankaRodEnv._make_collision_shapes(pillar_state, body_names=body_names, plot=plot, asset_name_to_eps_arr=asset_name_to_eps_arr)
        if "constants/drawer_dims" not in pillar_state.get_prop_names():
            return super_shapes
        drawer_dims = pillar_state.get_values_as_vec(["constants/drawer_dims"])
        edge_pose = np.array(get_pose_pillar_state(pillar_state, "drawer"))
        drawer_center_pose = edge_pose.copy()
        drawer_center_pose[1] += drawer_dims[1]/2
        drawer_shape = make_shape(drawer_center_pose, FIXED_ROTATION, drawer_dims, asset_name_to_eps_arr["drawer"])
        all_shapes = super_shapes + [drawer_shape,]
        #plot_shapes(all_shapes, ["a"]*4)
        return all_shapes
