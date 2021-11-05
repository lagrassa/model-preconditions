import numpy as np
from isaacgym import gymapi
from pathlib import Path
from pillar_state import State

from isaacgym_utils.assets import GymURDFAsset
from isaacgym_utils.math_utils import transform_to_np, vec3_to_np, quat_to_np, np_to_vec3

from . import RealFrankaRodEnv, FrankaDrawerEnv
from .franka_env import FrankaRodEnv
from .utils import get_pose_pillar_state
from ..utils import yaw_from_np_quat, pillar_state_obj_to_transform, xyz_yaw_to_transform, r_flip_yz, make_shape

FIXED_ROTATION = [1,0,0,0]

class RealFrankaDrawerEnv(RealFrankaRodEnv):
    franka_name = "franka"
    object_name = "rod"
    goal_name = "goal"

    def __init__(self, cfg, setup_callbacks=[], baseboard=False):
        self._drawer_bottom_dims = cfg["env_props"]["drawer_bottom_dims"]
        super().__init__(cfg, make_drawer=1, setup_callbacks=[])
        self._asset_name_to_eps_arr["drawer"] =  [2*self._collision_eps, 2*self._collision_eps]
        pillar_state = self._pillar_states[0]
        pillar_state.update_property("constants/drawer_dims", self._drawer_bottom_dims)
        self._pillar_states[0] = pillar_state



    @staticmethod
    def pillar_state_to_sem_state(pillar_state, sem_state_obj_names, anchor_obj_name=None, ref_pillar_state=None):
        return FrankaDrawerEnv.pillar_state_to_sem_state(pillar_state, sem_state_obj_names, ref_pillar_state=ref_pillar_state)

    @staticmethod
    def sem_state_to_pillar_state(sem_state, ref_pillar_state, sem_state_obj_names, anchor_obj_name=None):
        return FrankaDrawerEnv.sem_state_to_pillar_state(sem_state, ref_pillar_state, sem_state_obj_names)


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
        return super_shapes + [drawer_shape,]
