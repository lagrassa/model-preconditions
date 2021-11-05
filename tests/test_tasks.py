import numpy as np
from autolab_core import YamlConfig

from isaacgym_utils.math_utils import set_seed, transform_to_np
from plan_abstractions.envs import make_franka_rod_test_env
from plan_abstractions.tasks.franka_tasks import MoveGripperToPose

from plan_abstractions.utils import transform_to_xyz_yaw



def test_gripper_at_pose_test():
    cfg = YamlConfig("cfg/collect_skill_data/franka_rods_pd.yaml")
    task_cfg = YamlConfig("cfg/tasks/franka_at_pose.yaml")["task"]
    env = make_franka_rod_test_env(cfg)
    test_idx = 0
    test_idx_tf = env.get_franka_ee_transform(test_idx)
    curr_xyz_yaw = transform_to_xyz_yaw(test_idx_tf)
    task_cfg['goal']['randomize'] = 0
    task_cfg['goal']["goal_xyz_yaw"] = np.r_[curr_xyz_yaw[:3], np.rad2deg(curr_xyz_yaw[3])]
    task = MoveGripperToPose(task_cfg)
    assert task.is_goal_state(env.get_state(test_idx))


def set_and_check_goal(task, state, goal_pos, tol):
    """
    utility function to set states such that they achieve a goal_pos within some tol
    by adding/subtracting the tol (so this only works in a few specific case)
    after doing that it checks if the goal is satisfied
    """
    rod0_pos = goal_pos[:2] + tol / 3
    rod1_pos = goal_pos[:2] - tol / 3

    original_rod_z = state.get_values_as_vec(["frame:rod0:pose/position"])[2]  # assume rods at same height
    rod0_pos = np.hstack([rod0_pos, original_rod_z])
    rod1_pos = np.hstack([rod1_pos, original_rod_z])

    state.update_property("frame:rod0:pose/position", rod0_pos)
    state.update_property("frame:rod1:pose/position", rod1_pos)
    assert task.is_goal_state(state)
