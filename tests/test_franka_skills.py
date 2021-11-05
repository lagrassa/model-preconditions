import numpy as np
import pytest
from autolab_core import YamlConfig
from pillar_state import State

from isaacgym_utils.math_utils import np_to_quat, rpy_to_quat, set_seed
from plan_abstractions.skills.franka_skills import *
#from plan_abstractions.envs import make_franka_rod_test_env, make_franka_test_env_already_holding
from plan_abstractions.envs.franka_env import make_franka_rod_test_env, make_franka_test_env_already_holding
from test_franka_env import  franka_pd_env_fixture, \
    franka_pd_env_fixture_already_holding, franka_lqr_waypoints_xyzyaw_env_fixture, franka_lqr_waypoints_env_fixture

from plan_abstractions.utils import get_object_names_in_pillar_state



def test_sample(franka_pd_env_fixture):
    env, cfg = franka_pd_env_fixture
    init_pillar_state = env.get_state(0)

    skill = FreeSpaceMoveFranka()
    param_array, param_types = next(skill.generate_parameters(env, init_pillar_state, num_parameters=env.n_envs, return_param_types=True))
    for params, param_type in zip(param_array, param_types):
        assert np.allclose(params.shape, skill.param_shape)
        assert skill.precondition_satisfied(init_pillar_state, params)



def test_pd_xyztheta():
    cfg = YamlConfig("cfg/envs/franka_env_slippery_no_drop.yaml")
    cfg["scene"]["gui"]=0
    env = make_franka_rod_test_env(cfg)
    skill = FreeSpaceMoveFranka()
    initial_states = env.get_all_states()
    param_gen = skill.generate_parameters(env, env.get_state(0), num_parameters=env.n_envs)
    parameters = next(param_gen)
    T_exec_max = 800
    data = skill.execute(env, initial_states, parameters, 1, T_exec_max)
    rod_names = [object_name for object_name in get_object_names_in_pillar_state(initial_states[0]) if
                    "franka" not in object_name]
    for env_idx in range(env.n_envs):
        franka_transform = env.get_franka_ee_transform(env_idx)
        rod_poses_before = [get_pose_pillar_state(initial_states[env_idx], rod_name) for rod_name in
                               rod_names]
        rod_poses_after = [
            get_pose_pillar_state(State.create_from_serialized_string(data["end_states"][env_idx]), rod_name) for
            rod_name in rod_names]
        pos = np.array([franka_transform.p.x, franka_transform.p.y, franka_transform.p.z])
        x_des = parameters[env_idx][0]
        y_des = parameters[env_idx][1]
        z_des = parameters[env_idx][2]
        des = np.array([x_des, y_des, z_des])
        assert np.linalg.norm(pos - des) < skill.position_tol
        assert_no_rod_movement(rod_poses_after, rod_poses_before)
        # TODO check theta
    env._scene.close()


def assert_no_rod_movement(rod_poses_after, rod_poses_before):
    for pose_before, pose_after in zip(rod_poses_before, rod_poses_after):
        assert np.allclose(pose_before[:3], pose_after[:3], atol=0.003)
        quat_before = quaternion.from_float_array(pose_before[3:])
        quat_after = quaternion.from_float_array(pose_after[3:])
        assert np.linalg.norm(angle_axis_between_quats(quat_before, quat_after)) < 0.01


def test_pd_pick_in_air():
    cfg = YamlConfig("cfg/collect_skill_data/franka_rods_pd.yaml")
    cfg['seed'] = 5 #5 for the test
    cfg['scene']['n_envs']= 5
    cfg['scene']['gui'] = 0
    #cfg['scene']['es'] = 0.4
    np.random.seed(cfg['seed'])
    env = make_franka_rod_test_env(cfg)
    #env, cfg = franka_pd_env_fixture
    skill = Pick(param_dist_cfg=cfg["skill"]["param_sampling_probabilities"])
    assert_obj_in_gripper_after_executing(cfg, env, skill)
    env._scene.close()


def assert_obj_in_gripper_after_executing(cfg, env, skill):
    release_skill = Release()
    initial_states = env.get_all_states()
    param_array = next(skill.generate_parameters(env, env.get_state(0), num_parameters=env.n_envs))
    body_names = ["franka:finger_left", "franka:finger_right", "rod0", "rod1"]
    for param_set in param_array:
        parameters = np.vstack([param_set for _ in range(env.n_envs)])
        relative_params = skill.parameters_to_relative_parameters(param_set, initial_states[0], "franka:ee")
        params_conversion = skill.relative_parameters_to_parameters(relative_params, initial_states[0],
                                                                    "franka:ee")
        assert np.allclose(param_set, params_conversion, atol=1e-5)
        data = skill.execute(env, initial_states, parameters, cfg['skill']['T_plan_max'], cfg['skill']['T_exec_max'])
        # env.is_in_collision(State.create_from_serialized_string(data["end_states"][0]), plot=True, body_names=body_names)
    release_dummy_parameters = [[] for _ in range(env.n_envs)]
    for state in env.get_all_states():
        assert release_skill.precondition_satisfied(state, release_dummy_parameters)

def test_open_drawer():
    cfg = YamlConfig("cfg/envs/franka_env_drawer.yaml")
    cfg['seed'] = 7 #other seed does not work for already holding placement
    cfg['scene']['n_envs'] = 1
    cfg['scene']['gui'] = 0
    np.random.seed(cfg['seed'])
    env = FrankaDrawerEnv(cfg)
    param_sampling_probs = {'object_centric':1, "relation_centric":0, "random":0, "task_oriented":0}
    skill = OpenDrawer(param_dist_cfg=param_sampling_probs)
    env._scene.step()
    initial_states = env.get_all_states()
    drawer_tf_before = env.get_drawer_transform(0)
    print(drawer_tf_before.p)
    env._drawer.set_joints(0, "drawer", [0.1])
    [env.step() for _ in range(100)]
    [env._scene.render() for _ in range(100)]
    drawer_tf_after = env.get_drawer_transform(0)
    print(drawer_tf_after.p)
    num_params = 5
    param_array = next(skill.generate_parameters(env, env.get_state(0), num_parameters=num_params))
    for param_set in param_array:
        parameters = np.vstack([param_set for _ in range(env.n_envs)])
        data = skill.execute(env, initial_states, parameters, 1, 1000)



if __name__ == "__main__":
    #test_pd_xyztheta()
    pass