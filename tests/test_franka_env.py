import numpy as np
import pytest
from autolab_core import YamlConfig

from isaacgym_utils.math_utils import rpy_to_quat, set_seed
from plan_abstractions.envs import make_franka_rod_test_env, make_franka_test_env_already_holding
from plan_abstractions.envs.franka_drawer_env import FrankaDrawerEnv
from plan_abstractions.envs.pb_franka_env import FrankaKinematicsWorld


@pytest.fixture(scope="module")
def franka_pd_env_fixture():
    cfg = YamlConfig("cfg/collect_skill_data/franka_rods_pd.yaml")
    # cfg['scene']['gui']=1
    cfg['seed'] = 5  # 0 is bad for pick :P
    set_seed(cfg['seed'])
    env = make_franka_rod_test_env(cfg)
    yield env, cfg
    env._scene.close()


@pytest.fixture(scope="module")
def franka_lqr_free_env_fixture():
    cfg = YamlConfig("cfg/collect_skill_data/franka_rods_lqr_free_space.yaml")
    # cfg['seed'] = 5  # 0 is bad for pick :P
    set_seed(cfg['seed'])
    env = make_franka_rod_test_env(cfg)
    yield env, cfg
    env._scene.close()


@pytest.fixture(scope="module")
def franka_lqr_waypoints_env_fixture():
    cfg = YamlConfig("cfg/collect_skill_data/franka_rods_lqr_waypoints_xyz.yaml")
    cfg['seed'] = 5  # 0 is bad for pick :P
    set_seed(cfg['seed'])
    env = make_franka_rod_test_env(cfg)
    cfg['scene']['gui'] = False
    yield env, cfg
    env._scene.close()


@pytest.fixture(scope="module")
def franka_lqr_waypoints_xyzyaw_env_fixture():
    cfg = YamlConfig("cfg/collect_skill_data/franka_rods_lqr_waypoints_xyzyaw.yaml")
    cfg['scene']['gui'] = False
    cfg['seed'] = 1
    # np.random.seed(cfg['seed'])
    # cfg['seed'] = 5  # 0 is bad for pick :P
    env = make_franka_rod_test_env(cfg)
    yield env, cfg
    env._scene.close()


@pytest.fixture(scope="module")
def franka_pd_env_fixture_already_holding():
    cfg = YamlConfig("cfg/collect_skill_data/franka_rods_pd.yaml")
    # cfg['scene']['gui']=1
    cfg['seed'] = 2  # need room to release gripper. 2 was god
    set_seed(cfg['seed'])
    env = make_franka_test_env_already_holding(cfg)
    yield env, cfg
    env._scene.close()

def test_get_state(franka_pd_env_fixture):
    env, cfg = franka_pd_env_fixture
    cfg = YamlConfig("cfg/envs/franka_env.yaml")
    cfg["scene"]["gui"] = 0
    cfg["scene"]["n_envs"] = 1
    env = make_franka_rod_test_env(cfg)
    for state in env.get_all_states():
        print(state)


def test_set_state(franka_pd_env_fixture):
    env, cfg = franka_pd_env_fixture
    state_to_set = env.get_state(0)
    test_quat = rpy_to_quat([0, 0, 0.1])
    test_quat_np = [test_quat.w, test_quat.x, test_quat.y, test_quat.z]
    z_val_on_baseboard = env._collision_eps + env._baseboard_dims['sz']/2 + cfg["rod"]["dims"]['sz']/2. #hardcoded based on 
    state_to_set.update_property("frame:franka:joint/position", [0] * 7)
    state_to_set.update_property("frame:franka:ee:pose/position", [0.088, 0, 0.8326])
    state_to_set.update_property("frame:franka:ee:pose/quaternion", [0, 0.92, 0.38, 0])  # from IK...
    state_to_set.update_property("frame:franka:gripper/width", [0.02])
    state_to_set.update_property("frame:rod0:pose/position", [-0.2, 0.1, z_val_on_baseboard])
    state_to_set.update_property("frame:rod0:pose/quaternion", test_quat_np)
    state_to_set.update_property("frame:rod1:pose/position", [0.2, 0.08, z_val_on_baseboard])
    state_to_set.update_property("frame:rod1:pose/quaternion", [1, 0, 0, 0])

    # don't wait to settle b/c that'll change actual state in the sim
    env.set_all_states([state_to_set] * env.n_envs, n_steps=0)

    for state in env.get_all_states():
        assert env.states_similar_for_env(state_to_set, state)


def test_collision_checker(franka_pd_env_fixture):
    env, cfg = franka_pd_env_fixture
    base_pillar_state = env.get_state(0)
    body_names = ["rod0", "rod1"]
    asset_name_to_eps_arr = env.asset_name_to_eps_arr
    assert not env.is_in_collision(pillar_state=base_pillar_state, body_names=body_names, asset_name_to_eps_arr = asset_name_to_eps_arr)

    base_pillar_state.update_property(f"frame:rod0:pose/position", [0.2, 0, 0.007])
    base_pillar_state.update_property(f"frame:rod1:pose/position", [0.2, 0, 0.007])
    assert env.is_in_collision(pillar_state=base_pillar_state, body_names=body_names, asset_name_to_eps_arr = asset_name_to_eps_arr)


def test_state_generation(franka_pd_env_fixture):
    env, cfg = franka_pd_env_fixture
    cfg = YamlConfig("cfg/collect_skill_data/franka_rods_pd.yaml")
    set_seed(cfg['seed'])

    for init_state in env.generate_init_states(cfg, min_samples=5, max_samples=600):
        env.set_all_states([init_state] * env.n_envs, n_steps=10)

        for _ in range(30):  # visually inspect to make sure they look OK!
            env._scene.render()


def test_state_generation_holding(viz=False):
    cfg = YamlConfig("cfg/collect_skill_data/franka_rods_pd.yaml")
    cfg['scene']['gui'] = viz
    env = make_franka_rod_test_env(cfg)
    set_seed(cfg['seed'])

    for init_state in env.generate_init_states(cfg, init_state_flag = "franka_holding", min_samples=2, max_samples=4000):
        env.set_all_states([init_state] * env.n_envs, n_steps=10)
        if viz:
            for _ in range(400):  # visually inspect to make sure they look OK!
                env._scene.render()
    env._scene.close()

def test_drawer_env():
    cfg = YamlConfig("cfg/envs/franka_env_drawer.yaml")
    env = FrankaDrawerEnv(cfg)
    sem_state_obj_names = ["franka:ee", "rod0", "rod1"]
    np.random.seed(cfg['seed'])
    initial_state_gen = env.generate_init_states(cfg)
    initial_state_0 = next(initial_state_gen)
    sem_state = env.pillar_state_to_sem_state(initial_state_0, sem_state_obj_names, anchor_obj_name=None)
    pillar_state_new = env.sem_state_to_pillar_state(sem_state, initial_state_0, sem_state_obj_names, anchor_obj_name=None)
    assert env.states_similar_for_env(initial_state_0, pillar_state_new)



def test_sem_pillar_state_conversion(franka_pd_env_fixture):
    env, cfg = franka_pd_env_fixture

    sem_state_obj_names = ["franka:ee", "rod0", "rod1"]

    np.random.seed(cfg['seed'])
    initial_state_gen = env.generate_init_states(cfg)
    initial_state_0 = next(initial_state_gen)
    initial_state_1 = next(initial_state_gen)

    # vanilla state conversion
    sem_state = env.pillar_state_to_sem_state(initial_state_0, sem_state_obj_names, anchor_obj_name=None)
    pillar_state_new = env.sem_state_to_pillar_state(sem_state, initial_state_0, sem_state_obj_names, anchor_obj_name=None)
    assert env.states_similar_for_env(initial_state_0, pillar_state_new)

    # state conversion w/ anchor obj on the same state
    anchor_obj_name = 'franka:ee'
    sem_state_anchor_0 = env.pillar_state_to_sem_state(initial_state_0, sem_state_obj_names, anchor_obj_name=anchor_obj_name,
                                                     ref_pillar_state=initial_state_0)
    pillar_state_new_anchor_0 = env.sem_state_to_pillar_state(sem_state_anchor_0, initial_state_0, sem_state_obj_names,
                                                            anchor_obj_name=anchor_obj_name)
    assert env.states_similar_for_env(initial_state_0, pillar_state_new_anchor_0)

    # state conversion w/ different anchor obj states
    sem_state_anchor_1 = env.pillar_state_to_sem_state(initial_state_1, sem_state_obj_names, anchor_obj_name=anchor_obj_name,
                                                     ref_pillar_state=initial_state_0)
    pillar_state_new_anchor_1 = env.sem_state_to_pillar_state(sem_state_anchor_1, initial_state_0, sem_state_obj_names,
                                                            anchor_obj_name=anchor_obj_name)
    assert env.states_similar_for_env(initial_state_1, pillar_state_new_anchor_1, check_joints=False)
