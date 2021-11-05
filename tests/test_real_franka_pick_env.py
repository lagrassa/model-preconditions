from autolab_core import YamlConfig, RigidTransform

from plan_abstractions.envs.franka_env_real import RealFrankaRodEnv
from plan_abstractions.envs.pb_franka_env import FrankaKinematicsWorld 
from plan_abstractions.skills.franka_skills import Pick, LiftAndDrop
from plan_abstractions.utils import quat_to_np, ee_yaw_to_np_quat

def _goto_theta(env, theta):
    pose = env._franka.get_pose()
    parameters =  [0.4,0,0.04, theta]
    goal_quat = quat_to_np(ee_yaw_to_np_quat(parameters[3]), format="wxyz")
    des_rotation = RigidTransform.rotation_from_quaternion(goal_quat)
    pose.rotation = des_rotation
    env._franka.goto_pose(pose)

def test_env_init():
    cfg = YamlConfig("cfg/envs/franka_env_slippery_no_drop.yaml")
    env = RealFrankaRodEnv(cfg)

def test_params():
    cfg = YamlConfig("cfg/envs/franka_env_slippery_no_drop.yaml")
    env = RealFrankaRodEnv(cfg)
    import ipdb; ipdb.set_trace()
    _goto_theta(env, 0)

def test_detect():
    cfg = YamlConfig("cfg/envs/franka_env_slippery_no_drop.yaml")
    pb_world = FrankaKinematicsWorld(visualize=True, load_previous=False)
    env = RealFrankaRodEnv(cfg)
    #print(env.get_rod_transforms()[0].p)
    states = env.get_all_states(env_idxs=[0])
    pb_world.show_effects(states)
    input("PB world look okay?")


def test_execute_pick():
    skill = Pick()
    cfg = YamlConfig("cfg/envs/franka_env_slippery_no_drop.yaml")
    cfg["scene"]["dt"] = 0.1
    env = RealFrankaRodEnv(cfg)
    state = env.get_all_states(env_idxs=[0])
    param_gen = skill.generate_parameters(env, state[0], num_parameters=1)
    parameters = next(param_gen)
    T_plan_max = 1
    T_exec_max = 30 #this is timesteps...
    skill.execute(env, state, parameters, T_plan_max, T_exec_max, set_state = False)

def test_execute_lift_and_drop():
    pick_skill = Pick()
    drop_skill = LiftAndDrop()
    cfg = YamlConfig("cfg/envs/franka_env_slippery_no_drop.yaml")
    cfg["scene"]["dt"] = 0.1
    env = RealFrankaRodEnv(cfg)
    state = env.get_all_states(env_idxs=[0])
    pick_param_gen = pick_skill.generate_parameters(env, state[0], num_parameters=1)
    pick_parameters = next(pick_param_gen)
    T_plan_max = 1
    T_exec_max = 30 #this is timesteps...
    if pick_parameters[0][-1] < -1.5:
        pick_parameters[0][-1] += 3.1416
    if pick_parameters[0][-1] > 3:
        pick_parameters[0][-1] -= 3.1416
    pick_skill.execute(env, state, pick_parameters, T_plan_max, T_exec_max, set_state = False)
    state = env.get_all_states(env_idxs=[0])
    #drop_param_gen = drop_skill.generate_parameters(env, state[0], num_parameters=1)
    #drop_parameters = next(drop_param_gen)
    drop_parameters = [[0.43119273, -0.12742051, 0.18      , 0.05403884]]
    import ipdb; ipdb.set_trace()
    drop_skill.execute(env, state, drop_parameters, T_plan_max, T_exec_max, set_state = False)


#
#test_execute_pick() #make sure they end up in a reasonable place
#test_execute_pick()
test_execute_lift_and_drop()
#test_params()
