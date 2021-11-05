import numpy as np
from autolab_core import YamlConfig
from isaacgym_utils.math_utils import set_seed

from plan_abstractions.skills import SkillDispatch, merge_exec_data
from plan_abstractions.skills.franka_skills import FreeSpaceMoveFranka
from plan_abstractions.envs import make_franka_rod_test_env
from pillar_state import State


def test_skill_dispatch_freespace_pd_move(viz=False):
    cfg = YamlConfig("cfg/envs/franka_env.yaml")
    cfg['scene']['es'] = 0 # needed to get test results more predictable
    cfg['scene']['n_envs'] = 30
    cfg['scene']['gui'] = viz
    set_seed(0)

    env = make_franka_rod_test_env(cfg)
    skill = FreeSpaceMoveFranka()

    # Generate different init state and params for each env
    initial_state_gen = env.generate_init_states(cfg, min_samples=env.n_envs, max_samples=int(1e9))
    initial_states = []
    parameters = []
    for _ in range(env.n_envs):
        initial_state = next(initial_state_gen)
        initial_states.append(initial_state)

        param_array_gen = skill.generate_parameters(env, initial_state)
        parameter = next(param_array_gen)[0]
        parameters.append(parameter)
    T_exec_max = 500
    # The original execute - runs all skills synchronously
    exec_data_sync = skill.execute(env, initial_states=initial_states, parameters=parameters, T_plan_max=1, T_exec_max=T_exec_max)

    # Get new exec data in an "async" fashion
    skill_dispatch = SkillDispatch(env, T_exec_max=T_exec_max)

    for env_idx in range(env.n_envs):
        skill_exec_cb = skill.get_exec_cb(initial_states[env_idx], parameters[env_idx], T_plan_max=1)
        skill_dispatch.set_skill_exec_cb(skill_exec_cb, env_idx)

    exec_data_async = [None] * env.n_envs
    while not skill_dispatch.all_skills_done:
        skill_dispatch.step()
        for env_idx in range(env.n_envs):
            if exec_data_async[env_idx] is None and skill_dispatch.is_skill_done(env_idx):
                exec_data_async[env_idx] = skill_dispatch.get_exec_data(env_idx)
    exec_data_async = merge_exec_data(exec_data_async)

    # exec_data_sync and exec_data_async should be similar
    # there are some small errors b/c IG is not perfectly deterministic
    assert np.isclose(exec_data_sync['T_exec'], exec_data_async['T_exec'], atol=10).mean() >= 0.8
    assert np.isclose(exec_data_sync['terminated'], exec_data_async['terminated']).mean() >= 0.8
    assert np.isclose(exec_data_sync['costs'], exec_data_async['costs'], atol=1).mean() >= 0.8

    state_similars = []
    for end_state_str_sync, end_state_str_async in zip(exec_data_sync['end_states'], exec_data_async['end_states']):
        end_state_sync = State.create_from_serialized_string(end_state_str_sync)
        end_state_async = State.create_from_serialized_string(end_state_str_async)
        state_similars.append(env.states_similar_for_env(end_state_sync, end_state_async, position_tol=0.02, angle_tol=0.2, velocity_tol = 0.3))
    print(f"Mean of similar states: {np.mean(state_similars)}")
    assert np.mean(state_similars) >= 0.8
    
    env._scene.close()


if __name__ == '__main__':
    test_skill_dispatch_freespace_pd_move(viz=False)
