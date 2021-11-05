from time import time
import numpy as np
from autolab_core import YamlConfig
from isaacgym_utils.math_utils import set_seed
from pillar_state import State

from plan_abstractions.skills.franka_skills import FreeSpaceMoveFranka
from plan_abstractions.envs import make_franka_rod_test_env
from plan_abstractions.planning.utils import execute_paths
from plan_abstractions.planning.common import Action, PathStep


def test_execute_paths(viz=False):
    cfg = YamlConfig("cfg/envs/franka_env_slippery_no_drop.yaml")
    cfg['scene']['es'] = 0
    cfg['scene']['n_envs'] = 10
    cfg['scene']['gui'] = viz
    set_seed(0)

    env = make_franka_rod_test_env(cfg)
    skill = FreeSpaceMoveFranka()
    skills = [skill]
    skill_idx = 0 # only one skill

    n_paths = 3
    path_length = 4
    T_plan_max = 1
    T_exec_max = 300

    initial_state_gen = env.generate_init_states(cfg, min_samples=n_paths)
    initial_states = []
    for _ in range(n_paths):
        initial_state = next(initial_state_gen)
        initial_states.append(initial_state)

    # construct paths by running in IG synchronously
    sync_exec_start_time = time()
    paths = []
    paths_exec_data_sync = []
    for path_idx in range(n_paths):
        current_state = initial_states[path_idx]
        path = [PathStep(current_state, None)]

        path_exec_data_list = []
        for path_step_idx in range(path_length):
            param_array_gen = skill.generate_parameters(env, current_state, return_param_types=True)
            parameters, param_types = next(param_array_gen)

            exec_data = skill.execute(env, 
                                    [current_state] * env.n_envs, 
                                    [parameters[0]] * env.n_envs, 
                                    T_plan_max=T_plan_max, 
                                    T_exec_max=T_exec_max, 
                                    set_state=path_step_idx == 0
                        )
            path_exec_data_list.append(exec_data)
            
            current_state = env.get_state(0)
            action_in = Action(skill_idx, parameters[0], param_types[0], 
                            exec_data['costs'][0], exec_data['T_exec'][0], 
                            exec_data['info_plan'][0]
                        )
            path.append(PathStep(current_state, action_in))

        paths_exec_data_sync.append(path_exec_data_list)
        paths.append(path)
    sync_exec_time = time() - sync_exec_start_time
    sync_exec_time_per_path = sync_exec_time / len(paths)

    # run constructed paths asynchronously
    async_exec_start_time = time()
    n_samples_per_path = 5
    paths_exec_data_async = []
    for path_exec_data_list in execute_paths(env, skills, paths, n_samples_per_path, 
                                    T_plan_max=T_plan_max, T_exec_max=T_exec_max, 
                                    # disable for testing
                                    randomize_dynamics=False):
        paths_exec_data_async.append(path_exec_data_list)
    async_exec_time = time() - async_exec_start_time
    async_exec_time_per_path = async_exec_time / len(paths_exec_data_async)

    print(f'Sync exec took {sync_exec_time_per_path:.1f}s/path | Async exec took {async_exec_time_per_path:.1f}s/path')
    print(f'Speed-up factor of {sync_exec_time_per_path/async_exec_time_per_path:.1f}x')

    # exec_data_sync and exec_data_async should be similar
    # there are some small errors b/c IG is not perfectly deterministic
    T_exec_asserts, terminated_asserts, costs_asserts, state_asserts = [], [], [], []
    for path_async in paths_exec_data_async:
        path_idx = path_async['path_idx']
        exec_data_list = [d['exec_data'] for d in path_async['all_data']]

        for exec_data_sync, exec_data_async in zip(paths_exec_data_sync[path_idx], exec_data_list):
            T_exec_asserts.append(np.isclose(exec_data_sync['T_exec'][0], exec_data_async['T_exec'], atol=30))
            terminated_asserts.append(np.isclose(exec_data_sync['terminated'][0], exec_data_async['terminated']))
            costs_asserts.append(np.isclose(exec_data_sync['costs'][0], exec_data_async['costs'], atol=1))

            end_state_sync = State.create_from_serialized_string(exec_data_sync['end_states'][0])
            end_state_async = State.create_from_serialized_string(exec_data_async['end_states'])
            state_asserts.append(env.states_similar_for_env(end_state_sync, end_state_async, position_tol=0.02, angle_tol=0.2))

    assert np.mean(T_exec_asserts) >= 0.7
    assert np.mean(terminated_asserts) >= 0.7
    assert np.mean(costs_asserts) >= 0.7
    assert np.mean(state_asserts) >= 0.7


if __name__ == '__main__':
    test_execute_paths(viz=True)
