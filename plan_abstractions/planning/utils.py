from pathlib import Path
import logging

import numpy as np
from isaacgym_utils.math_utils import set_seed
from async_savers import AsyncSaver
from tqdm import tqdm

from ..skills import SkillDispatch
from ..utils import extract_effects_dict_as_arrays
from ..utils.plot_utils import plot_sem_plan, plot_error_histograms

from pillar_state import State


def play_and_save_plan(cfg, init_state, plan, task, visualize=False):
    """

    Args:
        plan: list of (skill, param) tuples

    Returns:
        env after the plan has been played on it

    """
    if visualize:
        cfg['scene']['gui'] = 1
        cfg['scene']['es'] = 1.5
    test_env = PushRodEnv(cfg)
    test_env.set_all_states([init_state] * test_env.n_envs, n_steps=10)
    successful_params = []
    states = []
    for (skill, params) in plan:
        curr_states = test_env.get_all_states()
        parameter_matrix = np.vstack([[params] for _ in range(test_env.n_envs)])
        skill.execute(test_env, curr_states, parameter_matrix, T_plan_max=cfg['skill']['T_plan_max'],
                      T_exec_max=cfg['skill']['T_exec_max'])
        states.append(test_env.get_state(0).get_serialized_string())
        successful_params.append(params)
    for state in test_env.get_all_states():
        assert task.is_goal_state(state)
    plan_data = {}
    plan_data["init_state"] = init_state.get_serialized_string()
    plan_data["params"] = np.vstack(successful_params)
    plan_data["states"] = states
    np.save(cfg["plan_data_file"], plan_data)
    return test_env

def remove_from_queues(node, queues_list):
    for queue in queues_list:
        if node in queue:
            queue.remove(node)

def add_to_queues(node, queues_list):
    for queue in queues_list:
        if node not in queue:
            queue.append(node)


def save_states_from_tree_via_skill_dispatch(env, planner, skills, savers, leaf_nodes_list = None,
                                                max_leaf_nodes=10, max_path_length=50, 
                                                use_planner_state_only=False, n_samples_per_path=1,
                                                T_plan_max=1, T_exec_max=1000):
    #leaf_nodes_list = planner.find_all_leaf_nodes()
    logging.info(f"Found leaf nodes: {len(leaf_nodes_list)}")

    if max_leaf_nodes is not None and len(leaf_nodes_list) > max_leaf_nodes:
        np.random.shuffle(leaf_nodes_list)
        logging.info(f"Choosing {max_leaf_nodes} from a total of {len(leaf_nodes_list)} nodes")
        leaf_nodes_list = leaf_nodes_list[:max_leaf_nodes]
   
    paths = [
        leaf_node.find_path_from_root()[:max_path_length]
        for leaf_node in leaf_nodes_list
    ]

    n_data_saved = 0
    if use_planner_state_only:
        for path in tqdm(paths, desc='Saving paths using planner states'):
            for path_step_idx in range(1, len(path)):
                prev_path_step = path[path_step_idx - 1]
                curr_path_step = path[path_step_idx]

                dict_to_save = {
                    "initial_states": [prev_path_step.pillar_state.get_serialized_string()],
                    "parameters": [curr_path_step.action_in.params],
                    "exec_data": {
                        k: [v]
                        for k, v in curr_path_step.exec_data.items()
                    },
                }

                skill_idx = curr_path_step.action_in.skill_idx
                savers[skill_idx].save(dict_to_save)
                n_data_saved += 1
    else:
        for exec_data in execute_paths(env, skills, paths, n_samples_per_path=n_samples_per_path, 
                                    T_plan_max=T_plan_max, T_exec_max=T_exec_max, randomize_dynamics=False):
            path_idx, all_data = exec_data['path_idx'], exec_data['all_data']
            path = paths[path_idx]
            if len(path) != len(all_data) + 1:
                raise ValueError("Planning path and amount of data received do not match. Some inconsistency.")

            for path_step_idx in range(1, len(path)):
                curr_path_step = path[path_step_idx]
                data = all_data[path_step_idx - 1]
                
                dict_to_save = {
                    "initial_states": [data['initial_states']],
                    "parameters": [data['parameters']],
                    "exec_data": {
                        k: [v]
                        for k, v in data['exec_data'].items()
                    },
                }

                skill_idx = curr_path_step.action_in.skill_idx
                print("Calling saver")
                savers[skill_idx].save(dict_to_save)
                n_data_saved += 1

    return n_data_saved


def collect_states_from_mcts_tree(env_cfg, mcts, save_dir, save_every=10, n_envs=None, gui=None, 
                                  seed=None, max_leaf_nodes=10, max_path_length=50, env=None, 
                                  use_planner_state_only=True):
    '''Collect pillar states from mcts tree.'''

    if seed is not None:
        set_seed(seed)
    
    if type(save_dir) is str:
        save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    leaf_nodes_list = mcts.find_all_leaf_nodes()
    logging.info(f"Found leaf nodes: {len(leaf_nodes_list)}")

    # TODO: Expand this code to work with multiple skills. For now this is fine.
    saver_by_skill_dict = dict()
    saver = AsyncSaver(save_dir, 'data', save_every=save_every)
    saver.start()

    if n_envs is not None:
        env_cfg['scene']['n_envs'] = n_envs
    if gui is not None:
        env_cfg['scene']['gui'] = gui

    # TODO: Change the env class or better read it from config.
    if env is None:
        env_class = PushRodEnv
        env = env_class(env_cfg)

    object_names = env.get_object_names()

    nth_init_state = 0
    if max_leaf_nodes is not None and len(leaf_nodes_list) > max_leaf_nodes:
        leaf_node_idxs = np.random.choice(np.arange(len(leaf_nodes_list)), max_leaf_nodes, replace=False)
        logging.info(f"Choosing {max_leaf_nodes} from a total of {len(leaf_nodes_list)} nodes")
        leaf_nodes_list = [leaf_nodes_list[i] for i in leaf_node_idxs]
        
    for leaf_node_idx, leaf_node in enumerate(leaf_nodes_list):
        path_to_root = leaf_node.find_path_from_root()

        begin_state_list = [env.__class__.pillar_state_to_sem_state(path_to_root[0].pillar_state, object_names)
                            for _ in range(env.n_envs)]
        gt_sem_state_list = [np.array(begin_state_list)]

        logging.info(f"Will collect data for leaf node: {leaf_node_idx+1}/{len(leaf_nodes_list)}, path_length: {len(path_to_root)}")

        # Get the current pillar state and desired goal params
        current_state = path_to_root[0].pillar_state
        last_state_list = [current_state] * env.n_envs
        env.set_all_states(last_state_list, n_steps=10)

        skills = mcts._skills
        # We have (state_0, None) -> (state_1, action_0) -> (state_2, action_1) -> ...
        for path_idx in range(len(path_to_root) - 1):
            if path_idx >= max_path_length:
                break

            action = path_to_root[path_idx + 1].action_in
            skill, params, param_type = skills[action.skill_idx], action.params, action.param_type

            # TODO: should we keep fixed dynamics across each path (maybe)? Anyways the dynamics
            # do not change much at all currently.
            gen_shape_props, gen_rb_props = None, None
            if env_cfg['env_props']['dynamics']['randomize']:
                gen_shape_props, gen_rb_props = env.randomize_dynamics_params()

            # Ideally, generated shape properties and generated rb_props should be the same
            # but we save both for redundancy.
            curr_shape_props = env.get_shape_props_for_objects()
            curr_rb_props = env.get_rigid_body_props_for_objects()

            parameters = np.vstack([[params] for _ in range(env.n_envs)])
            param_types = [param_type] * env.n_envs

            # Make sure that the skill parameters were generated for the appropriate skill.
            # This should also be modified further when we have planning with multiple skills. 
            # Although, even that should not be super hard.
            assert type(skill).__name__ == env_cfg['skill']['type']
            exec_data = skill.execute(env, last_state_list, parameters, env_cfg['skill']['T_plan_max'], 
                                      env_cfg['skill']['T_exec_max'], set_state=True)
            dict_to_save = {
                "initial_states": [initial_state.get_serialized_string() for initial_state in last_state_list],
                "parameters": parameters,
                "param_types": param_types,
                "exec_data": exec_data,
                "nth_init_state": nth_init_state,
                "current_shape_props": curr_shape_props,
                "current_rb_props": curr_rb_props,
            }
            if gen_shape_props is not None:
                dict_to_save["randomized_shape_props"] = gen_shape_props
                assert gen_rb_props is not None
                dict_to_save["randomized_rb_props"] = gen_rb_props

            end_pillar_states = [State.create_from_serialized_string(end_state) for end_state in exec_data['end_states']]
            if use_planner_state_only:
                last_state_list = [path_to_root[path_idx + 1].pillar_state for _ in range(env.n_envs)]
            else:
                last_state_list = end_pillar_states

            saver.save(dict_to_save)
            nth_init_state += 1

    saver.stop()
            

def debug_mcts_tree(env_cfg, mcts, use_planner_state_only=False, gui=True, debug_plan=None):
    if debug_plan is None:
        goal_closest_leaf_node = mcts.find_closest_to_goal_leaf_node()['node']
        path_to_goal = goal_closest_leaf_node.find_path_to_root()
    else:
        path_to_goal = debug_plan

    env_cfg['scene']['n_envs'] = 8
    env_cfg['scene']['gui'] = gui
    assert abs(env_cfg['scene']['es']) < 0.01, f"Environment spacing should be 0: {cfg['scene']['es']}"
    test_env = PushRodEnv(env_cfg)
    object_names = test_env.get_object_names()
    begin_state_list = [test_env.__class__.pillar_state_to_sem_state(path_to_goal[0].pillar_state, object_names)
                        for _ in range(test_env.n_envs)]
    gt_sem_state_list = [np.array(begin_state_list)]
    last_state_list = [path_to_goal[0].pillar_state for _ in range(test_env.n_envs)]
    skills = mcts._skills

    for path_idx in range(len(path_to_goal) - 1):
        # planner_node, _ = path_to_goal[path_idx]
        action = path_to_goal[path_idx + 1].action_in
        skill, param = skills[action.skill_idx], action.params
        gt_data = skill.gt_effects(
            test_env,
            last_state_list,
            [param for _ in range(test_env.n_envs)],
            mcts.cfg['T_plan_max'],
            mcts.cfg['T_exec_max'],
        )
        end_pillar_states = [State.create_from_serialized_string(end_state) for end_state in gt_data['end_states']]
        # NOTE: We are converting to SEM state without converting to some anchor space hence
        # these SEM states are global.
        gt_sem_states = [test_env.__class__.pillar_state_to_sem_state(eps, object_names)
                         for eps in end_pillar_states]
        gt_sem_state_list.append(np.array(gt_sem_states))

        if use_planner_state_only:
            last_state_list = [path_to_goal[path_idx + 1].pillar_state for _ in range(test_env.n_envs)]
        else:
            last_state_list = end_pillar_states

    # NOTE: Get the predicted SEM state list. We are converting to SEM state 
    # without converting to some anchor space hence these SEM states are global.
    sem_state_list = [test_env.__class__.pillar_state_to_sem_state(path_element.pillar_state, object_names)
                      for path_element in path_to_goal]
    plot_sem_plan(sem_state_list, gt_sem_state_list, test_env, use_fixed_axes_limits=True)


def show_node_error(cfg, nodes, env):
    end_state_errors = []
    cost_errors = []
    T_exec_errors = []
    T_plan_errors = []
    obj_names = ["pusher", "rod0", "rod1"]
    num_nodes_to_test = 10
    num_nodes_tested = 0
    children_expanded = []
    for node in nodes:
        for child in node.children:
            if child in children_expanded:
                continue
            skill = child.action_in[0]
            parameters = child.action_in[1]
            predicted_effects = skill.effects(node.state, parameters)
            parameters_matrix = np.vstack([parameters for _ in range(env.n_envs)])
            initial_states = [node.state for _ in range(env.n_envs)]
            actual_effects = skill.execute(env, initial_states, parameters_matrix,
                                           T_plan_max=cfg['skill']['T_plan_max'], T_exec_max=cfg['skill'['T_exec_max']])
            end_states_vec_pred, costs_pred, T_exec_pred, T_plans_pred = extract_effects_dict_as_arrays(env,
                                                                                                        predicted_effects,
                                                                                                        obj_names,
                                                                                                        initial_states)
            end_states_vec_actual, costs_actual, T_exec_actual, T_plans_actual = extract_effects_dict_as_arrays(env,
                                                                                                                actual_effects,
                                                                                                                obj_names,
                                                                                                                initial_states)
            end_state_errors.extend(np.linalg.norm(end_states_vec_pred - end_states_vec_actual, axis=1))
            cost_errors.extend(np.abs(costs_pred - costs_actual))
            T_exec_errors.extend(np.abs(T_exec_pred - T_exec_actual))
            T_plan_errors.extend(np.abs(T_plans_pred - T_plans_actual))
            children_expanded.append(child)
            num_nodes_tested += 1
        if num_nodes_tested > num_nodes_to_test:
            break

    plot_error_histograms(end_state_errors, cost_errors, T_plan_errors, T_exec_errors)


def visualize_plan(plan, root, env, cfg):
    state = root
    for (skill, params) in plan:
        data = skill.gt_effects(
            env,
            [state],
            params.reshape(1, -1),
            cfg['T_plan_max'],
            cfg['T_exec_max'],
        )
        child = data["end_states"][0]
        child = State.create_from_serialized_string(child)
        state = child


def execute_plan(env, plan_to_execute, skills, task, T_plan_max, T_exec_max, set_states=True):
    print("Executing plan")
    init_state = plan_to_execute[0].pillar_state
    if set_states:
        env.set_all_states([init_state] * env.n_envs, n_steps=10)
    else:
        env.reset()
    
    plan_exec_data = {
        'plan_to_execute': plan_to_execute,
        'skill_exec_data': [],
        'reached_goal': np.zeros(env.n_envs, dtype=bool)
    }
    
    for step in tqdm(plan_to_execute[1:], desc='Executing Plan'):
        skill = skills[step.action_in.skill_idx]
        curr_states = [env.get_sem_state()]
        parameter_matrix = np.vstack([[step.action_in.params] for _ in range(env.n_envs)])

        if hasattr(env, 'set_skill_params'):
            for env_idx in range(env.n_envs):
                env.set_skill_params(env_idx, parameter_matrix[env_idx, :])
        skill_exec_data = skill.execute(env, curr_states, parameter_matrix,
                                    T_plan_max=T_plan_max, T_exec_max=T_exec_max, set_state=False)

        # We do not need to save settled states for now.
        del skill_exec_data['initial_settled_states']
        plan_exec_data['skill_exec_data'].append(skill_exec_data)
    
    for env_idx, state in enumerate([env.get_sem_state()]):
        plan_exec_data['reached_goal'][env_idx] = task.is_goal_state(state)
        if not task.is_goal_state(state):
            import ipdb; ipdb.set_trace()
            task.is_goal_state(state)

    return plan_exec_data


def plot_exec_plan(env, plan_exec_data, task=None, show=True, save_filename=None):
    object_names = env.get_object_names()
    pred_sem_state_list = []
    gt_sem_state_list = []

    for (step_idx, step) in enumerate(plan_exec_data['plan_to_execute']):
        pred_pillar_state = step.pillar_state
        pred_sem_state = env.pillar_state_to_sem_state(pred_pillar_state, object_names)
        pred_sem_state_list.append(pred_sem_state)

        # initial state are all the same
        if step_idx == 0:
            gt_sem_state_list.append(np.array([pred_sem_state] * env.n_envs))
            continue

        skill_exec_data = plan_exec_data['skill_exec_data'][step_idx - 1]

        gt_sem_states = []
        for env_idx in range(env.n_envs):
            gt_pillar_state_str = skill_exec_data['end_states'][env_idx]
            gt_pillar_state = State.create_from_serialized_string(gt_pillar_state_str)
            gt_sem_state = env.pillar_state_to_sem_state(gt_pillar_state, object_names)
            gt_sem_states.append(gt_sem_state)
        gt_sem_state_list.append(np.array(gt_sem_states))

    plot_sem_plan(pred_sem_state_list, gt_sem_state_list, env, use_fixed_axes_limits=True,
                    show=show, save_filename=save_filename)


def eval_nodes_errors(nodes, env, skills, T_plan_max, T_exec_max, max_n_nodes_to_eval=10):
    nodes_errors = {
        'end_states': [],
        'costs': [],
        'T_exec': [],
        'T_plan': []
    }

    object_names = env.get_object_names()

    evaluated_nodes = set()
    for node in nodes:
        if len(evaluated_nodes) >= max_n_nodes_to_eval:
            break
        
        initial_states = [node.pillar_state] * env.n_envs

        for child in node.children:
            if len(evaluated_nodes) >= max_n_nodes_to_eval:
                break

            if child in evaluated_nodes:
                continue

            skill = skills[child.action_in.skill_idx]
            parameters = child.action_in.params
            parameters_matrix = np.vstack([parameters for _ in range(env.n_envs)])
            
            pred_effects = {
                'end_states': [child.pillar_state] * env.n_envs,
                'costs': np.ones(env.n_envs) * child.action_in.cost,
                'T_exec': np.ones(env.n_envs) * child.action_in.T_exec,
                'info_plan': [{
                    'T_plan': child.action_in.T_plan
                }] * env.n_envs
            }

            gt_effects = skill.execute(env, initial_states, parameters_matrix,
                                            T_plan_max=T_plan_max, T_exec_max=T_exec_max)
            
            end_states_vec_pred, costs_pred, T_exec_pred, T_plans_pred = \
                extract_effects_dict_as_arrays(env, pred_effects, object_names, initial_states, as_numpy=True)
            end_states_vec_gt, costs_gt, T_exec_gt, T_plans_gt = \
                extract_effects_dict_as_arrays(env, gt_effects, object_names, initial_states, as_numpy=True)
            nodes_errors['end_states'].append(np.linalg.norm(end_states_vec_pred - end_states_vec_gt, axis=1))
            nodes_errors['costs'].append(np.abs(costs_pred - costs_gt))
            nodes_errors['T_exec'].append(np.abs(T_exec_pred - T_exec_gt))
            nodes_errors['T_plan'].append(np.abs(T_plans_pred - T_plans_gt))

            evaluated_nodes.add(child)

    for k, v in nodes_errors.items():
        nodes_errors[k] = np.concatenate(v)

    return nodes_errors


def execute_paths(env, skills, paths, n_samples_per_path=1, T_plan_max=1, T_exec_max=400, randomize_dynamics=False):
    '''Execute multiple different paths i.e. sequence of skills and associated params.
    '''
    # duplicate paths for n_samples_per_path each
    all_paths = []
    all_paths_idx_to_paths_idx = []
    for path_idx, path in enumerate(paths):
        all_paths.extend([path] * n_samples_per_path)
        all_paths_idx_to_paths_idx.extend([path_idx] * n_samples_per_path)

    next_path_to_start_idx = 0
    n_paths_finished = 0
    running_paths = [None] * env.n_envs
    running_paths_idx = [0] * env.n_envs
    running_paths_step_idx = [0] * env.n_envs
    all_skill_data_buffer = [[] for _ in range(env.n_envs)]

    skill_dispatch = SkillDispatch(env, T_exec_max=T_exec_max)
    with tqdm(total=len(all_paths), desc='Executing paths...') as progress_bar:
        while n_paths_finished < len(all_paths):
            # dispatch as many skills as possible to available envs
            for env_idx in skill_dispatch.available_envs_idxs:
                # no more paths to start
                if next_path_to_start_idx >= len(all_paths):
                    break
                path_to_start = all_paths[next_path_to_start_idx]

                if len(path_to_start) == 1:
                    next_path_to_start_idx += 1
                    continue

                initial_state = path_to_start[0].pillar_state
                
                action = path_to_start[1].action_in
                skill = skills[action.skill_idx]
                parameters = action.params

                skill_exec_cb = skill.get_exec_cb(initial_state, parameters, T_plan_max=T_plan_max, set_state=True)
                skill_dispatch.set_skill_exec_cb(skill_exec_cb, env_idx)

                if randomize_dynamics:
                    env.randomize_dynamics_params(env_idxs=[env_idx])

                running_paths[env_idx] = path_to_start
                running_paths_idx[env_idx] = next_path_to_start_idx
                running_paths_step_idx[env_idx] = 1

                next_path_to_start_idx += 1

            # step all the skills currently in dispatch
            skill_dispatch.step()
            
            for env_idx in skill_dispatch.has_skill_env_idxs:
                # process skills that are done
                if skill_dispatch.is_skill_done(env_idx):
                    all_skill_data = skill_dispatch.get_all_skill_data(env_idx)
                    all_skill_data_buffer[env_idx].append(all_skill_data)

                    path = running_paths[env_idx]
                    running_paths_step_idx[env_idx] += 1

                    # all skills in this path have been executed, so yield their exec_datas and remove skill from dispatch
                    if running_paths_step_idx[env_idx] >= len(path):
                        all_skill_data_list = all_skill_data_buffer[env_idx]
                        path_idx = all_paths_idx_to_paths_idx[running_paths_idx[env_idx]]

                        all_skill_data_buffer[env_idx] = []
                        n_paths_finished += 1
                        progress_bar.update(1)
                        
                        skill_dispatch.remove_skill(env_idx)
                        assert len(all_skill_data_list) + 1 == len(path), "Path and exec data length do not match"

                        yield {
                            'path_idx': path_idx,
                            'all_data': all_skill_data_list,
                        }
                    # otherwise, dispatch the next skill in the path
                    else:
                        action = path[running_paths_step_idx[env_idx]].action_in
                        skill = skills[action.skill_idx]
                        parameters = action.params

                        # don't set state b/c executing skils in sequence
                        skill_exec_cb = skill.get_exec_cb(None, parameters, T_plan_max=T_plan_max, set_state=False)
                        skill_dispatch.set_skill_exec_cb(skill_exec_cb, env_idx)
