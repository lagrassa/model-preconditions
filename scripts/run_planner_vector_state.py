from isaacgym import gymapi as _ # needed to get imports working
import time

import os
from pathlib import Path
from pickle import dump
import logging
import hydra
from isaacgym_utils.math_utils import set_seed
import numpy as np

from plan_abstractions.envs import *
from plan_abstractions.tasks import *
from plan_abstractions.skills import *
from plan_abstractions.planning import *
from tqdm import trange, tqdm
from plan_abstractions.planning.utils import execute_plan, plot_exec_plan, eval_nodes_errors
from plan_abstractions.utils import make_save_dir_and_get_plan_results_filename
from plan_abstractions.utils.plot_utils import plot_error_histograms
from plan_abstractions.utils.planner_utils import planner_graph_to_dot


logger = logging.getLogger(__name__)

def get_and_create_transition_fn(cfg):
        save_transitions = cfg["save_transition_cfg"].get("save_transitions", True)
        transition_data_dirpath = cfg["save_transition_cfg"]["dirpath"]
        transition_data_fn = None
        if save_transitions:
            if not os.path.isdir(transition_data_dirpath):
                os.mkdir(transition_data_dirpath)
            for i in range(2000):
                fn = os.path.join(transition_data_dirpath, f"transition_data_{i}.npy")
                if not os.path.exists(fn):
                    transition_data_fn = fn
                    return transition_data_fn


def run_planner_on_task(cfg, env, skills, task, save_dir, init_state, eval_env = None,current_transition_records=None, transition_data_fn=None):
    logger.info('Running planner...')

    planner_type = cfg['planner']['type']
    load=False
    if load:
        from pickle import load
        fn_root = "/mnt/hdd/jacky/plan_abstractions/planner/push_rod/default/2021-04-22_18-11-39/try_000/"
        fn_root = "/mnt/hdd/jacky/plan_abstractions/planner/push_rod/default/2021-04-26_17-16-28/try_009/"
        fn_root = "/mnt/hdd/jacky/plan_abstractions/planner/push_rod/default/2021-05-25_20-16-59/try_003/"
        plan_fn =  fn_root + "plan.pkl"
        planner_fn = fn_root + "planner.pkl"
        #planner = load(open(planner_fn, 'rb'))
        planner = eval(planner_type)(task, env, skills, cfg['planner'][planner_type], root_dir =f"{cfg['original_cwd']}/")
        #plan = planner.load(plan_fn, task, env, skills, cfg['planner'][planner_type])  # planner.load(fn, task, env, skills, cfg['planner'][planner_type])
        goal_node, plan = load(open(plan_fn, 'rb'))
    else:
        planner = eval(planner_type)(task, env, skills, cfg['planner'][planner_type], root_dir=f"{cfg['original_cwd']}/")
        start_time = time.time()
        init_state = env.get_sem_state()
        goal_node, plan = planner.search(init_state, timeout=cfg['planner']['timeout'],
                         max_search_depth=cfg['planner']['max_search_depth'])
        elapsed_time = time.time() - start_time
    #planner_fn ="/mnt/hdd/jacky/plan_abstractions/planner/push_rod/default/2021-04-22_17-19-41/try_000/planner.pkl"
    #planner = np.load(planner_fn, allow_pickle=True)
    found_plan = len(plan) > 0
    print("Foundp lan? ", found_plan)
    if not found_plan:
        import ipdb; ipdb.set_trace()
    logger.info(f'Planner done. Found plan? {found_plan}')

    # Save initial state to replay during debug_planner
    data_to_save = {
        'found_plan': found_plan}
    if isinstance(init_state, np.ndarray):
        data_to_save['init_state'] = init_state
    else:
        data_to_save['init_state']= init_state.get_serialized_string()
    env_data_path = save_dir / 'env_data.pkl'
    with open(env_data_path, 'wb') as f:
        dump(data_to_save, f)
        logging.info(f"Did save env data: {env_data_path}")

    logger.info('Saving plan...')
    planner.save_plan(plan, save_dir / 'plan.pkl')
    if found_plan:
        plan_to_execute = plan
        distance_to_goal_state = task.evaluate(plan[-1].pillar_state)
        debug_plan_str = task.pretty_print_with_reference_to_pillar_state(plan[-1].pillar_state)
    else:
        #goal_closest_leaf_node = planner.find_closest_to_goal_leaf_node()['node']
        #distance_to_goal_state = task.evaluate(goal_closest_leaf_node.pillar_state)
        #debug_plan_str = task.pretty_print_with_reference_to_pillar_state(goal_closest_leaf_node.pillar_state)
        #if goal_closest_leaf_node == planner.root_node:
        #    logging.info("Closest node to goal is leaf node. There is some issue.")
        #plan_to_execute = goal_closest_leaf_node.find_path_from_root()
        plan_to_execute = []
        debug_plan_str = ""
    max_gt_path_length_to_execute = cfg.get('max_gt_path_length_to_execute', 20)
    if len(plan_to_execute) > max_gt_path_length_to_execute:
        logging.info(f"Length of plan to execute: {len(plan_to_execute)}. Too long will trim to {max_gt_path_length_to_execute}.")
        plan_to_execute = plan_to_execute[:max_gt_path_length_to_execute]

    #planner_graph_to_dot(planner.root_node,
    #                     plan_to_execute,
    #                     save_dir / 'search_tree.dot',
    #                     env.planner_state_to_viz_string,
    #                     env.states_similar_for_env,
    #                     max_depth=5)

    logger.info(f'Length of plan to execute: {len(plan_to_execute) - 1}')

    logger.info('Executing plan...')
    plan_exec_reached_goal = False
    if eval_env is None:
        eval_env = env
        T_exec_max = cfg['planner'][planner_type]['T_exec_max']
    else:
        T_exec_max = cfg['T_exec_max_real']
    if len(plan_to_execute) > 0 and found_plan:
        #input("Ready to execute plan?")
        plan_exec_data = execute_plan(eval_env, plan_to_execute, skills, task,
                                      cfg['planner'][planner_type]['T_plan_max'], 
                                      T_exec_max=T_exec_max, set_states=0)
        print("Reached goal",plan_exec_data['reached_goal'])
        logger.info(f"These envs reached goals: {np.argwhere(plan_exec_data['reached_goal']).flatten()}")
        logger.info(f"Num model evals: {planner.num_model_evals}")
        update_transition_records(plan_exec_data, current_transition_records, transition_data_fn)
        with open(save_dir / 'plan_exec_data.pkl', 'wb') as f:
            dump(plan_exec_data, f)
        
        if type(plan_exec_data.get('reached_goal')) is np.ndarray and len(plan_exec_data['reached_goal']) > 0:
            reached_goal_envs = plan_exec_data['reached_goal']
            plan_exec_reached_goal = np.sum(reached_goal_envs)/len(reached_goal_envs) >= 0.6

        #plot_exec_plan(env, plan_exec_data, show=cfg['vis'], save_filename=save_dir / 'exec_plan.png')
    else:
        plan_exec_reached_goal = []
        distance_to_goal_state = -1

    info_dict = {
        'planner': planner,
        'plan_found': found_plan,
        'elapsed_time' : elapsed_time,
        'plan_to_execute': plan_to_execute,
        'plan_exec_reached_goal': plan_exec_reached_goal,
        'distance_to_goal_state': distance_to_goal_state,
        'debug_plan_str': debug_plan_str,
    }
    info_dict["num_model_evals"] = planner.num_model_evals
    info_dict["model_type_per_skill_idx"] = planner.model_type_per_skill_idx
    return info_dict


def update_transition_records(plan_exec_data, current_transition_records, transition_data_fn):
    skill_exec_data_list = plan_exec_data['skill_exec_data']
    #assuming just 1 env idx 
    data_item_names = list(current_transition_records.values())[0].keys()
    env_idx = 0
    for skill_exec_data_point in skill_exec_data_list:
        skill_name = skill_exec_data_point["skill_name"]
        if current_transition_records[skill_name]['init_states'] is None:
            #Need to initialize
            for name in data_item_names:
                current_transition_records[skill_name][name] = skill_exec_data_point[name][env_idx]
        else:
            #Can also pre-allocate, but this is nice because this process often crashes/needs to be interrupted
            for name in data_item_names:
                current_transition_records[skill_name][name] = np.vstack([current_transition_records[skill_name][name],skill_exec_data_point[name][env_idx]])
    np.save(transition_data_fn, current_transition_records)


# @hydra.main(config_path='../cfg/planner', config_name='mcts_push_rods_lqr.yaml')
@hydra.main(config_path='../cfg/planner', config_name='solve_push_one_rod_franka_4_skills.yaml')
def main(cfg):
    print("Setting seed to be", cfg['seed'])
    set_seed(cfg['seed'])
    save_dir = Path(os.getcwd())
    logger.info(f'Saving data to {save_dir}')
    if 'scene' in cfg.keys():
        cfg['scene']['gui'] = cfg['vis']
        if not cfg['vis']: #avoid instability
            cfg['scene']['es'] = 0
    cfg['original_cwd'] = hydra.utils.get_original_cwd()

    total_goals_to_reach_per_iter = 1
    num_initial_states_to_test = cfg['n_init_states']
    total_goals_to_reach = num_initial_states_to_test * total_goals_to_reach_per_iter
    current_transition_records = {}
    for skill_type, skill_cfg in cfg['skills'].items():
        current_transition_records[skill_type] = {'init_states':None,'params':None,'end_states':None}


    logger.info('Making env, task, and skills...')

    task_type = cfg["task"]["type"]
    task = eval(task_type)(cfg["task"], real_robot=cfg["env_props"].get("real_robot"))
    task_specific_env_callbacks = task.setup_callbacks

    init_state_kwargs = {}
    #env = make_env_with_init_states(eval(cfg['env']), cfg, setup_callbacks=task_specific_env_callbacks,
    #                                **init_state_kwargs)
    env = make_env(eval(cfg['env']), cfg)
    if cfg["env_props"].get("real_robot", False):
        plan_env = env
        real_env = eval(cfg['real_env'])(cfg)
        task.set_detector()
        real_robot = True
        def init_state_gen():
            while True:
                real_env.reset_to_viewable()
                input("Reset rod. OK?")
                init_state = real_env.get_state(0, update_rod_poses=True)
                yield init_state
        initial_state_generator = init_state_gen()
        real_robot=True
    else:
        real_robot = False
        def init_state_gen():
            for _ in range(cfg['n_init_states']):
                env.reset()
                yield None
        initial_state_generator = init_state_gen()

    skills = []
    for skill_type, skill_cfg in cfg['skills'].items():
        if skill_cfg['use']:
            skills.append(eval(skill_type)(sem_cfg = None, #skill_cfg.get("sem_cfg", None),
                                           deviation_cfg=skill_cfg.get('deviation_cfg', None),
                                           models_cfg = skill_cfg.get('high_level_models', None),
                                           real_robot=real_robot,
                                           param_dist_cfg=skill_cfg['param_sampling_probabilities']))

    planner_goals_reached, actual_goals_reached = 0, 0
    num_goals_tested = 0
    elapsed_times = []
    plan_results_list = []
    num_model_evals_per_sec_total= []
    plan_results_dirname = os.path.join(cfg["data_root_dir"], "plan_results/")
    plan_results_filename = make_save_dir_and_get_plan_results_filename(plan_results_dirname)
    if "save_transition_cfg" in cfg.keys():
        transition_data_fn = get_and_create_transition_fn(cfg)
        save_transitions = cfg["save_transition_cfg"].get("save_transitions", True)
    else:
        save_transitions = False
        transition_data_fn=None
    for init_state in tqdm(initial_state_generator, total=cfg['n_init_states'], desc='Init states'):
        ith_goal_save_dir = save_dir / f'try_{num_goals_tested:03}'
        ith_goal_save_dir.mkdir()
        if real_robot:
            plan_results = run_planner_on_task(cfg, plan_env, skills, task, ith_goal_save_dir, init_state, eval_env = real_env, current_transition_records=current_transition_records, transition_data_fn=transition_data_fn)
        else:
            plan_results = run_planner_on_task(cfg, env, skills, task, ith_goal_save_dir, init_state, current_transition_records=current_transition_records, transition_data_fn=transition_data_fn)
        plan_results_list.append(plan_results.copy())
        if plan_results['plan_found']:
            planner_goals_reached += 1
        if plan_results['plan_found'] and plan_results['plan_exec_reached_goal']:
            actual_goals_reached += 1
        if plan_results["num_model_evals"] > 0:
            num_model_evals_per_sec_total.append(plan_results["num_model_evals"]/plan_results["elapsed_time"])
            elapsed_times.append(plan_results["elapsed_time"])
            print("num model evals", num_model_evals_per_sec_total)
            print("elapsed times", elapsed_times)
        for plan_result_to_save in plan_results_list:
            if 'planner' in plan_result_to_save.keys():
                del plan_result_to_save["planner"]
        np.save(plan_results_filename, plan_results_list)

        old_goal, new_goal = task.resample_goal(env=env)
        num_goals_tested +=1
        if old_goal is not None and new_goal is not None:
            logging.info("Resampled goal: \n"
                         f"                      old goal:  \t  {np.array_str(old_goal, precision=3, suppress_small=True)}\n"
                         f"                      new goal:  \t  {np.array_str(new_goal, precision=3, suppress_small=True)}"
                         )


    logging.info(f"    Total goals to reach:    \t   {total_goals_to_reach} \n"
                 + f"    planner goals reached: {planner_goals_reached}, ({planner_goals_reached/total_goals_to_reach:.2f})\n"
                 + f"    actual  goals reached: {actual_goals_reached}, ({actual_goals_reached/total_goals_to_reach:.2f})"
                 + f"    model_evals_per_sec {np.mean(num_model_evals_per_sec_total)}\n"
                 + f"    elapsed time {np.mean(elapsed_times)}")


if __name__ == '__main__':
    main()
