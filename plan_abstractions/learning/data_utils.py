import numpy as np
from autolab_core import YamlConfig
from pathlib import Path
from async_savers import load_shards
from pillar_state import State
import matplotlib.pyplot as plt
from torch_utils import get_numpy
from tqdm import tqdm
import os
from ..utils.utils import get_numpy

import os.path as osp
import logging

#from plan_abstractions.learning.mde_gnn_utils import pillar_state_to_graph, add_action_feature_to_graph, \
#    graph_distance_function
from plan_abstractions.utils import dists_and_actions_from_states_and_parameters, get_pose_pillar_state
from plan_abstractions.utils.distance_functions import *

logger = logging.getLogger(__name__)

FAR_POS_BOX = [0.5, 4]
FAR_POS_DRAWER = [0.5, -4, 0.005]


def extract_low_level_model_deviations_from_processed_datas(processed_datas, low_level_model, distance_function,
                                                            shuffle=True, graph_transitions=False):
    deviations = []
    transitions = processed_datas["low_level_transitions"]
    states_and_parameters = []
    logged_actual_poses = []
    logged_predicted_poses = []
    debug=True
    for transition in transitions:
        if graph_transitions:
            states_and_parameters.append(add_action_feature_to_graph(transition[0], transition[1]))
        else:
            states_and_parameters.append(np.hstack([transition[0], transition[1]]).flatten())
        predicted_next_state = low_level_model(transition[0], transition[1])
        if debug:
            gripper_pose = get_numpy(transition[2].x)[0,:3]
            predicted_gripper_pose = get_numpy(predicted_next_state.x)[0,:3]
            rod_pose = get_numpy(transition[2].x)[0,:3]
            predicted_rod_pose = get_numpy(predicted_next_state.x)[1,:3]
            logged_actual_poses.append(gripper_pose)
            logged_predicted_poses.append(predicted_rod_pose)
        deviation = distance_function(predicted_next_state, transition[2], ee_only=False)
        deviations.append(deviation)

    order = np.arange(len(deviations))
    plt.plot(order, np.vstack(logged_actual_poses)[:,0], label="x")
    #plt.plot(order, np.vstack(logged_actual_poses)[:,1], label="y")
    #plt.plot(order, np.vstack(logged_actual_poses)[:,2], label="z")
    plt.plot(order, np.vstack(logged_predicted_poses)[:,0], label="x_pred")
    #plt.plot(order, np.vstack(logged_predicted_poses)[:,1], label="y_pred")
    #plt.plot(order, np.vstack(logged_predicted_poses)[:,2], label="z_pred")
    plt.plot(order, deviations, label="dev")
    plt.legend()
    plt.show()
    random_order = np.random.permutation(order)
    if shuffle:
        order = random_order
    deviations_shuffled = np.array(deviations)[order].reshape(-1, 1)
    if not graph_transitions:
        states_and_parameters_shuffled = np.array(states_and_parameters)[order]
    else:
        states_and_parameters_shuffled = []
        for idx in order:
            states_and_parameters_shuffled.append(states_and_parameters[idx])

    return deviations_shuffled, states_and_parameters_shuffled


def add_box_or_drawer_to_env(env, pillar_state):
    """

    Args:
        env: env object
        pillar_state:  state to set to

    Modifies env to match pillar_state, include inferring whether the drawer or box should be there
    TODO lagrassa make this less hacky: use SimModel
    """
    if "frame:drawer:pose/position" in pillar_state.get_prop_names():
        env.reset_real_box(FAR_POS_BOX)
        env.reset_real_drawer( [0.4, 0.38, 0.005])
    else:
        print(pillar_state.get_prop_names())
        goal_pose =  [0.43119273, -0.12742051]
        env.reset_real_drawer(FAR_POS_DRAWER)
        env.reset_real_box(goal_pose)

def remove_outliers(dataset, high_deviation=10.5):
    new_dataset = {}
    data_idxs = [0,1]
    for data_key in ["training", "test"]:
        new_dataset[data_key] = [[],[]]
        for i in range(dataset[data_key][1].shape[0]):
            input_vec, output_scalar = dataset[data_key][0][i],dataset[data_key][1][i]
            if abs(output_scalar.item()) < high_deviation:
                new_dataset[data_key][0].append(input_vec)
                new_dataset[data_key][1].append(output_scalar)
        for data_idx in data_idxs:
            new_dataset[data_key][data_idx] = np.array(new_dataset[data_key][data_idx])
    return new_dataset


def extract_model_deviations_from_processed_datas(cfg, processed_datas, skill, env_cls, sem_state_obj_names, plot,
                                                  use_sim_model=False, graph_transitions=False, save_all=False,
                                                  do_data_aug=False, state_and_param_to_features=True, data_aug_cfg=None, env=None,
                                                  shuffle=True):
    if "Rod" in env_cls.__name__ or "Drawer" in env_cls.__name__:
        return rod_extract_model_deviations_from_processed_datas(processed_datas, skill, env_cls, sem_state_obj_names, plot, use_sim_model, graph_transitions, save_all, do_data_aug, feature_type, data_aug_cfg, env, shuffle)

    deviations = []
    init_states = processed_datas["init_states"]
    parameters = processed_datas["parameters"]
    end_states = processed_datas["end_states"]
    if cfg["shared_info"]["distance_function"] is not None:
        distance_function = eval(cfg["shared_info"]["distance_function"])
    else:
        distance_function = lambda pred_effects, end_state : np.linalg.norm(pred_effects-end_state)
    for init_state, parameter, end_state in zip(init_states, parameters, end_states):
        pred_effects = skill.effects(init_state, parameter)["end_states"][0]
        deviation = distance_function(pred_effects,end_state)
        deviations.append(deviation)
    states_and_parameters = np.hstack([init_states, parameters])
    deviations = np.array(deviations)
    if do_data_aug:
        states_and_parameters, deviations  = vector_data_augmentation(states_and_parameters, deviations, data_aug_cfg=data_aug_cfg)
    features = state_and_param_to_features(states_and_parameters)
    order = np.arange(len(deviations))
    random_order = np.random.permutation(order)
    if shuffle:
        order = random_order
    return deviations[order].reshape(-1, 1), features[order]


def rod_extract_model_deviations_from_processed_datas(processed_datas, skill, env_cls, sem_state_obj_names, plot,
                                                  use_sim_model=False, graph_transitions=False, save_all=False,
                                                  do_data_aug=False, state_and_param_to_features=True, data_aug_cfg=None, env=None, shuffle=True):
    if use_sim_model:
        assert env is not None
    deviations = []
    init_pillar_states = processed_datas["init_pillar_states"]
    init_states = processed_datas["init_states"]
    precond_init_pillar_states = []
    parameters = processed_datas["parameters"]
    end_states = processed_datas["end_states"]
    ee_yaw_diffs = []
    rod_yaw_diffs = []
    ee_state_diffs = []
    rod_state_diffs = []
    bin_diffs = []
    states_and_parameters = []
    compute_yaw_diffs = False
    for init_pillar_state, init_state, parameter, end_state in zip(init_pillar_states, init_states, parameters,
                                                                   end_states):
        if not skill.precondition_satisfied(init_pillar_state, parameter):
            continue
    fix_data_collection_bug = True
    i = 0
    for init_pillar_state, init_state, parameter, end_state in zip(init_pillar_states, init_states, parameters,
                                                                   end_states):
        if i % 10 == 0:
            print(f"{i} out of {len(end_states)}")
        if fix_data_collection_bug:
            error_z = 0.2
            if end_state[6] > error_z or end_state[10] > error_z:
                print("Skipping data point where perception was wrong")
                continue
        if use_sim_model:
            lowest_z = 0.019
            for obj_name in ["rod0", "rod1"]:
                if get_pose_pillar_state(init_pillar_state, obj_name)[2] < lowest_z:
                    position_prop_name = f"frame:{obj_name}:pose/position"
                    old_position = init_pillar_state.get_values_as_vec([position_prop_name])
                    new_position = old_position.copy()
                    new_position[2] = lowest_z
                    init_pillar_state.update_property(position_prop_name, new_position)
            #hack: infer which env the sample was from to determine whether to add the box or drawer
            add_box_or_drawer_to_env(env, init_pillar_state)
            pred_effects = skill.gt_effects(env, [init_pillar_state], [parameter], T_plan_max=1, T_exec_max=1000)
            print(get_pose_pillar_state(init_pillar_state, "rod0")[2])
            print(get_pose_pillar_state(init_pillar_state, "rod1")[2])
            pred_effects["end_states"] = [State.create_from_serialized_string(end_state) for end_state in
                                          pred_effects["end_states"]]
        else:
            pred_effects = skill.effects(init_pillar_state, parameter)
        pred_effects_sem = env_cls.pillar_state_to_sem_state(pred_effects["end_states"][0],
                                                             sem_state_obj_names=sem_state_obj_names)
        collision_eps = 0
        asset_array_to_eps_array = {"finger_left": [2 * collision_eps, 2 * collision_eps],
                                    "finger_right": [2 * collision_eps, 2 * collision_eps],
                                    "rod": [collision_eps, collision_eps]
                                    }
        body_names = ["franka:finger_left", "franka:finger_right", "rod0", "rod1"]
        # env_cls.is_in_collision(init_pillar_state,body_names=body_names,asset_name_to_eps_arr=asset_array_to_eps_array, plot=True)
        # env_cls.is_in_collision(pred_effects["end_states"][0],body_names=body_names,asset_name_to_eps_arr=asset_array_to_eps_array, plot=True)
        if compute_yaw_diffs:
            pencil0_dist = np.linalg.norm(end_state[:2] - end_state[4:6])
            pencil1_dist = np.linalg.norm(end_state[:2] - end_state[8:10])
            init_pencil0_dist = np.linalg.norm(init_state[:2] - init_state[4:6])
            init_pencil1_dist = np.linalg.norm(init_state[:2] - init_state[8:10])
            min_dist = min(pencil0_dist, pencil1_dist)

            print("#####################")
            print("yaw from parameter",parameter[3])
            print("end ee yaw", end_state[3])
            if min_dist> 0.1:
                print("Rod far: probably dropped")
            elif init_pencil0_dist < init_pencil1_dist:
                print("inital ee yaw", init_state[3])
                print("inital rod0 yaw", init_state[7])
                print("initial diff", init_state[3] - init_state[7])
                print("end rod0yaw", end_state[7])
                print("pred end rod0yaw", pred_effects_sem[7])
                vis_end_state = pred_effects['end_states'][0]
                env_cls.visualize_pillar_state(vis_end_state)
            else:
                print("inital ee yaw", init_state[3])
                print("inital rod1 yaw", init_state[11])
                print("initial diff", init_state[3] - init_state[11])
                print("end rod1yaw", end_state[11])
                print("pred end rod1yaw", pred_effects_sem[11])
                vis_end_state = pred_effects['end_states'][0]
                env_cls.visualize_pillar_state(vis_end_state)
            print("#####################")
            ee_yaw_diffs.append(abs(pred_effects_sem[3] - end_state[3]))
            rod_yaw_diffs.append(abs(pred_effects_sem[7] - end_state[7]))
            rod_yaw_diffs.append(abs(pred_effects_sem[11] - end_state[11]))
        ee_state_diffs.append(np.linalg.norm(pred_effects_sem[:3] - end_state[:3]))
        rod_state_diffs.append(np.linalg.norm(np.linalg.norm(pred_effects_sem[4:7] - end_state[4:7])))
        rod_state_diffs.append(np.linalg.norm(np.linalg.norm(pred_effects_sem[8:11] - end_state[8:11])))
        precond_init_pillar_states.append(init_pillar_state)
        if len(end_state) >= 16:
            bin_diffs.append(np.linalg.norm(end_state[12:15] - pred_effects_sem[12:15]))
        else:
            bin_diffs.append(0)
        deviations.append(np.linalg.norm(pred_effects_sem - end_state))
        states_and_parameters.append(np.hstack([init_state, parameter]))
        i += 1
    # states_and_parameters = np.hstack([init_states, parameters])
    states_and_parameters = np.array(states_and_parameters)
    ee_state_diffs = np.array(ee_state_diffs)
    rod_state_diffs = np.array(rod_state_diffs)
    bin_diffs = np.array(bin_diffs)
    all_rod_state_diffs = 0.5 * (rod_state_diffs[::2] + rod_state_diffs[1::2])
    deviations = ee_state_diffs + all_rod_state_diffs + bin_diffs
    if do_data_aug:
        states_and_parameters, deviations  = rod_vector_data_augmentation(states_and_parameters, deviations, data_aug_cfg=data_aug_cfg)
    else: #not "true" data augmentation, just flips rod positions
        states_and_parameters, deviations = augment_with_flipped(states_and_parameters, deviations)

    threshold = 0.015
    good = deviations < threshold
    save_good = False
    save_all = False
    if save_good:
        print(f"Saving {np.sum(good)} states")
        np.save("/home/lagrassa/git/plan-abstractions/data/good_states_1us.npy",
                [state.get_serialized_string() for state in init_pillar_states[good]])
        np.save("/home/lagrassa/git/plan-abstractions/data/good_params_1us.npy", parameters[good])
    if save_all:
        np.save("/home/lagrassa/git/plan-abstractions/data/dev_states.npy",
                [state.get_serialized_string() for state in init_pillar_states])
        np.save("/home/lagrassa/git/plan-abstractions/data/dev_params.npy", parameters)
    order = np.arange(len(deviations))
    random_order = np.random.permutation(order)
    print("Total dataset size", deviations.shape)
    states_and_parameters = state_and_params_to_features(states_and_parameters)
    plot_min_dist = False
    if plot_min_dist:
        assert feature_type == "dists_and_actions_only"
        min_dists = np.min(states_and_parameters[:,:2], axis=1)
        plt.scatter(min_dists, deviations)
        plt.xlabel("Min dist")
        plt.ylabel("deviation")
        plt.show()
    if shuffle:
        order = random_order
    return deviations[order].reshape(-1, 1), states_and_parameters[order]

def augment_with_flipped(states_and_parameters, deviations):
    states_and_parameters_flipped = states_and_parameters.copy()
    states_and_parameters_flipped[:, 4:8] = states_and_parameters[:, 8:12].copy()
    states_and_parameters_flipped[:, 8:12] = states_and_parameters[:, 4:8].copy()
    states_and_parameters = np.vstack([states_and_parameters_flipped, states_and_parameters])
    deviations = np.hstack([deviations, deviations])
    return states_and_parameters, deviations

def vector_data_augmentation(states_and_parameters, deviations, data_aug_cfg):
    num_noise_aug = data_aug_cfg.get('num_noise_aug', 3000)
    state_mag = data_aug_cfg.get('state_noise_mag',0.005)  # 0.015
    action_mag = data_aug_cfg.get('action_noise_mag',0.05) # 0.1
    state_ndim = data_aug_cfg["state_ndim"]
    num_og_data = len(states_and_parameters)
    augmented_states_and_parameters = np.zeros(
        (states_and_parameters.shape[0] * (num_noise_aug), states_and_parameters.shape[1]))
    augmented_deviations = np.zeros((deviations.shape[0] * (num_noise_aug),))
    # Half are adding random noise to the parameters. The other half add a grid of x and y values. you'll add data_aug_num to it.
    for aug_i in range(num_noise_aug):
        if aug_i < num_noise_aug:
            # augmented_state_param_sample = states_and_parameters + np.random.uniform(low=state_low, high= state_high, size = states_and_parameters.shape)
            augmented_state_param_sample = states_and_parameters.copy()
            augmented_state_param_sample[:, :state_ndim] += np.random.normal(0, state_mag / 1.96,
                                                                     size=states_and_parameters[:, :state_ndim].shape)
            augmented_state_param_sample[:, state_ndim:] += np.random.normal(0, action_mag / 1.96,
                                                                     size=states_and_parameters[:, state_ndim:].shape)
            augmented_dev_sample = deviations

        augmented_states_and_parameters[aug_i * num_og_data:aug_i * num_og_data + num_og_data,
        :] = augmented_state_param_sample
        augmented_deviations[aug_i * num_og_data: aug_i * num_og_data + num_og_data] = augmented_dev_sample
    states_and_parameters = np.vstack([augmented_states_and_parameters, states_and_parameters])
    deviations = np.hstack([augmented_deviations, deviations])
    return states_and_parameters, deviations

def rod_vector_data_augmentation(states_and_parameters, deviations, data_aug_cfg):
    num_noise_aug = data_aug_cfg.get('num_noise_aug', 3000)
    # 3000, #1200, #2,#,40,
    num_grid_aug = data_aug_cfg.get("num_grid_aug", 10)
    state_mag = data_aug_cfg.get('state_noise_mag',0.005)  # 0.015
    action_mag = data_aug_cfg.get('action_noise_mag',0.05) # 0.1
    state_low = -state_mag  # percentages
    state_high = state_mag
    shift_mag = 0.1
    states_and_parameters, deviations = augment_with_flipped(states_and_parameters, deviations)
    num_og_data = len(states_and_parameters)
    total_grid_aug_num = (2 * num_grid_aug + 1) ** 2  # grid aug num is the number to add in x and y
    augmented_states_and_parameters = np.zeros(
        (states_and_parameters.shape[0] * (num_noise_aug + total_grid_aug_num), states_and_parameters.shape[1]))
    augmented_deviations = np.zeros((deviations.shape[0] * (num_noise_aug + total_grid_aug_num),))
    # Half are adding random noise to the parameters. The other half add a grid of x and y values. you'll add data_aug_num to it.
    for aug_i in range(num_noise_aug):
        if aug_i < num_noise_aug:
            # augmented_state_param_sample = states_and_parameters + np.random.uniform(low=state_low, high= state_high, size = states_and_parameters.shape)
            augmented_state_param_sample = states_and_parameters.copy()
            augmented_state_param_sample[:, :12] += np.random.normal(0, state_mag / 1.96,
                                                                     size=states_and_parameters[:, :12].shape)
            augmented_state_param_sample[:, 12:] += np.random.normal(0, action_mag / 1.96,
                                                                     size=states_and_parameters[:, 12:].shape)
            augmented_dev_sample = deviations

        augmented_states_and_parameters[aug_i * num_og_data:aug_i * num_og_data + num_og_data,
        :] = augmented_state_param_sample
        augmented_deviations[aug_i * num_og_data: aug_i * num_og_data + num_og_data] = augmented_dev_sample
    aug_i = num_noise_aug
    x_vals = np.array([0, 4, 8])
    y_vals = np.array([1, 5, 9])
    if num_grid_aug > 0:
        for i in np.linspace(-shift_mag, shift_mag, 2 * num_grid_aug + 1):
            for j in np.linspace(-shift_mag, shift_mag, 2 * num_grid_aug + 1):
                state_param_copy = states_and_parameters.copy()
                state_param_copy[:, x_vals] += i
                state_param_copy[:, y_vals] += j
                augmented_states_and_parameters[aug_i * num_og_data:aug_i * num_og_data + num_og_data,
                :] = state_param_copy
                augmented_deviations[aug_i * num_og_data: aug_i * num_og_data + num_og_data] = deviations
                aug_i += 1
    states_and_parameters = np.vstack([augmented_states_and_parameters, states_and_parameters])
    deviations = np.hstack([augmented_deviations, deviations])
    return states_and_parameters, deviations


def unnormalize_data(data, normalization_type, z_normalization_std=None, z_normalization_mean=None):
    norm_type = normalization_type
    if norm_type == 'z_normalization':
        out = (data * z_normalization_std) + z_normalization_mean
    elif norm_type == 'none':
        out = data
    else:
        raise ValueError("Invalid normalization type")
    return out


def load_datas_from_cfg(cfg):
    return load_datas_from_dir_with_tags(cfg['root'], cfg['tags'])


def get_latest_data_paths_with_tags(dataset_root_path, tags):
    paths = []
    if type(dataset_root_path) is str and not osp.exists(dataset_root_path):
        logging.info(f"Trying to load data that does not exist: {dataset_root_path}. Wll return.")
        return paths
    if type(dataset_root_path) is Path and not dataset_root_path.exists():
        logging.info(f"Trying to load data that does not exist: {dataset_root_path}. Wll return.")
        return paths

    for tag in tqdm(tags, desc='In load_datas_from_cfg'):
        tag_path = Path(dataset_root_path) / tag
        if not tag_path.exists():
            logging.info(f"Trying to load data with tag that does not exist: {tag_path}.")
            continue

        all_data_paths = list(filter(lambda p: p.is_dir(), list(tag_path.iterdir())))
        all_data_paths.sort()
        latest_data_path = all_data_paths[-1] / 'data'
        paths.append(latest_data_path)

    return paths

def load_shards_custom(path):
    datas = []
    for fn in os.listdir(path):
        data = np.load(os.path.join(path,fn), allow_pickle=True)
        datas.extend(data)
    return datas
        

def load_datas_from_dir_with_tags(dataset_root_path, tags):
    paths = get_latest_data_paths_with_tags(dataset_root_path, tags)
    datas = []
    for path in tqdm(paths, desc='In load_datas_from_cfg'):
        datas.extend(load_shards_custom(path))
    return datas


def get_datas(cfg, data_root_key, data_tags_key):
    if isinstance(cfg[data_root_key], str):
        datas = load_datas_from_dir_with_tags(cfg[data_root_key], cfg[data_tags_key])
    elif isinstance(cfg[data_root_key], dict) or isinstance(cfg[data_root_key], DictConfig):
        datas = []
        for data_key, data_values in cfg[data_root_key].items():
            data_root, data_tags = data_values['root'], data_values['tags']
            curr_datas = load_datas_from_dir_with_tags(data_root, data_tags)
            datas.extend(curr_datas)
    else:
        raise ValueError(
            f"Invalid data_root_key type: {type(cfg[data_root_key])}, data_root_key: {data_root_key}")
    return datas


def process_datas(datas, sem_state_obj_names, env_cls, skill_cls, only_terminating_trajs=True, anchor_obj_name=None,
                  get_sem_dict_repr=False,
                  convert_to_arrays=True,
                  use_settled_init_states=False,
                  get_transitions=False,
                  graph_transitions=False):
    processed_datas = {
        key: []
        for key in ['parameters', 'init_states', 'init_pillar_states', 'end_states', 'end_states_diff',
                    'T_pis', 'costs', 'T_execs', 'terminated', 'data_idx_env_idx', 'low_level_transitions',
                    'object_masks', 'sem_dict_repr']
    }
    processed_datas["low_level_transitions"] = []
    init_state_key = 'initial_states' if not use_settled_init_states else 'initial_settled_states'
    logging.info(f"Processing data with initial state key: {init_state_key}")
    already_warned = False
    for d_idx, d in tqdm(enumerate(datas), total=len(datas), desc='In process_datas'):
        if init_state_key not in d.keys():
            if not already_warned:
                init_state_key = "initial_states"
                logger.warning("Can't use initial_settled_states because not in data. This is expected for real robot data")
        ref_pillar_state = State.create_from_serialized_string(d[init_state_key][0])
        if get_transitions:
            if graph_transitions:
                state_conversion_fn = pillar_state_to_graph
            else:
                state_conversion_fn = lambda pillar_state: env_cls.pillar_state_to_sem_state(pillar_state,
                                                                                             sem_state_obj_names,
                                                                                             ref_pillar_state=ref_pillar_state,
                                                                                             anchor_obj_name=anchor_obj_name)
            processed_datas["low_level_transitions"].extend(
                extract_transitions(d, init_state_key=init_state_key, state_conversion_fn=state_conversion_fn))
        for env_idx, end_state_str in enumerate(d['exec_data']['end_states']):
            terminated = d['exec_data']['terminated'][env_idx]
            if only_terminating_trajs and not terminated:
                continue

            if 'relative_states' in d and anchor_obj_name is not None:
                initial_state = d['relative_states'][anchor_obj_name][init_state_key][env_idx]
                end_state = d['relative_states'][anchor_obj_name]['end_states'][env_idx]
                parameters = d['relative_parameters'][anchor_obj_name][env_idx]
            else:

                init_pillar_state = State.create_from_serialized_string(d[init_state_key][env_idx])
                initial_state = env_cls.pillar_state_to_sem_state(init_pillar_state, sem_state_obj_names,
                                                                  anchor_obj_name=anchor_obj_name,
                                                                  ref_pillar_state=init_pillar_state)

                end_pillar_state = State.create_from_serialized_string(end_state_str)
                end_state = env_cls.pillar_state_to_sem_state(end_pillar_state, sem_state_obj_names,
                                                              anchor_obj_name=anchor_obj_name,
                                                              ref_pillar_state=init_pillar_state)

                parameters = d['parameters'][env_idx]
                if anchor_obj_name is not None:
                    parameters = skill_cls.parameters_to_relative_parameters(parameters, init_pillar_state,
                                                                             anchor_obj_name)
                processed_datas['init_pillar_states'].append(init_pillar_state)
            processed_datas['init_states'].append(initial_state)
            processed_datas['end_states'].append(end_state)

            processed_datas['end_states_diff'].append(end_state - initial_state)

            masks = env_cls.pillar_state_to_sem_state_masks(end_pillar_state, init_pillar_state, sem_state_obj_names)
            processed_datas['object_masks'].append(masks)

            # Use for GNNs
            if get_sem_dict_repr:
                sem_dict_repr = env_cls.get_dict_representation_for_sem_data(initial_state, end_state,
                                                                             processed_datas['end_states_diff'][-1],
                                                                             masks,
                                                                             sem_state_obj_names)
                processed_datas['sem_dict_repr'].append(sem_dict_repr)

            processed_datas['parameters'].append(parameters)
            processed_datas['T_pis'].append(d['exec_data']['info_plan'][env_idx]['T_plan'])

            processed_datas['costs'].append(d['exec_data']['costs'][env_idx])
            processed_datas['T_execs'].append(d['exec_data']['T_exec'][env_idx])

            processed_datas['terminated'].append(int(terminated))
            processed_datas['data_idx_env_idx'].append((d_idx, env_idx))

    if convert_to_arrays:
        not_to_process_keys = ['data_idx_env_idx', 'sem_dict_repr']
        for key, val in processed_datas.items():
            if key in not_to_process_keys:
                continue
            processed_datas[key] = np.array(val)

    return processed_datas


def extract_transitions(d, state_conversion_fn=None, init_state_key="initial_states"):
    transitions = []
    total_dataset_t_per_env_idx = {idx: 0 for idx in range(len(d[init_state_key]))}
    for env_idx, env_traj_set in enumerate(d['exec_data']['low_level_states']):
        transitions_from_this_env_idx = []
        T_exec = d['exec_data']["T_exec"][env_idx]
        start_t = int(total_dataset_t_per_env_idx[env_idx])
        end_t = total_dataset_t_per_env_idx[env_idx] + int(T_exec)
        traj_unprocessed = env_traj_set[start_t: end_t] + [d["exec_data"]["end_states"][env_idx], ]
        traj = [state_conversion_fn(State.create_from_serialized_string(state)) for state in traj_unprocessed]
        for time_in_traj_idx in range(len(traj) - 1):
            if time_in_traj_idx == 0:
                low_level_first_state = State.create_from_serialized_string(traj_unprocessed[0])
                high_level_first_state = State.create_from_serialized_string(d[init_state_key][env_idx])
                #assert get_pose_pillar_state(low_level_first_state, "rod0") == get_pose_pillar_state(
                #    high_level_first_state, "rod0")
            init_low_level_state = traj[time_in_traj_idx]
            end_low_level_state = traj[time_in_traj_idx + 1]
            low_level_action = d['exec_data']['low_level_actions'][env_idx][start_t + time_in_traj_idx]
            transition = (init_low_level_state, low_level_action, end_low_level_state)
            transitions.append(transition)
            transitions_from_this_env_idx.append(transition)
        total_dataset_t_per_env_idx[env_idx] += int(T_exec)
        assert len(transitions_from_this_env_idx) == len(d['exec_data']['low_level_states'][
                                                             env_idx])  # would be len all states minus end state but we add the end state
        assert len(transitions_from_this_env_idx) == int(d['exec_data']['T_exec'][env_idx])
    return transitions


def get_min_dist(states_and_params):
    rod0_dist = np.linalg.norm(states_and_params[:, :2] - states_and_params[:, 4:6], axis=1)
    rod1_dist = np.linalg.norm(states_and_params[:, :2] - states_and_params[:, 8:10], axis=1)
    min_dist = np.min(np.vstack([rod0_dist, rod1_dist]), axis=0)
    return min_dist

def feature_type_to_state_and_param_to_features_fn(feature_type):
    if feature_type == "dists_and_actions_only":
        fn = lambda states_and_parameters : dists_and_actions_from_states_and_parameters(states_and_parameters,
                                                                             only_dists=True,
                                                                             state_ndims=end_states[0].shape[0])
    elif feature_type == "pose_only":
        lambda states_and_parameters : states_and_parameters
    elif feature_type == "pose_and_dists_and_actions":
        states_and_parameters = lambda states_and_parameters: dists_and_actions_from_states_and_parameters(states_and_parameters,
                                                                             only_dists=False,
                                                                             state_ndims=end_states[0].shape[0])
    else:
        ValueError(f"Unknown feature type: {feature_type}")
    return fn


def make_deviation_datalists(cfg, feature_type=False, plot=0, shuffle=True, graphs=False, setup_callbacks=[], processed_datas_train=None, processed_datas_val=None, state_and_param_to_features=None):
    """
    Note: feature_type is kept for compatibility but should not be used if state_and_param_to_features is not None
    """
    from ..envs import FrankaRodEnv, FrankaDrawerEnv, WaterEnv2D, WaterEnv3D
    from ..skills import FreeSpaceMoveToGroundFranka, OpenDrawer, LiftAndPlace, LiftAndDrop, Pick, WaterTransport2D, Pour
    if feature_type and state_and_param_to_features is None:
        state_and_param_to_features = feature_type_to_state_and_param_to_features_fn(feature_type)
    states_and_parameters_train_all_skills = []
    states_and_parameters_val_all_skills = []
    deviations_train_all_skills = []
    deviations_val_all_skills = []
    for skill_type, skill_cfg in cfg['skills'].items():
        env_cls = eval(skill_cfg['env'])
        skill = eval(skill_type)(low_level_models_cfg=skill_cfg.get('low_level_models', None),
                                 sem_cfg = cfg["shared_info"].get('sem_cfg', None),
                                 real_robot=False)
        skill_cls = skill.__class__
        use_sim_model = cfg["shared_info"].get('use_sim_model', False)
        if use_sim_model:
            env_cfg = cfg
            env_cfg["scene"]["n_envs"] = 1
            def add_box_far_away_cb(env, scene, env_idx):
                box_cfg = YamlConfig(os.path.join(cfg.original_cwd, "cfg/tasks/box_rod_franka.yaml"))['task']
                box_dims = np.array(list(box_cfg['goal']['dims'].values()))
                env.add_real_box_cb(FAR_POS_BOX, box_dims)

            def add_drawer_far_away_cb(env, scene, env_idx):
                #drawer_cfg = YamlConfig("cfg/tasks/drawer_task.yaml")
                env.add_real_drawer_cb(env, scene, env_idx, FAR_POS_DRAWER)

            if env_cls.__name__ == "FrankaDrawerEnv":
                setup_callbacks = []
            else:
                setup_callbacks = [add_box_far_away_cb, add_drawer_far_away_cb]
            env = env_cls(env_cfg, setup_callbacks=setup_callbacks, for_mde_training=True) #Terrible hack because the envs are different and it's hard to run multiple IG instances (lagrassa)
        else:
            env = None
        states_and_parameters_this_skill, deviations_this_skill = get_deviations_from_data(cfg, "root", "tags", env_cls, skill,
                                                                     skill_cfg, skill_cls, plot, do_data_aug=True,
                                                                     state_and_param_to_features=state_and_param_to_features,
                                                                     shuffle=shuffle, graphs=graphs,
                                                                     use_sim_model=use_sim_model,
                                                                     processed_datas=processed_datas_train,
                                                                     env=env)
        states_and_parameters_val_this_skill, deviations_val_this_skill = get_deviations_from_data(cfg, "val_root", "val_tags", env_cls, skill,
                                                                             skill_cfg, skill_cls, plot, shuffle=shuffle,
                                                                             save_all=True, graphs=graphs,
                                                                             state_and_param_to_features=state_and_param_to_features,
                                                                             processed_datas=processed_datas_val,
                                                                             use_sim_model=use_sim_model, env=env)
        states_and_parameters_train_all_skills.append(states_and_parameters_this_skill)
        states_and_parameters_val_all_skills.append(states_and_parameters_val_this_skill)
        deviations_train_all_skills.append(deviations_this_skill)
        deviations_val_all_skills.append(deviations_val_this_skill)
    states_and_parameters = np.vstack(states_and_parameters_train_all_skills)
    states_and_parameters_val = np.vstack(states_and_parameters_val_all_skills)
    deviations = np.vstack(deviations_train_all_skills)
    deviations_val = np.vstack(deviations_val_all_skills)
    if False and not graphs:
        logger.info(f"Validation dataset : {states_and_parameters_val.shape}")
        max_distance_to_training_data = [
            min(np.linalg.norm(states_and_parameters_val[i] - states_and_parameters, axis=1)) for i in
            range(len(states_and_parameters_val))]
        logger.debug(f"Max distances to training data:  {max_distance_to_training_data}")
    data = {"training": [states_and_parameters, deviations],
            "test": [states_and_parameters_val, deviations_val]}
    return data


def get_deviations_from_data(cfg, train_data_key, data_tags_key, env_cls, skill, skill_cfg, skill_cls, plot,
                             save_all=False, state_and_param_to_features=False, do_data_aug=False, use_sim_model=False, sem_state_obj_names=None,
                             graphs=False, env=None, shuffle=False, processed_datas=None):
    if processed_datas is None:
        datas = get_datas(skill_cfg["data"], data_root_key=train_data_key, data_tags_key=data_tags_key)
        sem_state_obj_names = skill_cfg["data"]["sem_state_obj_names"]
        processed_datas = process_datas(datas, list(sem_state_obj_names), env_cls, skill_cls, use_settled_init_states=True,
                                        only_terminating_trajs=skill_cfg.get("only_terminated", False),
                                        get_transitions=skill_cfg.get("low_level"),
                                        graph_transitions=graphs)
    if graphs:
        distance_function = graph_distance_function
    else:
        def distance_function(state1, state2):
            return np.linalg.norm(state1[:3] - state2[:3])
    if skill_cfg.get("low_level", False):
        deviations, states_and_parameters = extract_low_level_model_deviations_from_processed_datas(processed_datas,
                                                                                                    skill.low_level_models[
                                                                                                        0],
                                                                                                    distance_function,
                                                                                                    shuffle=shuffle,
                                                                                                    graph_transitions=graphs)
    else:
        deviations, states_and_parameters = extract_model_deviations_from_processed_datas(cfg, processed_datas, skill,
                                                                                          env_cls,
                                                                                          sem_state_obj_names, plot,
                                                                                          save_all=save_all,
                                                                                          do_data_aug=do_data_aug,
                                                                                          data_aug_cfg = cfg['data_aug'],
                                                                                          # 10,#2, #20,
                                                                                          state_and_param_to_features=state_and_param_to_features,
                                                                                          use_sim_model=use_sim_model,
                                                                                          shuffle=shuffle,
                                                                                          env=env)
    return states_and_parameters, deviations

def eval_model(deviation_model, train_states_and_params, train_deviations,
               test_states_and_params, test_deviations, validation_states_and_params,
               validation_deviations, do_pr_curve=False, do_traj_curve=0, plot=0):
    data_type_keys = ["train", "validation", "test"]
    stats_dict={key: {} for key in data_type_keys}
    pred_validation_deviations = deviation_model.predict(validation_states_and_params, already_transformed_state_vector=True)
    pred_train_deviations = deviation_model.predict(train_states_and_params, already_transformed_state_vector=True)
    pred_test_deviations = deviation_model.predict(test_states_and_params, already_transformed_state_vector=True)
    import ipdb; ipdb.set_trace()
    if plot:
        plt.scatter(train_deviations, pred_train_deviations)
        plt.show()
        plt.scatter(test_deviations, pred_test_deviations)
        plt.show()
    train_stats = print_and_log_stats(train_deviations, pred_train_deviations)
    logger.info("Validation deviation error")
    validation_stats = print_and_log_stats(validation_deviations, pred_validation_deviations)
    logger.info("Test data deviation error")
    test_deviations = test_deviations.flatten()
    pred_test_deviations = pred_test_deviations.flatten()
    test_lg_raw = np.mean((deviation_model.evaluate_loss(test_deviations, pred_test_deviations)))
    logger.info(f"Test LG sqrt : {np.mean(np.sqrt(deviation_model.evaluate_loss(test_deviations, pred_test_deviations)))}")
    logger.info(f"Test LG raw: {test_lg_raw}")
    test_stats = print_and_log_stats(test_deviations, pred_test_deviations)
    min_dist = np.min(test_states_and_params[:2], axis=1)
    for key, specific_stats_dict in zip(data_type_keys, [train_stats, validation_stats, test_stats]):
        stats_dict[key].update(specific_stats_dict)

    stats_dict["test"]["lg"] = test_lg_raw #To get the others can look at wandb curve
    if do_pr_curve:
        from plan_abstractions.utils.plot_utils import precision_recall_curve
        precisions, recalls, thresholds_with_data = precision_recall_curve(test_deviations,
                                                                           pred_test_deviations)
        return precisions, recalls, thresholds_with_data
    if do_traj_curve:
        order = np.arange(len(test_deviations))
        plt.plot(order, test_deviations, label="ground truth deviations")
        plt.plot(order, pred_test_deviations, label="pred deviations")
        plt.plot(order, min_dist, label="min_dist")
        plt.xlabel("Timesteps (for multiple trajectories)")
        plt.ylabel("d (m)")
        plt.legend()
        plt.show()
    return stats_dict

def train_and_fit_model(cfg, deviation_model, train_states_and_params, train_deviations, states_and_parameters_from_planning, deviations_from_planning, validation_states_and_params, validation_deviations, experiment, validate_on_split_data=True):
    if validate_on_split_data:
        deviation_model.train(cfg, train_states_and_params, train_deviations, validation_states_and_params, validation_deviations, wandb_experiment=experiment)
    else:
        deviation_model.train(cfg, train_states_and_params, train_deviations, states_and_parameters_from_planning, deviations_from_planning,wandb_experiment=experiment)

def compute_deviation_helper(pred_deviations, actual_deviations):
    return np.mean(np.abs(pred_deviations - actual_deviations))


def print_and_log_stats(gt_deviations, pred_deviations):
    gt_deviations = gt_deviations.flatten()
    pred_deviations = pred_deviations.flatten()
    mean_error = compute_deviation_helper(pred_deviations, gt_deviations)
    diff = gt_deviations - pred_deviations
    data = {"mean_error": mean_error, "max_error":np.max(diff), 
            "min_error": np.min(diff),
            "overestimate_10": np.sum(diff > 0.10)/len(diff), 
            "overestimate_7": np.sum(diff > 0.07)/len(diff), 
            "underestimate_10": np.sum(diff < -0.10)/len(diff), 
            "underestimate_7": np.sum(diff < -0.07)/len(diff)} 
    for key in data.keys():
        print(f"{key} : {data[key]}")
    return data

def data_restrict_training_set(dataset_data, max_num_data, skill_name):
    """
    Randomly selects max_num_data from the training tag of dataset_data. 
    """
    training_data = dataset_data["training"]
    order = np.arange(len(training_data[1]))
    random_order = np.random.permutation(order)
    random_subset_idxs = random_order[:max_num_data]
    random_subset = []
    for arr in training_data:
        random_subset.append(arr[random_subset_idxs])
    dataset_data["training"] = random_subset



def make_vector_datas(cfg, skill_name=None, tag_name="tags"):
    data_root = cfg["data"]["root"]
    folder_name = cfg["data"][tag_name][0]
    data_dir = os.path.join(data_root, folder_name)
    data_list = []
    for exp_name in os.listdir(data_dir):
        data =  np.load(os.path.join(data_dir, exp_name), allow_pickle=True).item()
        if skill_name is not None:
            data = data[skill_name]
        data['parameters']  = data["params"]
        data_list.append(data)
    data_combined = {}
    for key in data_list[0].keys(): #assumes same keys
        try:
            data_combined[key] = np.vstack([dataset[key] for dataset in data_list])
        except:
            import ipdb; ipdb.set_trace()
    return data_combined
