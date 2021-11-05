import numpy as np

import copy
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
import pandas as pd

import copy
import seaborn as sns
import logging
import wandb

from plan_abstractions.envs import FrankaRodEnv
from plan_abstractions.learning.data_utils import train_and_fit_model

sns.set()


def plot_3d_output_data(ax, x, x_pred, x_hat, x_idx, y_idx, z_idx, color, label,
                        title='', xlabel='X', ylabel='Y', init_state=None, init_rod_states=None,
                        final_gt_rod_states=None, final_pred_rod_states_sampled=None,
                        invert_x_axes=False, invert_y_axes=False):
    '''Plots output data for some input scene.
    
    x: Input end state data. Numpy array or List (1D).
    x_pred: Predicted end state, usually obtained from the encoder. Numpy array or list (1D).
    x_hat: Distribution of predicted end states. 2D numpy array.
    x_idx: x-index of data within x/x_pred/x_hat array. Int.
    y_idx: y-index of data within x/x_pred/x_hat array. Int.
    z_idx: yaw index of data within x/x_pred/x_hat array. Int.
    color: List of color RGB with values between [0, 1].

    invert_x_axes: Invert X-axes for plotting. Set to False to match isaac simulation viewer.
    invert_y_axes: Invert Y-axes for plotting. Set to True to match isaac simulation viewer.
    '''
    ax.scatter(x[x_idx], x[y_idx], label=f'{label}-gt', color=1.0 - color, s=14 ** 2, marker='s')
    yaw = x[z_idx]
    scale = 1
    u, v = scale * np.cos(yaw), scale * np.sin(yaw)
    ax.quiver(x[x_idx], x[y_idx], x[x_idx] + u, x[y_idx] + v)

    min_x, min_y = x[x_idx], x[y_idx]
    max_x, max_y = x[x_idx], x[y_idx]

    if init_state is not None:
        arrow_color = sns.color_palette("tab10")[1]
        ax.scatter(init_state[x_idx], init_state[y_idx], marker='s', label='Init Pusher', color='r', s=16 ** 2)
        yaw = init_state[2]
        u, v = scale * np.cos(yaw), scale * np.sin(yaw)
        ax.quiver(init_state[x_idx], init_state[y_idx], init_state[x_idx] + u, init_state[y_idx] + v)
        ax.arrow(init_state[x_idx], init_state[y_idx], x[x_idx] - init_state[x_idx], x[y_idx] - init_state[y_idx],
                 color=arrow_color, linewidth=2, zorder=2.0, width=0.001, length_includes_head=True)

        min_x, min_y = min(init_state[x_idx], min_x), min(init_state[y_idx], min_y)
        max_x, max_y = max(init_state[x_idx], max_x), max(init_state[y_idx], max_y)

    if init_rod_states is not None:
        num_rods = int(len(init_rod_states) // 3)
        rod_colors = sns.color_palette("Paired", num_rods * 2)

    assert x_idx in (0, 1) and y_idx in (0, 1)
    if init_rod_states is not None:
        for rod_idx in range(num_rods):
            rod_start = 0 + rod_idx * 3
            color = rod_colors[rod_idx * 2]
            rod_x, rod_y = init_rod_states[rod_start + x_idx], init_rod_states[rod_start + y_idx]
            ax.scatter(rod_x, rod_y, marker='s', label=f'Init Rod-{rod_idx}',
                       color=color, s=16 ** 2)
            yaw = init_rod_states[rod_start + 2]
            u, v = scale * np.cos(yaw), scale * np.sin(yaw)
            ax.quiver(rod_x, rod_y, rod_x + u, rod_y + v, color=color)

            min_x, min_y = min(rod_x, min_x), min(rod_y, min_y)
            max_x, max_y = max(rod_x, max_x), max(rod_y, max_y)

    if final_pred_rod_states_sampled is not None:
        for rod_idx in range(num_rods):
            rod_start = 0 + rod_idx * 3
            color = rod_colors[rod_idx * 2 + 1] 
            rod_xy = final_pred_rod_states_sampled[:, rod_start:rod_start + 2] 
            ax.scatter(rod_xy[:, x_idx], rod_xy[:, y_idx], marker='o',
                       color=color, s=8 ** 2, alpha=0.3, label=f'Pred Rod-{rod_idx}-sampled')
            rod_xy_min, rod_xy_max = rod_xy.min(axis=0), rod_xy.max(axis=0)
            min_x, min_y = min(rod_xy_min[x_idx], min_x), min(rod_xy_min[y_idx], min_y)
            max_x, max_y = max(rod_xy_max[x_idx], max_x), max(rod_xy_max[y_idx], max_y)

    if final_gt_rod_states is not None:
        for rod_idx in range(num_rods):
            rod_start = 0 + rod_idx * 3
            color = rod_colors[rod_idx * 2 + 1]
            rod_x, rod_y = final_gt_rod_states[rod_start + x_idx], final_gt_rod_states[
                rod_start + y_idx]
            ax.scatter(rod_x, rod_y, marker='X', label=f'GT Final-Rod-{rod_idx}',
                       color=color, s=20 ** 2)
            yaw = final_gt_rod_states[rod_start + 2]
            u, v = scale * np.cos(yaw), scale * np.sin(yaw)
            ax.quiver(rod_x, rod_y, rod_x + u, rod_y + v, color=color)

            min_x, min_y = min(rod_x, min_x), min(rod_y, min_y)
            max_x, max_y = max(rod_x, max_x), max(rod_y, max_y)

    if init_rod_states is not None and final_gt_rod_states is not None:
        for rod_idx in range(num_rods):
            rod_start = 0 + rod_idx * 3
            color = rod_colors[rod_idx * 2 + 1]
            init_rod_xy = init_rod_states[rod_start:rod_start + 2]
            final_rod_xy = final_gt_rod_states[rod_start:rod_start + 2]

            ax.arrow(init_rod_xy[x_idx], init_rod_xy[y_idx],
                     final_rod_xy[x_idx] - init_rod_xy[x_idx],
                     final_rod_xy[y_idx] - init_rod_xy[y_idx], color='r',
                     linewidth=2, zorder=2.0, width=0.001, length_includes_head=True)

    if type(color) is np.ndarray:
        ax.scatter(x_pred[0, x_idx], x_pred[0, y_idx], label=f'{label}-pred(encoder)',
                   marker='v', s=14 ** 2, color=color)
        yaw_pred = x_pred[0, z_idx]
        u, v = scale * np.cos(yaw_pred), scale * np.sin(yaw_pred)
        ax.quiver(x_pred[0, x_idx], x_pred[0, y_idx], x_pred[0, x_idx] + u, x_pred[0, y_idx] + v, color=color)
    else:
        ax.scatter(x_pred[0, x_idx], x_pred[0, y_idx], label=f'{label}-pred(encoder)',
                   marker='v', s=14 ** 2, color=color)
    ax.scatter(x_hat[:, x_idx], x_hat[:, y_idx], label=f'{label}-pred(sample)', alpha=0.3,
               color='r')
    yaw_hat = x_hat[:, z_idx]
    u, v = scale * np.cos(yaw_hat), scale * np.sin(yaw_hat)
    # ax.quiver(x_hat[:, x_idx], x_hat[:, y_idx], x_hat[:, x_idx]+u, x_hat[:, y_idx]+v, color=color, alpha=0.3)

    if final_gt_rod_states is not None:
        print(f"min_xy: {min_x:.4f}, {max_x:.4f},  max: ({max_x:.4f}, {max_y:.4f})")
        axes_th = 0.03
        if invert_x_axes:
            ax.set_xlim(max_x + axes_th, min_x - axes_th)
        else:
            ax.set_xlim(min_x - axes_th, max_x + axes_th)
        if invert_y_axes:
            ax.set_ylim(max_y + axes_th, min_y - axes_th)
        else:
            ax.set_ylim(min_y - axes_th, max_y + axes_th)

        ax.legend(loc=(1.04, 0))
    else:
        ax.legend(loc=3)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_error_histograms(end_state_errors, cost_errors, T_plan_errors, T_exec_errors, subplot_tool=False, show=True, save_filename=None):
    fig, axes = plt.subplots(4, figsize=(8, 16), dpi=300)
    
    sns.histplot(end_state_errors, ax=axes[0], label="end_state")
    sns.histplot(cost_errors, ax=axes[1])
    sns.histplot(T_plan_errors, ax=axes[2])
    sns.histplot(T_exec_errors, ax=axes[3])
    
    axes[0].set_title(f"End state errors | mean: {end_state_errors.mean():.2f} | max: {end_state_errors.max():.2f}")
    axes[1].set_title(f"Cost errors | mean: {cost_errors.mean():.2f} | max: {cost_errors.max():.2f}")
    axes[2].set_title(f"T_plan errors | mean: {T_plan_errors.mean():.2f} | max: {T_plan_errors.max():.2f}")
    axes[3].set_title(f"T_exec errors | mean: {T_exec_errors.mean():.2f} | max: {T_exec_errors.max():.2f}")
    
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    if subplot_tool:
        plt.subplot_tool()
    
    if save_filename is not None:
        plt.savefig(save_filename, bbox_inches='tight')

    if show:
        plt.show()


def plot_gt_and_model_prediction_for_2d_state(ax, X_gt, X_pred, x_idx, y_idx, color, z_idx=None, plot_legend=True,
                                              title='', xlabel='', ylabel='', label='', X_encoded=None):
    '''Scatter plot for ground truth and predicted values.
    TODO: Invert axes to match axes for isaac gym viewer.
    '''
    assert len(X_gt.shape) == 2 and len(X_pred.shape) == 2

    # First plot gt with orientation
    # Add some noise so that if all X_gt are at same location we get
    X_gt = np.copy(X_gt)
    X_gt += np.random.uniform(1e-4, 1e-3, X_gt.shape)
    ax.scatter(X_gt[:, x_idx], X_gt[:, y_idx], label=f'{label}-gt', color=1.0-color, s=14**2, marker='s')
    yaw = X_gt[:, z_idx]
    scale = 1
    u, v = scale * np.cos(yaw), scale * np.sin(yaw)
    ax.quiver(X_gt[:, x_idx], X_gt[:, y_idx], X_gt[:, x_idx]+u, X_gt[:, y_idx]+v)

    # Now plot prediction with orientation
    ax.scatter(X_pred[:, x_idx], X_pred[:, y_idx], label=f'{label}-pred', color=color, s=12**2, marker='o', alpha=0.3)
    yaw = X_pred[:, z_idx]
    scale = 1
    u, v = scale * np.cos(yaw), scale * np.sin(yaw)
    ax.quiver(X_pred[:, x_idx], X_pred[:, y_idx], X_pred[:, x_idx]+u, X_pred[:, y_idx]+v)

    if X_encoded is not None:
        ax.scatter(X_encoded[:, x_idx], X_encoded[:, y_idx], label=f'{label}-encoder', color='r', s=10**2, marker='v')

    if plot_legend:
        ax.legend(loc=(1.04, 0))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def calculate_sem_prediction_error(X_pred, X_gt):
    ''' Calculate sem prediction error. 
    
    TODO: Should generalize this function.
    X_pred: 2D array of predicted end-states and parameters (N_samples, N_data)
    X_gt:   2D array of ground truth end-states and parameters (N_samples, N_data)
    '''
    # assert X_pred.shape == X_gt.shape

    total_err = np.linalg.norm(X_pred - X_gt, axis=1)
    pusher_error = np.linalg.norm(X_pred[:, :2] - X_gt[:, :2], axis=1)
    pusher_orient_error = np.linalg.norm(X_pred[:, 2:3] - X_gt[:, 2:3], axis=1)
    rod0_error = np.linalg.norm(X_pred[:, 3:5] - X_gt[:, 3:5], axis=1)
    rod0_orient_error = np.linalg.norm(X_pred[:, 5:6] - X_gt[:, 5:6], axis=1)
    rod1_error = np.linalg.norm(X_pred[:, 6:8] - X_gt[:, 6:8], axis=1)
    rod1_orient_error = np.linalg.norm(X_pred[:, 8:9] - X_gt[:, 8:9], axis=1)
    T_exec_error = np.linalg.norm(X_pred[:, 9:10] - X_gt[:, 9:10], axis=1)
    costs_error = np.linalg.norm(X_pred[:, 10:11] - X_gt[:, 10:11], axis=1)
    T_plan_error = np.linalg.norm(X_pred[:, 11:12] - X_gt[:, 11:12], axis=1)

    return dict(
        total_err=total_err,
        pusher_error=pusher_error,
        pusher_orient_error=pusher_orient_error,
        rod0_error=rod0_error,
        rod0_orient_error=rod0_orient_error,
        rod1_error=rod1_error,
        rod1_orient_error=rod1_orient_error,
        T_exec_error=T_exec_error,
        costs_error=costs_error,
        T_plan_error=T_plan_error,
    )


def calculate_min_max_support_error(X_data, X_gen):
    assert X_data.shape[1] == X_gen.shape[1]
    assert X_data.shape[1] == 12

    min_error_for_gt_data_dict = dict(
        total_err=[],
        pusher_error=[],
        pusher_orient_error=[],
        rod0_error=[],
        rod0_orient_error=[],
        rod1_error=[],
        rod1_orient_error=[],
        T_exec_error=[],
        costs_error=[],
        T_plan_error=[],
    )
    min_error_for_gen_data_dict = copy.deepcopy(min_error_for_gt_data_dict)
    for i in range(X_data.shape[0]):
        error_dict = calculate_sem_prediction_error(X_data[i:i+1], X_gen)
        for k, v in error_dict.items():
            min_error_for_gt_data_dict[k].append(np.min(v))
    for i in range(X_gen.shape[0]):
        error_dict = calculate_sem_prediction_error(X_gen[i:i+1], X_data)
        for k, v in error_dict.items():
            min_error_for_gen_data_dict[k].append(np.min(v))

    return min_error_for_gt_data_dict, min_error_for_gen_data_dict


def plot_errors_for_gt_and_pred_distribution_match(total_min_error_for_gt_data_dict, total_min_error_for_sampled_data_dict,
                                                   subplot_tool=False, plot_together=False, save_fig=False):
    return
    if plot_together:
        fig, axes = plt.subplots(len(total_min_error_for_gt_data_dict.keys()), 2)
    axes_idx = 0
    for pred_var, pred_values in total_min_error_for_gt_data_dict.items():
        df = pd.DataFrame(pred_values, columns=[pred_var])
        df['cdf'] = df.rank(method = 'average', pct = True)

        if plot_together: 
            ax = axes[axes_idx, 0]
        else:
            fig = plt.figure(figsize=(16, 8))
            ax = fig.add_subplot(121)

        series = pd.Series(pred_values)
        n, bins, patches = ax.hist(series, bins=20)

        ax2 = ax.twinx()
        # ax2 = fig.add_subplot(122)
        df.sort_values(pred_var).plot(x=pred_var, y = 'cdf', grid = True, 
                                    ax=ax2, colormap='spring', linewidth=4)
        ax.set_title(f"Best prediction for GT: {pred_var}")
        
        # Not plot the sampled values
        if plot_together:
            ax = axes[axes_idx, 1]
        else:
            ax = fig.add_subplot(122)

        sampled_pred_values = total_min_error_for_sampled_data_dict[pred_var]
        df = pd.DataFrame(sampled_pred_values, columns=[pred_var])
        df['cdf'] = df.rank(method = 'average', pct = True)
        
        series = pd.Series(sampled_pred_values)
        n, bins, patches = ax.hist(series, bins=20)
        
        ax2 = ax.twinx()
        df.sort_values(pred_var).plot(x=pred_var, y = 'cdf', grid = True, 
                                    ax=ax2, colormap='winter', linewidth=4)
        ax.set_title(f"Best GT for Predection: {pred_var}")
        ax.set_xlabel(pred_var)

        axes_idx += 1
        
    fig.tight_layout()
    # plt.subplots_adjust(hspace=0.3)
    if subplot_tool:
        plt.subplot_tool()
    plt.show()


#### Plot Plans ####

def convert_sem_state_to_list_of_objects(sem_state, env_class):
    if env_class == PushRodEnv:
        if type(sem_state) is list or len(sem_state.shape) == 1:
            assert len(sem_state) == 9
            pusher_state = sem_state[0:3]
            rod0_state = sem_state[3:6]
            rod1_state = sem_state[6:9]
        elif type(sem_state) is np.ndarray and len(sem_state.shape) == 2:
            assert sem_state.shape[1] == 9
            pusher_state = sem_state[:, 0:3]
            rod0_state = sem_state[:, 3:6]
            rod1_state = sem_state[:, 6:9]
        else:
            raise ValueError(f"Invalid sem_state type: {type(sem_state)}")
        return [pusher_state, rod0_state, rod1_state]
    elif env_class == FrankaRodEnv:
        if type(sem_state) is list or len(sem_state.shape) == 1:
            assert len(sem_state) == 12
            pusher_state = np.hstack([sem_state[0:2],sem_state[3]])
            rod0_state = np.hstack([sem_state[4:6], sem_state[7]])
            rod1_state = np.hstack([sem_state[8:10], sem_state[11]])
        elif type(sem_state) is np.ndarray and len(sem_state.shape) == 2:
            assert sem_state.shape[1] == 12
            pusher_state = np.hstack([sem_state[:, 0:2], sem_state[:, 3].reshape(-1,1)])
            #rod0_state = sem_state[:, 5:7,9]
            rod0_state = np.hstack([sem_state[:, 4:6], sem_state[:,7].reshape(-1,1)])
            rod1_state = np.hstack([sem_state[:, 8:10], sem_state[:, 11].reshape(-1,1)])
        else:
            raise ValueError(f"Invalid sem_state type: {type(sem_state)}")
        return [pusher_state, rod0_state, rod1_state]


def plot_sem_plan(sem_state_list, multi_sem_state_list, env, **kwargs):
    kwargs['multi_sem_state_list'] = multi_sem_state_list
    if env.__class__ in (PushRodEnv, FrankaRodEnv):
        return _plot_sem_plan_rods(sem_state_list, env.__class__, env.n_envs, env.num_rods, **kwargs)
    elif env.__class__ in (FrankaBinEnv,):
        return _plot_sem_plan_bin(sem_state_list, env, **kwargs)


def _plot_sem_plan_rods(sem_state_list, env_class, n_envs, n_rods, use_diff_arrow_for_sim=False,
                  use_fixed_axes_limits=True, multi_sem_state_list=None,
                  title='Compare SEM pred vs Sim (gt)', xlabel='X', ylabel='Y',
                  show=True, save_filename=None):
    '''Scatter plot for ground truth and predicted values.

    sem_state_list: List of SEM states generated from a plan.
    multi_sem_state_list: List of array of SEM states generated from either 
        running multiple envs or samples from  the decoder.
    TODO: Invert axes to match axes for isaac gym viewer.
    '''
    assert type(sem_state_list) is list and len(sem_state_list) > 0
    plt.rcParams['font.size']=10
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111)

    def _plot_2d_position(ax, x, y, theta, label='', color='r', marker='.'):
        ax.scatter(x, y, label=label, color=color, marker=marker)
        if theta is not None:
            scale = 1
            u, v = scale * np.cos(theta), scale * np.sin(theta)
            ax.quiver(x, y, x + u, y + v, color=color)

    x_idx, y_idx, z_idx = 1, 0, 2
    markers = ['s', 'x', 'o']
    pusher_color = env_class.get_pusher_color()
    pusher_color_sem = pusher_color.tolist() + [0.3]  # Change alpha
    rod_colors_sem = env_class.get_rod_colors(n_rods, only_sem_colors=True)
    rod_colors_ig = env_class.get_rod_colors(n_rods, only_ig_colors=True)

    if multi_sem_state_list is not None:
        colors = sns.color_palette("Paired", 3*2)
        color_idx = [0, 2, 4]
    else:
        colors = sns.color_palette("Set2", 3)
        color_idx = [0, 1, 2]

    #if task is not None:
    #    pose = task.goal_pos.copy()
    #    dims = task.goal_dims.copy()
    #    pose[0] -= (dims[0] / 2.)
    #    pose[1] -= (dims[1] / 2.)
    #    color = 'auto'
    #    patch = matplotlib.patches.Rectangle(pose, dims[0], dims[1], fill=False,
    #                                         linewidth=10)
    #    ax.add_patch(patch)

    for path_idx in range(len(sem_state_list)):
        sem_state = sem_state_list[path_idx]
        pusher_state, rod0_state, rod1_state = convert_sem_state_to_list_of_objects(sem_state, env_class)
        label_list = ['Pusher-SEM', 'Rod0-SEM', 'Rod1-SEM'] if path_idx == 0 else ['', '', '']
        sem_colors = [pusher_color_sem] + [rod_colors_sem[j] for j in range(n_rods)]
        _plot_2d_position(ax, pusher_state[x_idx], pusher_state[y_idx], pusher_state[z_idx], 
                          label=label_list[0], color=sem_colors[0], marker=markers[0])
        _plot_2d_position(ax, rod0_state[x_idx], rod0_state[y_idx], rod0_state[z_idx],
                          label=label_list[1], color=sem_colors[1], marker=markers[1])
        _plot_2d_position(ax, rod1_state[x_idx], rod1_state[y_idx], rod1_state[z_idx],
                          label=label_list[2], color=sem_colors[2], marker=markers[2])
        if path_idx > 0:
            prev_sem_state = sem_state_list[path_idx - 1]
            prev_pusher_state, prev_rod0_state, prev_rod1_state = convert_sem_state_to_list_of_objects(prev_sem_state, env_class)
            prev_next_states = [(prev_pusher_state, pusher_state), (prev_rod0_state, rod0_state), (prev_rod1_state, rod1_state)]

            obj_idx = 0
            for prev_obj_state, curr_obj_state in prev_next_states:
                ax.arrow(prev_obj_state[x_idx], prev_obj_state[y_idx],
                         curr_obj_state[x_idx] - prev_obj_state[x_idx], curr_obj_state[y_idx] - prev_obj_state[y_idx],
                         color=sem_colors[obj_idx], alpha=0.4, linewidth=2, zorder=2.0, width=0.001, length_includes_head=True)
                obj_idx += 1
        
        if multi_sem_state_list is not None:
            multi_sem_state = multi_sem_state_list[path_idx]
            pusher_state_2d, rod0_state_2d, rod1_state_2d = convert_sem_state_to_list_of_objects(multi_sem_state, env_class)
            label_list = ['Pusher-Sim (GT)', 'Rod0-Sim (GT)', 'Rod1-Sim (GT)'] if path_idx == 0 else ['', '', '']
            ig_colors = [pusher_color] + [rod_colors_ig[j] for j in range(n_rods)]
            _plot_2d_position(ax, pusher_state_2d[:, x_idx], pusher_state_2d[:, y_idx], pusher_state_2d[:, z_idx], 
                              label=label_list[0], color=ig_colors[0], marker=markers[0])
            _plot_2d_position(ax, rod0_state_2d[:, x_idx], rod0_state_2d[:, y_idx], rod0_state_2d[:, z_idx],
                               label=label_list[1], color=ig_colors[1], marker=markers[1])
            _plot_2d_position(ax, rod1_state_2d[:, x_idx], rod1_state_2d[:, y_idx], rod1_state_2d[:, z_idx],
                               label=label_list[2], color=ig_colors[2], marker=markers[2])
            if path_idx > 0:
                prev_multi_sem_state = multi_sem_state_list[path_idx - 1]
                prev_pusher_state_2d, prev_rod0_state_2d, prev_rod1_state_2d = convert_sem_state_to_list_of_objects(prev_multi_sem_state, env_class)
                prev_next_states = [(prev_pusher_state_2d, pusher_state_2d), 
                                    (prev_rod0_state_2d, rod0_state_2d), 
                                    (prev_rod1_state_2d, rod1_state_2d)]
                obj_idx = 0
                for prev_obj_state, curr_obj_state in prev_next_states:
                    prev_obj_state_mean = np.mean(prev_obj_state, axis=0)
                    curr_obj_state_mean = np.mean(curr_obj_state, axis=0)
                    if use_diff_arrow_for_sim:
                        patch = FancyArrowPatch((prev_obj_state_mean[x_idx], prev_obj_state_mean[y_idx]),
                                        (curr_obj_state_mean[x_idx],
                                        curr_obj_state_mean[y_idx]),
                                        linestyle='dashed',
                                        arrowstyle='-|>',
                                        mutation_scale=20,
                                        color=colors[color_idx[obj_idx]+1],
                                        alpha=1.0)
                        ax.add_patch(patch)
                    else:
                        ax.arrow(prev_obj_state_mean[x_idx], prev_obj_state_mean[y_idx],
                                curr_obj_state_mean[x_idx] - prev_obj_state_mean[x_idx], 
                                curr_obj_state_mean[y_idx] - prev_obj_state_mean[y_idx],
                                color=ig_colors[obj_idx], 
                                alpha=1.0, 
                                linewidth=2, 
                                zorder=2.0, 
                                width=0.001, 
                                length_includes_head=True)
                    obj_idx += 1

    if use_fixed_axes_limits:
        if env_class == PushRodEnv:
            ax.set_xlim(-0.3, 0.3)
            ax.set_ylim(-0.1, 0.5)
        else:
            ax.set_xlim(-0.8, 0.8)
            ax.set_ylim(-0.8, 0.8)

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # ax.legend(loc=(1.04, 0), fontsize=16)     # 1.04 is slightly outside
    ax.legend(loc=3)   # 3 is Lower left (https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.legend.html)
    
    if save_filename is not None:
        plt.savefig(save_filename, bbox_inches='tight')
        logging.info(f"Did save plot: {save_filename}")
    if show:
        plt.show()


def _plot_sem_plan_bin(sem_state_list, env, use_diff_arrow_for_sim=False, 
                  use_fixed_axes_limits=True, multi_sem_state_list=None,
                  title='Compare SEM pred vs Sim (gt)', xlabel='Y (m)', ylabel='X (m)', 
                  show=True, save_filename=None):
    '''Scatter plot for ground truth and predicted values.

    sem_state_list: List of SEM states generated from a plan.
    multi_sem_state_list: List of array of SEM states generated from either 
        running multiple envs or samples from  the decoder.
    '''
    assert type(sem_state_list) is list and len(sem_state_list) > 0
    plt.rcParams['font.size']=10
    fig = plt.figure(figsize=(12, 6), dpi=100)
    ax = fig.add_subplot(111)

    # plot table, tray, and bin
    xs, ys = env.table_shape.exterior.xy
    ax.fill(np.array(ys), np.array(xs), alpha=0.5, fc='lightgray', ec='none', label='Table')
    
    xs, ys = env.tray_shape.exterior.xy
    ax.fill(np.array(ys), np.array(xs), alpha=0.5, fc='dimgray', ec='none', label='Tray')
    
    bin_shapes = env.bin_shapes
    xs, ys = bin_shapes[0].exterior.xy
    ax.fill(np.array(ys), np.array(xs), alpha=0.5, fc='orange', ec='none', label='Bin Close')
    xs, ys = bin_shapes[1].exterior.xy
    ax.fill(np.array(ys), np.array(xs), alpha=0.5, fc='gold', ec='none', label='Bin Far')

    # plot block trajs
    def plot_block_traj(traj, ax, color, alpha):
        # matplotlib x and y are not the IG x and y.
        # IG x is negative matplotlib y, IG y is positive matplotlib x
        ax.scatter(traj[:, 1], traj[:, 0], color=color, alpha=alpha)

        for t in range(len(traj) - 1):
            ax.arrow(
                traj[t, 1], traj[t, 0],
                traj[t + 1, 1] - traj[t, 1], traj[t + 1, 0] - traj[t, 0],
                width=5e-3, length_includes_head=True,
                color=color, alpha=alpha
            )

    pred_block_trajs = np.array([sem_state.reshape(-1, 3) for sem_state in sem_state_list])
    gt_block_trajs = np.array([
        [sem_state.reshape(-1, 3) for sem_state in lst]
        for lst in multi_sem_state_list
    ])
    for block_idx, block_color in enumerate(env.block_colors):
        plot_block_traj(pred_block_trajs[:, block_idx], ax, block_color, 0.4)
        # Only plotting 1 gt traj b/c otherwise too crowded
        plot_block_traj(gt_block_trajs[:, 0, block_idx], ax, block_color, 1)

    ax.set_xlim(-0.7, 0.3)
    ax.set_ylim(0.2, 0.7)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=3, handles=[
        Patch(color='lightgray', alpha=0.5, label='Table'),
        Patch(color='dimgray', alpha=0.5, label='Tray'),
        Patch(color='orange', alpha=0.5, label='Bin Close'),
        Patch(color='gold', alpha=0.5, label='Bin Far'),
        Patch(alpha=0.4, label='SEM (Pred)'),
        Patch(alpha=1, label='GT (Sim)'),
    ])
    
    if save_filename is not None:
        plt.savefig(save_filename, bbox_inches='tight')
        logging.info(f"Did save plot: {save_filename}")
    if show:
        plt.show()

        
def plot_planner_results_for_iterative_config_to_wandb(planner_results):
    valid_keys = []                
    for k in planner_results.keys():
        # check if valid iteration key
        if k == planner_results[k]['iter']:
            valid_keys.append(k)
    if len(valid_keys) > 0:
        valid_keys_sorted = sorted(valid_keys)
        config_keys = ['all', 'config-0', 'config-1', 'config-2']
        for config_key in config_keys:
            stat_iters = []
            stat_searched, stat_found = [], []
            stat_success_exec = []
            for k in valid_keys_sorted:
                if planner_results[k]['results'].get(config_key) is not None:
                    stat_searched.append(planner_results[k]['results'][config_key]['searched'])
                    found_ratio = float(planner_results[k]['results'][config_key]['found']) / stat_searched[-1]
                    stat_found.append(found_ratio) 
                    success_exec_ratio = float(planner_results[k]['results'][config_key]['success_exec']) / stat_searched[-1]
                    stat_success_exec.append(success_exec_ratio)
                    stat_iters.append(k)
            if len(stat_iters) > 0:
                # Plot results
                searched_data = [[x, y] for (x, y) in zip(stat_iters, stat_searched)]
                table = wandb.Table(data=searched_data, columns = ["x", "y"])
                wandb.log({f"plans_searched_{config_key}" : wandb.plot.line(table, "x", "y", stroke=None, title="Plans searched")})

                found_data = [[x, y] for (x, y) in zip(stat_iters, stat_found)]
                table = wandb.Table(data=found_data, columns = ["x", "y"])
                wandb.log({f"plans_found_{config_key}" : wandb.plot.line(table, "x", "y", stroke=None, title="Plans found")})

                success_exec_data = [[x, y] for (x, y) in zip(stat_iters, stat_success_exec)]
                table = wandb.Table(data=success_exec_data, columns = ["x", "y"])
                wandb.log({f"plans_success_exec_{config_key}" : wandb.plot.line(table, "x", "y", stroke=None, title="Plans success exec")})

                logging.info(f"Did plot prev planner results: {config_key}")


def precision_recall_curve(deviations_from_planning, pred_test_deviations):
    import matplotlib.pyplot as plt
    thresholds = np.linspace(0.005, 0.5, 500)
    recalls = []
    precisions = []
    thresholds_with_data = []
    for threshold in thresholds:
        true_pos = np.sum(np.logical_and(pred_test_deviations > threshold, deviations_from_planning > threshold))
        num_classified_as_pos = np.sum(pred_test_deviations > threshold)
        num_true_pos = np.sum(deviations_from_planning > threshold)
        if num_classified_as_pos == 0 or true_pos == 0:
            continue
        thresholds_with_data.append(threshold)
        precisions.append(true_pos / num_classified_as_pos)
        recalls.append(true_pos / num_true_pos)
    plt.plot(thresholds_with_data, recalls, label="recall")
    plt.plot(thresholds_with_data, precisions, label="precision")
    plt.legend()
    plt.show()
    return precisions, recalls, thresholds_with_data


def sample_complexity_test(cfg, deviation_model, train_states_and_params, train_deviations,
                           validation_states_and_params, validation_deviations, states_and_parameters_from_planning,
                           deviations_from_planning, experiment):
    if cfg.get('sample_complexity_test', False):
        test_losses = []
        range_num_training_samples = range(4, 20)
        for num_training_samples in range_num_training_samples:
            train_and_fit_model(cfg, deviation_model, train_states_and_params[:num_training_samples],
                                train_deviations[:num_training_samples],
                                states_and_parameters_from_planning, deviations_from_planning,
                                validation_states_and_params, validation_deviations, experiment, is_classification,
                                validate_on_split_data=False)
            pred_planning_deviations = deviation_model.predict(states_and_parameters_from_planning,
                                                               already_transformed_state_vector=True)
            loss = deviation_model.evaluate_loss(deviations_from_planning, pred_planning_deviations)
            test_losses.append(np.mean(loss))
        exp_name = deviation_model.__class__.__name__
        file_name_start = "/home/lagrassa/git/plan-abstractions/data/sample_complexity"
        np.save(f"{file_name_start}/{exp_name}_test_losses.npy", test_losses)
        np.save(f"{file_name_start}/{exp_name}_test_num_training_samples.npy", range_num_training_samples)
