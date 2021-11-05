from ..envs import *

import logging
# import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
# Need to import before torch

import torch
from torch import Dataset
from ..utils.utils import from_numpy
from omegaconf.dictconfig import DictConfig

from ..skills import *
from .data_utils import process_datas, unnormalize_data, load_datas_from_dir_with_tags, get_datas
from ..envs.franka_bin_env import FrankaBinEnv


def filter_correct_X_for_push_rod_env(X, conds, object_masks):
    '''This only works for push rod task and that too hard codes the number of rods.'''
    # Rod positions beyond 1m are most likely due to some error
    X_abs = np.abs(np.copy(X))
    X_incorrect = ((X_abs[:, 3] > 1) | (X_abs[:, 4] > 1) | (X_abs[:, 6] > 1) | (X_abs[:, 7] > 1)).astype(np.int32)
    valid_mask = X_incorrect == 0
    X = X[valid_mask]
    conds = conds[valid_mask]
    object_masks = object_masks[valid_mask]
    return X, conds


def filter_correct_X_for_franka_rod_env_freespace_pd_skill(X, conds, object_masks, cfg):
    pos_tol = 0.002  # 2 mm
    yaw_tol = 0.05   # radians
    valid_data = np.ones(X.shape[0], dtype=np.bool)
    for i in range(4, X.shape[1]-3, 4):
        end_rod_pos, end_rod_yaw = X[:, i:i+3], X[:, i+3:i+4]
        if cfg['state_info']['use_state_diff_in_end_state']:
            valid_pos_th = np.all(np.abs(end_rod_pos) < pos_tol, axis=1)
            valid_yaw_th = np.all(np.abs(end_rod_yaw) < yaw_tol, axis=1)

            valid_data = valid_data & (valid_pos_th & valid_yaw_th)
    
    return X[valid_data], conds[valid_data], object_masks[valid_data]


def filter_correct_X_for_franka_sweep_xyzyaw_skill(X, conds, object_masks, cfg):
    '''This code assumes two rods.'''
    # -1, -2 are the rods to move
    goal_pos = conds[:, -5:-3]
    goal_yaw = conds[:, -3:-2]
    ee_end_state_pos = X[:, :2]
    ee_end_state_yaw = X[:, 3:4]
    if cfg['state_info']['use_state_diff_in_end_state']:
        ee_end_state_pos += conds[:, :2]
        ee_end_state_yaw += conds[:, 3:4]
    
    pos_th = 0.01
    yaw_th = 0.05

    valid_pos_th = np.all(np.abs(ee_end_state_pos - goal_pos) < pos_th, axis=1)
    # why +, since ee_end_state_yaw = -goal yaw, since different frames.
    valid_yaw_th = np.all(np.abs(ee_end_state_yaw + goal_yaw) < yaw_th, axis=1) 
    valid_data = valid_pos_th & valid_yaw_th

    return X[valid_data], conds[valid_data], object_masks[valid_data]

## ==== Franka Bin Env ====
def filter_correct_X_for_franka_bin_pick_place_skill(X, conds, object_masks, cfg):
    '''Filter objects in which more than 1 object moved.
    
    For pick-place skill only 1 object should move.
    '''
    pos_tol = 0.004  # 2 mm
    yaw_tol = 0.04   # radians

    num_objects_moved = np.zeros(X.shape[0], dtype=np.int32)
    for i in range(0, X.shape[1]-3, 3):
        X_init = conds[:, i:i+3]
        X_end = X[:, i:i+3]

        # X_diff = np.abs(X_end - (X_init - [0, 0, 0.015]))
        X_diff = np.abs(X_end - X_init)
        # envs_with_object_i_moved = (np.all(np.abs(X_diff[:, :3]) > pos_tol, axis=1)) | (np.abs(X_diff[:, 3]) < yaw_tol)
        envs_with_object_i_moved = np.all(np.abs(X_diff[:, :3]) < pos_tol, axis=1)
        num_objects_moved[envs_with_object_i_moved == False] += 1
    
    valid_data = num_objects_moved >= 1
    return X[valid_data], conds[valid_data], object_masks[valid_data]


def filter_correct_X_for_env(X, conds, object_masks, cfg):
    env_type = cfg['env']
    skill_type = cfg['skill']
    if env_type == 'PushRodEnv':
        return filter_correct_X_for_push_rod_env(X, conds, object_masks)
    elif env_type == 'FrankaRodEnv':
        if skill_type in ('FreeSpaceMoveLQRFranka', 'FreeSpaceMoveFranka'):
            return filter_correct_X_for_franka_rod_env_freespace_pd_skill(X, conds, object_masks, cfg)
        elif skill_type == 'LQRWaypointsXYZYawFranka':
            return filter_correct_X_for_franka_sweep_xyzyaw_skill(X, conds, object_masks, cfg)
        else:
            return X, conds, object_masks
    elif env_type == 'FrankaBinEnv':
        if skill_type == 'FrankaPickPlace':
            return filter_correct_X_for_franka_bin_pick_place_skill(X, conds, object_masks, cfg)
        else:
            return X, conds, object_masks
    else:
        raise ValueError(f"Invalid env cannot filter data for it: {env_type}")


def concatenate_datas(datas_list, use_state_diff_in_end_state=False):
    datas_conds = np.c_[datas_list['init_states'], datas_list['parameters']]
    end_states_key = 'end_states_diff' if use_state_diff_in_end_state else 'end_states'
    datas_X = np.c_[
        datas_list[end_states_key],
        datas_list['T_execs'],
        datas_list['costs'],
        datas_list['T_pis']
    ]
    datas_masks = np.c_[datas_list['object_masks']]
    return datas_X, datas_conds, datas_masks


class SEMDatasetFromFile(Dataset):

    def __init__(self, cfg, plot=False, datas=None, data_shuffle=True, split_data=True, 
                 data_root_key='root', data_tags_key='tags'):
        '''
        2D free space motion skill where skill param is the desired x,y location (and theta but ignoring that for now)

        Termination states are x,y states after applying a simple feedback controller
        Planning time is constant
        Execution cost is the distance between init state and end state
        Execution time scales also with this distance.
        '''
        self._cfg = cfg
        if datas is None:
            datas = get_datas(cfg, data_root_key, data_tags_key)

        env_cls = eval(cfg['env'])
        self._env_cls = env_cls
        skill_cls = eval(cfg['skill'])
        self._skill_cls = skill_cls

        anchor_obj_name = None
        if 'anchor_obj_name' in cfg['state_info']:
            anchor_obj_name = cfg['state_info']['anchor_obj_name']
        self._anchor_obj_name = anchor_obj_name
        process_data_kwargs = {} if 'process_data_kwargs' not in cfg else cfg['process_data_kwargs']
        processed_datas = process_datas(datas, list(cfg['sem_state_obj_names']), env_cls, 
                                        skill_cls, anchor_obj_name=anchor_obj_name, **process_data_kwargs)
        X, Conds, object_masks = concatenate_datas(processed_datas, self._cfg['state_info']['use_state_diff_in_end_state'])

        before_data_size = X.shape[0]
        X, Conds, object_masks = filter_correct_X_for_env(X, Conds, object_masks, cfg)
        logging.info(f"Before data size: {before_data_size}, after filter data size: {X.shape[0]}")
        self._processed_datas = processed_datas #TODO lagrassa dont do this in this class

        self._dim_state = processed_datas['init_states'].shape[1]
        self._dim_data = X.shape[1]
        self._dim_params = processed_datas['parameters'].shape[1]
        self._dim_cond = Conds.shape[1]
        # Remove Franka or robot mask.
        self._dim_masks = object_masks.shape[1] - 1

        logging.info(f"Total data size: {X.shape[0]}")

        # Normalize data
        self._normalization_type = cfg['normalization_type']
        if self._normalization_type == 'z_normalization':
            self._z_normalization_mean = np.array(cfg[self._normalization_type]['mean'])
            self._z_normalization_std = np.array(cfg[self._normalization_type]['std'])
            eps = 1e-6 
            X = (X - self._z_normalization_mean) / (self._z_normalization_std + eps)
            print(np.array_str(X.mean(axis=0), precision=4, suppress_small=True, max_line_width=120))
            print(np.array_str(X.std(axis=0), precision=4, suppress_small=True, max_line_width=120))
        elif self._normalization_type == 'none':
            pass
        else:
            raise ValueError("Invalid normalization type")

        # Process into dataset
        self._split_data = split_data
        if split_data:
            X_tr, X_t, Conds_tr, Conds_t, object_masks_tr, object_masks_t = train_test_split(
                X, Conds, object_masks, test_size=cfg['test_size'], shuffle=data_shuffle)
            # Set state vars
            self._X_tr, self._X_t = from_numpy(X_tr), from_numpy(X_t)
            self._Conds_tr, self._Conds_t = from_numpy(Conds_tr), from_numpy(Conds_t)
            self._object_masks_tr, self._object_masks_t = from_numpy(object_masks_tr[:, 1:]), from_numpy(object_masks_t[:, 1:])
        else:
            X_tr, X_t, Conds_tr, Conds_t = X, None, Conds, None
            object_masks_tr, object_masks_t = object_masks, None
            self._X_tr, self._Conds_tr = from_numpy(X_tr), from_numpy(Conds_tr)
            self._object_masks_tr = from_numpy(object_masks_tr[:, 1:])

        self._len = len(X)
        self._train_idxs = np.arange(len(X_tr))
        self._test_idxs = np.arange(len(X_tr), self._len) if split_data else np.array([])

        self.sem_data_idx_to_data_env_idx = processed_datas['data_idx_env_idx']

        if plot:
            end_states = processed_datas['end_states']
            fig, axes = plt.subplots(2, 1, figsize=(8, 8))
            axes[0].scatter(processed_datas['parameters'][:, 0], processed_datas['parameters'][:, 1],
                            c=np.array((0.8, 0.1, 0.8)), label="params")
            axes[0].set_xlabel("Params")
            axes[1].scatter(end_states[:, 0], end_states[:, 1], c=np.array((0, 0.8, 0.2)), label="end_states")
            if 'rod0' in cfg['sem_state_obj_names']:
                axes[1].scatter(end_states[:, 3], end_states[:, 4], c='r', label='rod0_end_state')
            if 'rod1' in cfg['sem_state_obj_names']:
                axes[1].scatter(end_states[:, 6], end_states[:, 7], c='y', label='rod1_end_state')
            axes[1].set_xlabel("End states")
            plt.legend()
            plt.show()


    def add_datas_to_trainer(self, datas):
        if type(datas) is not list:
            datas = [datas]
        process_data_kwargs = {} if 'process_data_kwargs' not in self._cfg else self._cfg['process_data_kwargs']
        processed_datas = process_datas(datas, list(self._cfg['sem_state_obj_names']), self._env_cls,
                                        self._skill_cls, anchor_obj_name=self._anchor_obj_name, **process_data_kwargs)
        if len(processed_datas['end_states']) == 0:
            logging.warn("Cannot add data to trainer. Processed data returns none.")
            return

        X, Conds, object_masks = concatenate_datas(processed_datas, self._cfg['state_info']['use_state_diff_in_end_state'])
        before_filter_data_size = X.shape[0]
        X, Conds, object_masks = filter_correct_X_for_env(X, Conds, object_masks, self._cfg['env'])
        logging.info(f"Before filter data size: {before_filter_data_size} New total data size: {X.shape[0]}")
        if X.shape[0] == 0:
            return
        X = self.normalize_data(X)
        X_tr, Conds_tr, object_masks_tr = from_numpy(X), from_numpy(Conds), from_numpy(object_masks)
        self._X_tr = torch.cat([self._X_tr, X_tr], axis=0)
        self._Conds_tr = torch.cat([self._Conds_tr, Conds_tr], axis=0)
        self._object_masks_tr = torch.cat([self._object_masks_tr, object_masks_tr], axis=0)
        assert not self._split_data, "Overall data length is wrong now. Just don't use it with this."

        self._len = self._X_tr.size(0)
        self._train_idxs = np.arange(self._len)
        logging.info(f"New data has length: {self._len}")

    @property
    def dim_state(self):
        ''' Dimension of the state used for init/end state predictions
        '''
        return self._dim_state

    @property
    def dim_data(self):
        ''' Dimension of everything the SEM predicts. 
        
        Should be dim_state + 3 (T_exec, cost, T_pi)
        '''
        return self._dim_data

    @property
    def dim_params(self):
        ''' Dimension of skill parameters
        '''
        return self._dim_params

    @property
    def dim_cond(self):
        ''' Dimension of the conditioned part of the CVAE, used as input for SEM predictions.

        Should be dim_state + dim_params
        '''
        return self._dim_cond

    @property
    def dims(self):
        return {
            'state': self.dim_state,
            'data': self.dim_data,
            'params': self.dim_params,
            'cond': self.dim_cond,
        }

    @property
    def env_class(self):
        return self._env_cls
    
    @property
    def env_name(self):
        return self._cfg['env']

    @property
    def train_idxs(self):
        return self._train_idxs.copy()

    @property
    def test_idxs(self):
        return self._test_idxs.copy()
    
    @property
    def train_data_size(self):
        return self._X_tr.shape[0]
    
    def normalize_data(self, data):
        norm_type = self._normalization_type
        if norm_type == 'z_normalization':
            eps = 1e-6
            out = (data - self._z_normalization_mean) / (self._z_normalization_std + eps)
        elif norm_type == 'none':
            out = data
        else:
            raise ValueError("Invalid normalization type")
        return out

    def unnormalize_data(self, X):
        unnormalize_data(X, self._normalization_type, self._z_normalization_mean, self._z_normalization_std)

    def plot_preds(self, x, cond, x_hat, nlls=None):  # TODO this somehow needs to be different for each dataset
        start_states = cond[:, :2]
        desired_states_start_idx = 3
        if 'rod0' in self._cfg['sem_state_obj_names'] and \
                'rod1' in self._cfg['sem_state_obj_names']:
            desired_states_start_idx += 6
        desired_states = cond[:, desired_states_start_idx:]  # pretty close to gt, omitting for this viz

        # Pusher end states
        end_states_gt = x[:, :2]
        if self._cfg['state_info']['use_state_diff_in_end_state']:
            # We are already predicting 
            end_states_pred = x_hat[:, :2]
        else:
            end_states_pred = x_hat[:, :2] - start_states

        fig = plt.figure(figsize=(10, 10))
        plt.scatter(end_states_gt[:, 0], end_states_gt[:, 1], label='GT', color='blueviolet')

        rgba_colors = np.zeros((len(end_states_pred), 4))
        rgba_colors[:, :3] = (1, 0.5, 0.055)
        if nlls is not None:
            rgba_colors[:, 3] = np.clip(1 - (nlls - nlls.min()) / (nlls.max() - nlls.min()), 0.01, 0.99)
        plt.scatter(end_states_pred[:, 0], end_states_pred[:, 1], label='Pred')  # c=rgba_colors)

        # plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.title('Predicted and Ground Truth Termination States')
        # plt.show()

        return fig

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if idx < len(self._train_idxs):
            x, cond, object_mask = self._X_tr[idx], self._Conds_tr[idx], self._object_masks_tr[idx]
        else:
            assert self._X_t is not None, "Should not come here."
            idx -= len(self._train_idxs)
            x, cond, object_mask = self._X_t[idx], self._Conds_t[idx], self._object_masks_t[idx]

        return {
            'x': x,
            'cond': cond,
            'object_mask': object_mask,
        }


class SEMMockDataset(Dataset):

    def __init__(self, cfg):
        ''' 
        Simulates a 2D free space motion skill where skill param is the desired movement angle and magnitude

        Termination states are a GMM w/ 3 means placed 30 degrees apart from the target angle.
        Planning time is constant from a narrow normal distribution.
        Execution cost is the distance between initial and termination state.
        Execution time scales also with this distance.
        '''
        N_data = 100000

        # SEM inputs
        init_states = np.random.uniform([-10, -10], [10, 10], (N_data, 2))
        angles = np.random.uniform(0, 2 * np.pi, N_data)
        mags = np.clip(np.random.normal(3, 1, N_data), 2, 4)
        thetas = np.c_[angles, mags]
        Conds = np.c_[init_states, thetas]

        # SEM outputs
        delta_mean_angles = np.random.choice([-np.deg2rad(30), 0, np.deg2rad(30)], N_data)
        actual_mean_angles = angles + delta_mean_angles
        actual_angles = actual_mean_angles + np.random.normal(scale=np.deg2rad(1), size=N_data)
        end_states = init_states + mags[:, None] * np.c_[np.cos(actual_angles), np.sin(actual_angles)]

        T_pis = np.clip(np.random.normal(loc=10, scale=1, size=N_data), 7, 13)
        Cs = np.linalg.norm(end_states - init_states, axis=1)
        ts = Cs / 5

        X = np.c_[end_states, ts, Cs, T_pis]

        # Process into dataset
        X_tr, X_t, Conds_tr, Conds_t = train_test_split(X, Conds, test_size=cfg['test_size'])

        self._dim_state = init_states.shape[1]
        self._dim_data = X.shape[1]
        self._dim_params = thetas.shape[1]
        self._dim_cond = Conds.shape[1]

        # Set state vars
        self._X_tr, self._X_t = from_numpy(X_tr), from_numpy(X_t)
        self._Conds_tr, self._Conds_t = from_numpy(Conds_tr), from_numpy(Conds_t)

        self._len = len(X)
        self._test_idxs = np.arange(len(X_tr), self._len)
        self._train_idxs = np.arange(len(X_tr))

    @property
    def dim_state(self):
        ''' Dimension of the state used for init/end state predictions
        '''
        return self._dim_state

    @property
    def dim_data(self):
        ''' Dimension of everything the SEM predicts. 
        
        Should be dim_state + 3 (T_exec, cost, T_pi)
        '''
        return self._dim_data

    @property
    def dim_params(self):
        ''' Dimension of skill parameters
        '''
        return self._dim_params

    @property
    def dim_cond(self):
        ''' Dimension of the conditioned part of the CVAE, used as input for SEM predictions.

        Should be dim_state + dim_params
        '''
        return self._dim_cond

    @property
    def train_idxs(self):
        return self._train_idxs.copy()

    @property
    def test_idxs(self):
        return self._test_idxs.copy()

    def plot_preds(self, x, cond, x_hat, nlls):
        start_states = cond[:, :2]
        angles = cond[:, 2]

        end_states_gt = x[:, :2]
        end_states_pred = x_hat[:, :2]

        diffs_gt = end_states_gt - start_states
        diffs_pred = end_states_pred - start_states

        c, s = np.cos(-angles), np.sin(-angles)
        coord_x_gt, coord_y_gt = diffs_gt[:, 0], diffs_gt[:, 1]
        coord_x_pred, coord_y_pred = diffs_pred[:, 0], diffs_pred[:, 1]

        rot_diffs_gt = np.c_[c * coord_x_gt - s * coord_y_gt, s * coord_x_gt + c * coord_y_gt]
        rot_diffs_pred = np.c_[c * coord_x_pred - s * coord_y_pred, s * coord_x_pred + c * coord_y_pred]

        fig = plt.figure(figsize=(10, 10))
        plt.scatter(rot_diffs_gt[:, 0], rot_diffs_gt[:, 1], label='GT', color=[0.122, 0.47, 0.705])

        rgba_colors = np.zeros((len(rot_diffs_pred), 4))
        rgba_colors[:, :3] = (1, 0.5, 0.055)
        rgba_colors[:, 3] = np.clip(1 - (nlls - nlls.min()) / (nlls.max() - nlls.min()), 0.01, 0.99)
        plt.scatter(rot_diffs_pred[:, 0], rot_diffs_pred[:, 1], label='Pred', c=rgba_colors)

        # plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.title('Predicted and Ground Truth Termination States')

        return fig

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if idx < len(self._train_idxs):
            x, cond = self._X_tr[idx], self._Conds_tr[idx]
        else:
            idx -= len(self._train_idxs)
            x, cond = self._X_t[idx], self._Conds_t[idx]

        return {
            'x': x,
            'cond': cond
        }
