import logging
logger = logging.getLogger(__name__)

from pathlib import Path
from collections import OrderedDict
from .water_models import *

import numpy as np
import wandb
from omegaconf import OmegaConf

from torch_utils import get_numpy, from_numpy

from ..envs import *
from ..learning.data_utils import unnormalize_data


def create_sem_wrapper_from_cfg(cfg, cache_dir='/tmp',skill_cls=None, sem_state_obj_names=None):
    learned_models = (),
    analytical_models = ('SEMSimpleFreeSpace', 'SEMAnalyticalRodsAndRobot', "SEMAnalyticalInsert", "SEMSimpleDrawerOpen", "SEMAnalyticalDrawerAndRobot")
    dim_state = cfg.get('dim_state', 12)
    is_analytical = False
    if cfg['type'] in learned_models:
        sem = eval(cfg['type']).load_from_checkpoint(ckpt_file.name).cuda()
        sem.train(False)
    elif cfg['type'] == "SEMAnalyticalDrawerAndRobot":
        drawer_edge_dims = cfg["drawer_edge_dims"]
        sem = eval(cfg['type'])(2, dim_state, drawer_edge_dims)
        is_analytical = True
    elif "Linear" in cfg['type'] or "RFR" in cfg['type']:
        sem = eval(cfg['type'])(cfg["model_cfg"])
    else:
        sem = eval(cfg['type'])(2, dim_state)
        is_analytical=True

    pillar_state_convert = cfg.get('pillar_state_convert', True)
    env_cls = eval(cfg['env'])
    sem_wrapper = SEMWrapper(sem, env_cls, skill_cls, {}, {'num_samples':1}, pillar_state_convert=pillar_state_convert, sem_state_obj_names=sem_state_obj_names, is_analytical=is_analytical)

    return sem_wrapper


class SEMWrapper:

    def __init__(self, sem, env_cls, skill_cls, train_cfg, predict_cfg, sem_state_obj_names=None, is_analytical=False, pillar_state_convert=True):
        self._sem = sem
        self._env_cls = env_cls
        self._skill_cls = skill_cls
        self._train_cfg = train_cfg
        self._predict_cfg = predict_cfg
        self._pillar_state_convert = pillar_state_convert
        if self._pillar_state_convert:
            if sem_state_obj_names is None:
                self._sem_state_obj_names = list(self._train_cfg['data']['sem_state_obj_names'])
            else:
                self._sem_state_obj_names = sem_state_obj_names
            self._use_diff_states = True #self._train_cfg['data']['state_info']['use_state_diff_in_end_state']


        self._is_analytical = is_analytical

        self._anchor_obj_name = None

    @property
    def anchor_obj_name(self):
        return self._anchor_obj_name
    
    @property
    def sem_model(self):
        return self._sem

    @property
    def sem(self):
        return self._sem
    
    @property
    def train_cfg(self):
        return self._train_cfg

    def get_gnn_sem_model_predictions(self, pillar_state, parameters):
        sem_state = self._env_cls.pillar_state_to_sem_state(pillar_state, self._sem_state_obj_names, 
                                    anchor_obj_name=self._anchor_obj_name, ref_pillar_state=pillar_state)
        sem_state_indexes = self._env_cls.get_state_indexes_for_sem_objects(self._sem_state_obj_names)
        sem_dict_repr = OrderedDict()
        for object_idx, object_name in enumerate(self._sem_state_obj_names):
            init_state = sem_state[sem_state_indexes[object_idx]]
            sem_dict_repr[f'node_{object_idx}'] = {
                'initial_state': init_state, 'name': object_name, 'is_robot': 'franka' in object_name}
        
        # TODO(Mohit): Move this to config similar to predict_cfg
        add_all_edges = True
        geom_data = self._sem.get_geom_data_for_sem_dict_repr(sem_dict_repr, parameters, self._sem_state_obj_names, add_all_edges)
        outputs = self._sem.sample(geom_data.to(self._sem.device), **self._predict_cfg)
        sem_dict_repr_hat = outputs['sem_dict_repr']

        # TODO: Normalization?
        if self._use_diff_states:
            for node_key in sem_dict_repr_hat.keys():
                sem_dict_repr_hat[node_key]['end_state'] += sem_dict_repr[node_key]['initial_state']

        if hasattr(self._sem, 'predicts_object_masks') and self._sem.predicts_object_masks:
            for node_key in sem_dict_repr_hat.keys():
                # object mask prediction: the object did not move
                if sem_dict_repr_hat[node_key]['mask'] < 0.5:
                    # Reset to initial state provided
                    sem_dict_repr_hat[node_key]['end_state'] = sem_dict_repr[node_key]['initial_state']
                else:
                    pass

        # TODO: Predict execution time, cost, plan time etc.
        T_execs = [1.0]
        costs = [1.0]
        T_plan = [1.0]

        end_state_sem = []
        node_idx = 0
        for node_key in sem_dict_repr_hat.keys():
            # Make sure that we save the correct end state for the correct object. 
            # Note that  we use sem_dict_repr below.
            assert self._sem_state_obj_names[node_idx] == sem_dict_repr[node_key]['name']
            end_state_sem.extend(sem_dict_repr_hat[node_key]['end_state'])
            node_idx += 1

        # TODO: Right now this code assumes only 1 output, should be easy to update though.
        end_states_sem = [end_state_sem]

        assert self._anchor_obj_name is None
        end_states = [
            self._env_cls.sem_state_to_pillar_state(
                _end_state_sem, pillar_state, self._sem_state_obj_names, anchor_obj_name=self._anchor_obj_name)
            for _end_state_sem in end_states_sem
        ]

        return {
            'end_states': end_states,
            'T_exec': T_execs,
            'costs': costs,
            'info_plan': {
                'T_plan': T_plan
            }
        }

    def __call__(self, state, parameters):
        if self._pillar_state_convert:
            return self.call_pillar_state(state, parameters)
        return self._sem.predict(state, parameters)

    def call_pillar_state(self, pillar_state, parameters):

        if not self._sem.fixed_input_size:
            return self.get_gnn_sem_model_predictions(pillar_state, parameters)
        sem_state = self._env_cls.pillar_state_to_sem_state(pillar_state, self._sem_state_obj_names,
                                    anchor_obj_name=self._anchor_obj_name, ref_pillar_state=pillar_state)

        if self._anchor_obj_name is not None:
            parameters = self._skill_cls.parameters_to_relative_parameters(
                                        parameters, pillar_state, self._anchor_obj_name)

        cond_np = np.r_[sem_state, parameters].reshape(1, -1)
        if self._is_analytical:
            outputs_np = self._sem.sample(cond_np, **self._predict_cfg)['x_hats']
        else:
            cond = from_numpy(cond_np).cuda()
            outputs = self._sem.sample(cond, **self._predict_cfg)
            outputs_np = get_numpy(outputs['x_hats'])

            outputs_np = unnormalize_data(outputs_np, self._normalization_type,
                                          z_normalization_mean=self._z_normalization_mean,
                                          z_normalization_std=self._z_normalization_std)

        end_states_sem = outputs_np[:, :self._sem.dim_state]
        if self._use_diff_states:
            end_states_sem += sem_state

        # If the mask has 0, then we should use the GT value directly
        if hasattr(self._sem, 'predicts_object_masks') and self._sem.predicts_object_masks:
            # Fixed input size implies an MLP being used.
            if self._sem.fixed_input_size:
                object_mask = outputs['object_mask']
                object_indexes = self._env_cls.get_state_indexes_for_sem_objects(self._sem_state_obj_names)
                object_indexes_for_masked_objects = [indexes for i, indexes in enumerate(object_indexes) 
                                                     if 'franka' not in self._sem_state_obj_names[i]]
                assert len(object_indexes_for_masked_objects) == len(object_mask)
                for object_idx, object_state_indexes in enumerate(object_indexes_for_masked_objects):
                    # Object did not move
                    if object_mask[object_idx] < 0.5:
                        end_states_sem[:, object_state_indexes] = sem_state[:, object_state_indexes]
            else:
                raise NotImplementedError

        T_execs = outputs_np[:, self._sem.dim_state]
        costs = outputs_np[:, self._sem.dim_state + 1]
        T_plan = outputs_np[:, self._sem.dim_state + 2]

        end_states = [
            self._env_cls.sem_state_to_pillar_state(
                end_state_sem, pillar_state, self._sem_state_obj_names, 
                anchor_obj_name=self._anchor_obj_name
            )
            for end_state_sem in end_states_sem
        ]

        return {
            'end_states': end_states,
            'T_exec': T_execs,
            'costs': costs,
            'info_plan': {
                'T_plan': T_plan
            }
        }
