from shutil import copy, copytree
from pillar_state import State

from ..envs import *
from ..skills import *


def insert_processed_relative_poses_gen(env_type, skill_type, sem_state_obj_names, anchor_obj_name):
    env_cls = eval(env_type)
    skill_cls = eval(skill_type)

    def insert_processed_relative_poses(d, data_idx, shard_idx):
        if 'relative_states' not in d:
            d['relative_states'] = {}
        d['relative_states'][anchor_obj_name] = {
                'initial_states': [],
                'end_states': []
            }

        if 'relative_parameters' not in d:
            d['relative_parameters'] = {}
        d['relative_parameters'][anchor_obj_name] = []

        for env_idx, end_state_str in enumerate(d['exec_data']['end_states']):
            init_pillar_state = State.create_from_serialized_string(d["initial_states"][env_idx])
            initial_state = env_cls.pillar_state_to_sem_state(init_pillar_state, sem_state_obj_names, 
                                    anchor_obj_name=anchor_obj_name, ref_pillar_state=init_pillar_state)

            end_pillar_state = State.create_from_serialized_string(end_state_str)
            end_state = env_cls.pillar_state_to_sem_state(end_pillar_state, sem_state_obj_names, 
                                anchor_obj_name=anchor_obj_name, ref_pillar_state=init_pillar_state)

            d['relative_states'][anchor_obj_name]['initial_states'].append(initial_state)
            d['relative_states'][anchor_obj_name]['end_states'].append(end_state)

            parameters = d['parameters'][env_idx]
            relative_parameters = skill_cls.parameters_to_relative_parameters(parameters, init_pillar_state, anchor_obj_name)
            d['relative_parameters'][anchor_obj_name].append(relative_parameters)

        return d

    return insert_processed_relative_poses


def copy_non_data_files(hydra_dir, output_dir, cfg_path):
    copy(hydra_dir / 'collect_skill_data.log', output_dir)
    copy(cfg_path, output_dir)
    try:
        copytree(hydra_dir / '.hydra', output_dir / '.hydra')
    except FileExistsError:
        pass