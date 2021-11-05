from filelock import Timeout, FileLock
import pandas as pd
import copy

import logging
import numpy as np
from pathlib import Path

from omegaconf import OmegaConf


skill_data_csv_headers = [
    'data_tag',
    'root_tag',
    'root_n_iter',
    'seed',
    'skill_type',
    'env_type',
    'task_type',
    'hydra_dir',
    'data_dir',
    'used_trained_models',
    'sem_cfgs_repr',
    'data_dir_processed',
    'has_been_processed',
]

def str2bool(s):
    s_trim = str(s).strip()
    if s_trim in ("True", "true"):
        return True
    elif s_trim in ("False", "false"):
        return False
    else:
        raise ValueError("WTF")

skill_data_csv_converters = {'has_been_processed': lambda x: str2bool(x)}


model_csv_headers = [
    'model_tag',
    'root_tag',
    'root_n_iter',
    'seed',
    'model_type',
    'skill_type',
    'env_type',
    'hydra_dir',
    'wandb_run_path',
    'wandb_checkpoint',
    'parent_model_tag',
    'data_root_dir',
    'train_tags',
    'val_tags'
]


def _get_lock_file_path(path_to_file):
    return path_to_file.parent / f'.{path_to_file.name}.lock'


def _get_file_lock(path_to_file):
    lock_file_path = _get_lock_file_path(path_to_file)
    lock = FileLock(lock_file_path)
    return lock


def load_after_maybe_create_unsafe(path_to_csv, headers, converters=None):
    if not path_to_csv.exists():
        df = pd.DataFrame({
            key: []
            for key in headers
        })
        df.to_csv(path_to_csv)

    records_df = pd.read_csv(path_to_csv, converters=converters)
    return records_df


def load_after_maybe_create_records(path_to_csv, headers, timeout=10, converters=None):
    lock = _get_file_lock(path_to_csv)

    try:
        with lock.acquire(timeout=timeout):
            return load_after_maybe_create_unsafe(path_to_csv, headers, converters=converters)
    except Timeout:
        raise ValueError(f'File lock timed out after {timeout}s for {path_to_csv}')


def save_records_unsafe(path_to_csv, records_df):
    records_df.to_csv(path_to_csv, index=False)


def save_records(path_to_csv, records_df, timeout=10):
    lock = _get_file_lock(path_to_csv)

    try:    
        with lock.acquire(timeout=timeout):
            save_records_unsafe(path_to_csv, records_df)
    except Timeout:
        raise ValueError(f'File lock timed out after {timeout}s for {path_to_csv}')


def get_data_folders_from_skill_data_records(path_to_skill_data_records_csv, cfg, assume_single_root_dir, 
                                             logger=None):
    # Make a copy of config so that we don't overwrite the passed one
    cfg = copy.deepcopy(cfg)
    if logger is None:
        logger = logging.getLogger(__name__)

    skill_data_records_df = load_after_maybe_create_records(
            path_to_skill_data_records_csv, skill_data_csv_headers, converters=skill_data_csv_converters
        )
    assert pd.api.types.is_bool_dtype(skill_data_records_df['has_been_processed']), "has_been_processed is not a bool type"

    rows_of_usable_skill_data = skill_data_records_df.loc[
            (skill_data_records_df['env_type'] == cfg['data']['env'])
            & (skill_data_records_df['skill_type'] == cfg['data']['skill'])
            # & skill_data_records_df['has_been_processed']
        ]

    if len(rows_of_usable_skill_data) == 0:
        logger.info(f"Did not find any usable skill data for skill {cfg['data']['skill']} - Exiting")
        return cfg
    
    if not assume_single_root_dir:
        data_idx = 0
        data_root_tags_dict = dict()
        for _, row in rows_of_usable_skill_data.iterrows():
            if row['data_dir_processed'] == None or len(row['data_dir_processed']) == 0:
                assert not row['has_been_processed'], "Data has been processed but no value for key `data_dir_processed`"
                logger.info(f"Processed data dir not found. Will use: {row['data_dir']}")
                data_path = Path(row['data_dir'])
            else:
                assert row['has_been_processed'], "Data has been processed but no value for key `data_dir_processed`"
                data_path = Path(row['data_dir_processed'])
            data_root_dir = data_path.parent.parent
            data_tag = data_path.parts[-2]
            data_root_tags_dict[f'data_{data_idx}'] = {'root': str(data_root_dir), 'tags': [data_tag]}
            data_idx += 1
        all_data_keys = list(data_root_tags_dict.keys())
        np.random.shuffle(all_data_keys)
        val_keys_count = max(int(len(all_data_keys) * 0.2), 1) 
        val_data_keys = all_data_keys[:val_keys_count]
        val_data_root_tags_dict = dict()
        for val_key in val_data_keys:
            val_data_root_tags_dict[val_key] = data_root_tags_dict[val_key]
            del data_root_tags_dict[val_key]
        cfg['data']['root'] = OmegaConf.create(data_root_tags_dict)
        cfg['data']['val_root'] = OmegaConf.create(val_data_root_tags_dict)
    else:
        column_key = 'data_dir' if rows_of_usable_skill_data['data_dir_processed'] == None or \
            len(rows_of_usable_skill_data['data_dir_processed']) == 0 else 'data_dir_processed'
        data_root_dir = Path(rows_of_usable_skill_data[column_key].iloc[0]).parent.parent
        tags = [
            Path(row[column_key]).parts[-2]
            for _, row in rows_of_usable_skill_data.iterrows()
        ]
        np.random.shuffle(tags)
        n_val_tags = int(np.clip(cfg['data']['test_size'] * len(tags), 1, len(tags) - 1))
        tags_tr, tags_val = tags[n_val_tags:], tags[:n_val_tags]
        logger.info(f'Found {len(tags)} tags in {data_root_dir}')
        logger.info(f'Using {len(tags) - n_val_tags} tags for training, {n_val_tags} tags for validation')
        cfg['data']['root'] = str(data_root_dir)
        cfg['data']['tags'] = tags_tr
        cfg['data']['val_root'] = str(data_root_dir)
        cfg['data']['val_tags'] = tags_val

    # Return updated config with data folders
    return cfg


class RecordsContext():

    def __init__(self, path_to_csv, headers, timeout=10):
        self._path_to_csv = path_to_csv
        self._headers = headers
        self._lock = _get_file_lock(path_to_csv)
        self._timeout = timeout
    
    def __enter__(self):
        self._lock.acquire(timeout=self._timeout)
        return load_after_maybe_create_unsafe(self._path_to_csv, self._headers)
    
    def __exit__(self, type, value, traceback):
        self._lock.release()


def get_rows_of_matching_models_from_records(model_records_df, model_type, skill_type, env_type):
    return model_records_df.loc[
        (model_records_df['model_type'] == model_type) &
        (model_records_df['skill_type'] == skill_type) &
        (model_records_df['env_type'] == env_type)
    ]
