# given some planner data train a model that predicts the deviation from the model, upload this model to wandb - RFR?
__spec__ = None  # to allow for ipdb during wandb

from sklearn.model_selection import train_test_split
from isaacgym import gymapi
from plan_abstractions.learning.data_utils import make_deviation_datalists, eval_model, train_and_fit_model, \
    remove_outliers
from plan_abstractions.models.mdes import *
#from plan_abstractions.models.gnn_dev_model import ECNModel

import logging
import os
from pathlib import Path

import numpy as np
import hydra
import wandb

from plan_abstractions.utils import get_formatted_time
from isaacgym_utils.math_utils import set_seed

logger = logging.getLogger(__name__)
logger.setLevel('INFO')


@hydra.main(config_path='../cfg/train/mde_train', config_name='train_sweep_model_validation_easy_only.yaml')
def main(cfg):
    log = logging.getLogger(__name__)
    cfg['original_cwd'] = hydra.utils.get_original_cwd()
    set_seed(cfg['seed'])
    # iterate through all train data, collect SEM preds, then train validation model
    if 'save_path_prefix' in cfg and len(cfg['save_path_prefix']) > 0:
        hydra_dir = Path(f"{os.getcwd()}/{cfg['save_path_prefix']}")
    else:
        hydra_dir = Path(os.getcwd())

    # Make dataset
    if "dataset_file_cache" in cfg.keys():
        dataset_data = np.load(cfg["dataset_file_cache"], allow_pickle=True).item()
    else:
        dataset_data = make_deviation_datalists(cfg, feature_type=cfg.get('feature_type', 'dists_and_actions_only'), shuffle=cfg.get('train', True),
                                                graphs=cfg.get("graphs", False))  # shuffle=cfg.get('train', True))
        if "dataset_save_loc" in cfg.keys():
            np.save(cfg["dataset_save_loc"], dataset_data)
    dataset_data = remove_outliers(dataset_data)
    states_and_parameters, deviations = dataset_data["training"]
    logger.info(f"training dataset of {len(states_and_parameters)} ")
    test_states_and_parameters, test_deviations = dataset_data["test"]
    train_states_and_params, validation_states_and_params, train_deviations, validation_deviations = train_test_split(
        states_and_parameters, deviations, shuffle=True, test_size=cfg['test_size'])
    if not cfg["upload_to_wandb"]:
        os.environ['WANDB_MODE'] = 'dryrun'
    experiment = wandb.init(**cfg['wandb']['init'], name=cfg['tag'], config=cfg['shared_info'])
    if cfg.get('train', True):
        model_cls = cfg.get('model_class', 'MLPModel')
        if model_cls == "MLPModel":
            deviation_model = MLPModel(cfg['shared_info'], train_states_and_params.shape[1])
        else:
            deviation_model = eval(model_cls)(cfg['shared_info'])
        train_cfg = cfg['shared_info'].get('train_cfg', None)
        train_and_fit_model(train_cfg, deviation_model, train_states_and_params, train_deviations, test_states_and_parameters,
                            test_deviations, validation_states_and_params, validation_deviations, experiment,
                            validate_on_split_data=False)
    else:
        deviation_model = create_deviation_wrapper_from_cfg(cfg["deviation_cfg"], graphs=cfg.get('graphs', False))
    result = eval_model(deviation_model, train_states_and_params, train_deviations, test_states_and_parameters,
                        test_deviations, validation_states_and_params, validation_deviations,
                        do_pr_curve=cfg.get("do_pr_curve", False), do_traj_curve=False)
    if isinstance(result, dict):
        mde_stats_path = cfg.get("mde_stats_path", "/tmp")
        if not os.path.isdir(mde_stats_path):
            os.mkdir(mde_stats_path)
        result["model"] = "Simulator" if cfg['shared_info']['use_sim_model'] else cfg["shared_info"]["sem_cfg"]["type"]
        result["skills"] = list(cfg["skills"].keys())
        all_deviations = np.vstack([deviations, test_deviations])
        result["deviation_mean"] = np.mean(all_deviations)
        result["deviation_std"] = np.std(all_deviations)
        formatted_datetime = get_formatted_time()
        filename = f"{formatted_datetime}_results.npy"
        path_and_filename = os.path.join(mde_stats_path, filename)
        np.save(path_and_filename, result)

    else: #assume is a PR curve
        precision, recall, thresholds = result
        file_name_start = cfg["precision_save_dir"]
        exp_name = deviation_model.__class__.__name__
        np.save(f"{file_name_start}/{exp_name}_precision.npy", precision)
        np.save(f"{file_name_start}/{exp_name}_recall.npy", recall)
        np.save(f"{file_name_start}/{exp_name}_thresholds.npy", thresholds)

    hydra_dir = Path(os.getcwd())
    save_filename = hydra_dir / 'validation_model.pkl'
    deviation_scaler_filename = hydra_dir / 'deviation_scaler.pkl'
    state_and_parameter_scaler_filename = hydra_dir / 'state_and_parameter_scaler.pkl'
    state_and_parameter_scaler_node_filename = hydra_dir / 'state_and_parameter_scaler_node.pkl'
    state_and_parameter_scaler_edge_filename = hydra_dir / 'state_and_parameter_scaler_edge.pkl'
    if cfg["train"]:
        deviation_model.save_model(save_filename, deviation_scaler_filename, state_and_parameter_scaler_filename)
    filenames = [save_filename, deviation_scaler_filename, state_and_parameter_scaler_filename, state_and_parameter_scaler_node_filename, state_and_parameter_scaler_edge_filename]
    if cfg["upload_to_wandb"] and cfg["train"]:
        if cfg['wandb']['saver']['upload']:
            for filename in filenames:
                wandb.save(str(filename), base_path=str(hydra_dir))
            wandb.save(str(hydra_dir / '.hydra' / '*.yaml'), base_path=str(hydra_dir))


if __name__ == '__main__':
    main()
