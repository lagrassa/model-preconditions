import numpy as np
from pickle import load
from autolab_core import YamlConfig
import os
import matplotlib.pyplot as plt

def plan_results_to_plan_found_plan_success(plan_results):
    num_plans_found = 0
    num_goals_reached = 0
    num_exps = 0
    for plan_result in plan_results:
        num_plans_found += int(plan_result["plan_found"])
        num_goals_reached += int(plan_result["plan_exec_reached_goal"])
        num_exps += 1
    return num_plans_found/num_exps, num_goals_reached/num_plans_found


data_root_dir = "/tmp/hyperparam/default/plan_results"
data_root_dir= "/tmp/hyperparam/plan_results"
data_to_hyperparams = []
for fn in os.listdir(data_root_dir):
    if 'yaml' in fn:
        continue
    full_fn = os.path.join(data_root_dir, fn)
    config_fn = f"{full_fn}_config.yaml"
    plan_results = np.load(full_fn, allow_pickle=1)
    plan_success, exec_success = plan_results_to_plan_found_plan_success(plan_results)
    import ipdb; ipdb.set_trace()
    cfg = np.load(config_fn, allow_pickle=1)
    dev_cfg = cfg["skills"]["WaterTransport2D"]["high_level_models"]["SEMModel1"]["deviation_cfg"]
    hyperparams = dev_cfg["beta"], dev_cfg["acceptable_deviation"]
    data_to_hyperparams.append(hyperparams)
    print(data_to_hyperparams)

