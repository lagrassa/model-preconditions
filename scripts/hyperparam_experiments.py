from autolab_core import YamlConfig
import os
from train_mde import train_mde
from subprocess import call
import numpy as np
data_root_dir = f"/home/lagrassa/plan_abstractions/hyperparam"
skill_name = "WaterTransport2D"
model_name = "SEMModel1"
dev_cfg = f"skills.{skill_name}.high_level_models.{model_name}.deviation_cfg"

def generate_data():
    #cfg["upload_to_wandb"] = True
    num_pts_dmax = 10
    num_pts_beta = 10
    for acceptable_deviation in np.linspace(0.01,1, num_pts_dmax):
        for beta in np.linspace(0.1,3, num_pts_beta):
            n_init_states = 30
            retcode = call(f"python scripts/run_planner_vector_state.py" + 
                    " --config-name=solve_water_transport "+
                    f" n_init_states={n_init_states} "+
                    f" data_root_dir={data_root_dir} "
                    f" {dev_cfg}.acceptable_deviation={acceptable_deviation} "+
                    f" {dev_cfg}.beta={beta} ",shell=True)

def plan_results_to_plan_found_plan_success(plan_results):
    num_plans_found = 0
    num_goals_reached = 0
    num_exps = 0
    for plan_result in plan_results:
        num_plans_found += int(plan_result["plan_found"])
        num_goals_reached += int(plan_result["plan_exec_reached_goal"])
        num_exps += 1
    return num_plans_found/num_exps, num_goals_reached/num_plans_found
        
def generate_plots():
    data_dir= os.path.join(data_root_dir, "plan_results")
    betas = []
    dmaxes = []
    plan_founds = []
    plan_successes = []
    for fn in os.listdir(os.path.join(data_root_dir, "plan_results")):
        if "yaml" in fn:
            continue
        config_fn = f"{fn}_config.yaml"
        plan_results = np.load(os.path.join(data_dir,fn), allow_pickle=True)
        config_data = np.load(os.path.join(data_dir,config_fn), allow_pickle=True)
        deviation_cfg = config_data['skills'][skill_name]['high_level_models'][model_name]['deviation_cfg']
        beta = deviation_cfg['beta']
        dmax = deviation_cfg['acceptable_deviation']
        betas.append(beta)
        dmaxes.append(dmax)
        plan_found, plan_success = plan_success, exec_success = plan_results_to_plan_found_plan_success(plan_results)
        plan_founds.append(plan_found)
        plan_successes.append(plan_success)
    plt.scatter(betas, plan_founds)




    


if __name__ == "__main__":
    generate_plots()
    #main()
