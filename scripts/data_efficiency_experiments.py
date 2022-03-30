from autolab_core import YamlConfig
from train_mde import train_mde
from subprocess import call

def main():
    cfg = YamlConfig("cfg/train/mde_train/learned_for_pour.yaml")
    cfg["upload_to_wandb"] = True
    for num_samples in range(3, 6):
        model_runpath = train_mde(cfg=cfg, max_num_data=num_samples)
        data_root_dir = f"/tmp/data_efficiency_{num_samples}"
        skill_name = "Pour"
        model_name = "SEMModel1"
        n_init_states = 2
        dev_cfg = f"skills.{skill_name}.high_level_models.{model_name}.deviation_cfg"
        retcode = call(f"python scripts/run_planner_vector_state.py" + 
                " --config-name=solve_water_in_box "+
                f" n_init_states={n_init_states} "+
                f" data_root_dir={data_root_dir} "
                f" {dev_cfg}.use_deviation_model=1 "+
                f" {dev_cfg}.run_path={model_runpath}", shell=True)
        


if __name__ == "__main__":
    main()
