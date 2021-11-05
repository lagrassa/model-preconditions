# Model preconditions

This is the code used to train and plan using model preconditions. The corl2021 branch contains the code used for "Learning Model Preconditions for Planning with Multiple Models"

## Installation

Run `pip install -e .`

Additional dependencies needed by cloning from github:

* [isaacgym_utils](https://github.com/iamlab-cmu/isaacgym-utils)

* [pillar-state](https://github.com/iamlab-cmu/pillar-state)

And please install [IsaacGym](https://developer.nvidia.com/isaac-gym)


### Running the planner
Run the planner by running `python scripts/run_planner.py --config-name=$PLAN_CONFIG_NAME` which should be in the `cfg/planner`  directory

### Data collection
Making the data collection code clean enough (even for research code) to be public is a WIP but I can share the file with you if you would like. 

### Training MDEs
You can train MDEs using our data by running `python scripts/train_mde.py --config-name=$MDE_TRAIN_CONFIG_NAME` which will be in the `cfg/train/mde_train/` directory.
This hasn't been cleaned up (biggest factor is you will need to change paths). If you would like to run it and are unable to, please file a GitHub issue and I (Alex) will help you out.

## Replication
We include all the code necessary to replicate our results, including make plots. Some manual script running is required. 
Our MDEs were trained on a fairly narrow dataset so if the performance is off, make sure the location of the drawer and box are in the same range as in the training data.
We recommend training them from scratch on data collected in your environment, especially if that data is collected offline.

## Planner
If you would like to view the planner code, it is in
`plan_abstractions/planning/astar_mr.py`


### MDE accuracy
Train the relevant model. The folder listed in the config as
`mde_stats_path` in the config will contain the mean, loss, and some other useful metrics such as the fraction of overestimates/underestimates on the set for training, validation and testing. Validation is a random fraction of the training data and is intended to be used for hyperparameter tuning. Test data should come from a different experiment run. To generate the values in the table run

`python scripts/plotting_scripts/print_table_data.py --data_path=$DATA_FOLDER_PATH`

With $DATA_FOLDER_PATH set to the directory set in `mde_stats_path`


### Distribution of models chosen
The data where this is saved is `data_root_dir` in the planning config which should be stored in `cfg/planner`. To process and plot this data, run 
`python scripts/plotting_scripts/plot_planning_stats --cfg=cfg/plot_planner_data`

To change which task is plotted, change the `task_for_bar_graph` variable. 

### Planning time and model evaluation rate
Using the same script that consolidates the model distribution data, planning time and model evaluation rate information will be printed. 


### Planning success
Open `data/Experiments\ for \revision.xlsx` and edit with the results you get by executing experiments on a real robot. 

`python scripts/plotting_scripts/plot_plan_performance`
