"""
Simple replacement for collect_skill_data script for softgym case. Does not use pillar_state
"""
import numpy as np
from autolab_core import YamlConfig
n_init_states =  3
n_parameters = 2
num_total_datapoint = n_init_states * n_parameters
param_dim = 1
vector_dim = 7
pointcloud_dim = 150
state_dim = vector_dim + pointcloud_dim #save keypoint dim and water point dim
init_state_data = np.zeros((num_total_datapoint, vector_dim+pointcloud_dim ))
end_state_data = np.zeros_like(init_state_data)
env_cfg = YamlConfig('cfg/envs/water_env.yaml')
env = eval(env_cfg.env)(env_cfg)
ctr = 0
for i in range(n_init_states):
    for j in (n_parameters):
        param = env.action_space.sample()
        controller = WaterTransportController(param)
        init_state_data[ctr] = env.get_vector_state()
        env.apply_actions(controller)
        end_state_data[ctr] = env.get_vector_state()

        ctr += 1
