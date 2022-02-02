"""
Simple replacement for collect_skill_data script for softgym case. Does not use pillar_state
"""
import numpy as np
from autolab_core import YamlConfig

from plan_abstractions.skills.water_skills import WaterTransport1D

n_init_states =  3
n_parameters = 2
num_total_datapoint = n_init_states * n_parameters
param_dim = 1
vector_dim = 7
env_cfg = YamlConfig('cfg/envs/water_env.yaml')
pointcloud_dim = env_cfg['env_props']['pointcloud_dim']
state_dim = vector_dim + pointcloud_dim #save keypoint dim and water point dim
init_state_data = np.zeros((num_total_datapoint, vector_dim+pointcloud_dim ))
end_state_data = np.zeros_like(init_state_data)
env = eval(env_cfg.env)(env_cfg)
state = env.get_vector_state()
ctr = 0
skill = WaterTransport1D({})
param_gen = skill.generate_parameters(env, state)
for i in range(n_init_states):
    for j in (n_parameters):
        init_state_data[ctr] = env.get_vector_state()
        params = next(param_gen)
        controller = skill.make_controllers([state], [params])
        env.apply_actions(controller)
        env.step()
        end_state_data[ctr] = env.get_vector_state()
        ctr += 1
