"""
Simple replacement for collect_skill_data script for softgym case. Does not use pillar_state
"""
import numpy as np
import os
from autolab_core import YamlConfig

from plan_abstractions.skills.water_skills import WaterTransport1D
from plan_abstractions.envs.water_env import WaterEnv

n_init_states =  3
n_parameters = 20
num_total_datapoint = n_init_states * n_parameters
param_dim = 1
vector_dim = 7
T = 10
data_root = "data/"
exp_name = "morexvar"
env_cfg = YamlConfig('cfg/envs/water_env.yaml')
pointcloud_dim = 0
state_dim = vector_dim + pointcloud_dim #save keypoint dim and water point dim
init_state_data = np.zeros((num_total_datapoint, vector_dim+pointcloud_dim ))
end_state_data = np.zeros_like(init_state_data)
env = eval(env_cfg["env"])(env_cfg)
state = env.get_sem_state()
ctr = 0
skill = WaterTransport1D(param_dist_cfg=env_cfg["skill_cfg"]['param_sampling_probabilities'])
param_gen = skill.generate_parameters(env, state, num_parameters=1)
param_data = np.zeros((num_total_datapoint, param_dim))
#cup_state = np.array([self.glass_x, self.glass_dis_x, self.glass_dis_z, self.height,self._get_current_water_height(), in_glass, out_glass])
for i in range(n_init_states):
    for j in range(n_parameters):
        init_state_data[ctr] = env.get_sem_state()
        params = next(param_gen)[0]
        param_data[ctr, :] = params
        controllers, _ = skill.make_controllers([state], params, total_horizon=T)
        controller = controllers[0] #there's only 1
        curr_state = state.copy()
        for t in range(T):
            curr_state = env.get_sem_state()
            action = controller(curr_state, t)
            skill.apply_action(env, 0, action)
            env.step()
        end_state_data[ctr] = env.get_sem_state()
        print(env.get_sem_state()[-1])
        env.reset()
        ctr += 1

data_dir = os.path.join(data_root, exp_name)
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
np.save(os.path.join(data_dir, "init_states.npy"), init_state_data)
np.save(os.path.join(data_dir, "end_states.npy"), end_state_data)
np.save(os.path.join(data_dir, "param_data.npy"), param_data)
