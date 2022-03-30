import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
exp_name = "transition_data_2.npy"
data_root = "/home/lagrassa/plan_abstractions/two_d_water_transport1"
data_dir = os.path.join(data_root, exp_name)
data =  np.load(os.path.join(data_root, exp_name), allow_pickle=True).item()
init_state_data = data["init_states"]
end_state_data = data["end_states"]
param_data = data["params"]
#end_state_data = np.load(os.path.join(data_dir, "end_states.npy"))
#param_data = np.load(os.path.join(data_dir, "param_data.npy"))

def rigid_model(state, action):
    """
    Assumes water moves with the cup and never leaves
    If it was out before it stays out
    """
    new_state = state.copy()
    new_x = action.item()
    new_state[0] = new_x
    return new_state

def linear_water_out(state, action):
    """
    Water moves out of the cup linearly 
    """
    new_state = state.copy()
    dx = action.item()
    new_x = state[0] + dx
    new_state[0] = new_x
    return new_state

des_x = param_data[:,0]
#reg = LinearRegression().fit(des_x.reshape(-1, 1), end_state_data[:,-1].reshape(-1, 1))
plt.scatter(param_data[:,0]-init_state_data[:,0], end_state_data[:,-1]-init_state_data[:,-1], label="out_water and delta x")
#plt.scatter(param_data[:,0], end_state_data[:,0], label="glass x")
#rigid_preds = np.array([rigid_model(init_state_data[i,:], param_data[i,0]) for i in range(len(param_data))])
#plt.scatter(param_data[:,0],rigid_preds[:,0], label="rigid model")
#plt.scatter(param_data[:,0],rigid_preds[:,-1], label="rigid model")
#plt.scatter(param_data[:,0], reg.predict(param_data[:,0].reshape(-1,1)), label="model")
plt.legend()
plt.show()
import ipdb; ipdb.set_trace()
