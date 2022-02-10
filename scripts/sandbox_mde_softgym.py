import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
exp_name = "morexvar"
data_root = "data/"
data_dir = os.path.join(data_root, exp_name)
init_state_data = np.load(os.path.join(data_dir, "init_states.npy"))
end_state_data = np.load(os.path.join(data_dir, "end_states.npy"))
param_data = np.load(os.path.join(data_dir, "param_data.npy"))
def rigid_model(state, action):
    """
    Assumes water moves with the cup and never leaves
    If it was out before it stays out
    """
    new_state = state.copy()
    dx = action.item()
    new_x = state[0] + dx
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
reg = LinearRegression().fit(des_x.reshape(-1, 1), end_state_data[:,-1].reshape(-1, 1))
#plt.scatter(param_data[:,0], end_state_data[:,-1], label="out_water")
plt.scatter(param_data[:,0], end_state_data[:,0], label="glass x")
rigid_preds = np.array([rigid_model(init_state_data[i,:], param_data[i,0]) for i in range(len(param_data))])
import ipdb; ipdb.set_trace()
plt.scatter(param_data[:,0],rigid_preds[:,0], label="rigid model")
#plt.scatter(param_data[:,0], reg.predict(param_data[:,0].reshape(-1,1)), label="model")
plt.legend()
plt.show()
