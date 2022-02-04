import numpy as np
import os
from sklearn.linear_model import LinearRegression
exp_name = "morevarx"
data_root = "data/"
data_dir = os.path.join(data_root, exp_name)
init_state_data = np.load(os.path.join(data_dir, "init_states.npy")
end_state_data = np.load(os.path.join(data_dir, "end_states.npy")
param_data = np.load(os.path.join(data_dir, "param_data.npy")
des_x = param_data[:,0]
reg = LinearRegression().fit(des_x, end_state_data[:,-1])
plt.scatter(param_data[:,0], end_state_data[:,-1])
plt.plot(param_data[:,0], reg.predict(param_data[:,0])

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

