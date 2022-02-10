import numpy as np
from sklearn.linear_model import LinearRegression



class RigidModel:
    def __init__(self, dim_state = None):
        self._dim_state = dim_state

    @property
    def dim_state(self):
        return self._dim_state

    @property
    def fixed_input_size(self):
        return True

    def predict(self, state, action, unnormalized=False):
        new_state = state.copy()
        dx = action.item()
        new_state[0] += dx
        return new_state


class LinearOutModel:
    def __init__(self, model_cfg=None, dim_state = None):
        self._dim_state = dim_state
        self._save_file = model_cfg["save_file"]
        if model_cfg["load"]:
            self._model = np.load(self._save_file, allow_pickle=True).item()
        else:
            self._model = None

    @property
    def dim_state(self):
        return self._dim_state

    @property
    def fixed_input_size(self):
        return True

    def train(self, state, actions, next_states):
        data = np.hstack([state, actions])
        print("data shape", data.shape)
        reg = LinearRegression().fit(data, next_states)
        self._model = reg
        np.save(self._save_file, reg)

    def predict(self, state, action, unnormalized=False):
        cond = np.hstack([state, action])
        if len(cond.shape) == 1:
            cond = cond.reshape(1,-1)
        new_state= self._model.predict(cond)[0]
        return new_state
