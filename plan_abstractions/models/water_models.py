import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from pickle import dump


class RigidModel():
    def __init__(self, dim_state = None, model_cfg=None):
        self._dim_state = dim_state

    @property
    def dim_state(self):
        return self._dim_state

    @property
    def fixed_input_size(self):
        return True

    def predict(self, state, action, unnormalized=False):
        new_state = state.copy()
        new_state[0:2] = action.copy()
        assert len(new_state) == 13
        costs = [np.linalg.norm(state[:2]-new_state[:2])]
        if return_np:
            return new_state
        return {
            'end_states': [new_state],
            'T_exec': [1],
            'costs': costs,
            'info_plan': {
                'T_plan': [1]
            }
        }



class PourConstantModel:
    def __init__(self, dim_state = None, model_cfg=None):
        self._dim_state = dim_state

    @property
    def dim_state(self):
        return self._dim_state

    @property
    def fixed_input_size(self):
        return True

    def predict(self, state, action, unnormalized=False):
        new_state = state.copy()
        new_state[-1] = 0.9
        new_state[-2] = 0.1
        assert len(new_state) == 13
        costs = [0.01] #some small cost for turning: not big since you cant really compare angles
        return {
            'end_states': [new_state],
            'T_exec': [1],
            'costs': costs,
            'info_plan': {
                'T_plan': [1]
            }
        }


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
        self._input_scaler = StandardScaler().fit(data)
        self._output_scaler = StandardScaler().fit(next_states)
        train_data = self._input_scaler.transform(data)
        train_labels = self._output_scaler.transform(next_states)
        reg = LinearRegression().fit(train_data, train_labels)
        self._model = reg
        np.save(self._save_file, reg)

    def predict(self, state, action, unnormalized=False):
        cond = np.hstack([state, action])
        if len(cond.shape) == 1:
            cond = cond.reshape(1,-1)
        new_state= self._model.predict(self._input_scaler.transform(cond))[0]
        return self._output_scaler.inverse_transform(new_state)

class RFRModel:
    def __init__(self, model_cfg=None, dim_state = None):
        self._dim_state = dim_state
        self._save_file = model_cfg["save_file"]
        if model_cfg["load"]:
            self._model = np.load(self._save_file, allow_pickle=True)
            input_scaler_fn, output_scaler_fn = self._get_scaler_filenames()
            self._input_scaler = np.load(input_scaler_fn, allow_pickle=True)
            self._output_scaler = np.load(output_scaler_fn, allow_pickle=True)
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
        self._input_scaler = StandardScaler().fit(data)
        self._output_scaler = StandardScaler().fit(next_states)
        train_data = self._input_scaler.transform(data)
        train_labels = self._output_scaler.transform(next_states)
        reg = RandomForestRegressor(max_depth=20, random_state=0)
        reg.fit(train_data, train_labels)
        self._model = reg
        np.save(self._save_file, reg)
        with open(self._save_file, 'wb') as f:
            dump(self._model, f)
        input_scaler_fn, output_scaler_fn = self._get_scaler_filenames()
        with open(input_scaler_fn, 'wb') as f:
            dump(self._input_scaler, f)
        with open(output_scaler_fn, 'wb') as f:
            dump(self._output_scaler, f)

    def _get_scaler_filenames(self):
        input_scaler_fn = f"{self._save_file}_input_scaler.pkl".replace(".npy", "")
        output_scaler_fn = f"{self._save_file}_output_scaler.pkl".replace(".npy", "")
        return input_scaler_fn, output_scaler_fn

    def predict(self, state, action, unnormalized=False):
        cond = np.hstack([state, action])
        if len(cond.shape) == 1:
            cond = cond.reshape(1,-1)
        new_state_normalized = self._model.predict(self._input_scaler.transform(cond))[0]
        new_state = self._output_scaler.inverse_transform(new_state_normalized)
        costs = [0.01]
        return {
            'end_states': [new_state],
            'T_exec': [1],
            'costs': costs,
            'info_plan': {
                'T_plan': [1]
            }
        }

class ResidualRFRModel:
    def __init__(self, model_cfg=None, dim_state = None):
        self._dim_state = dim_state
        self._save_file = model_cfg["save_file"]
        if model_cfg["load"]:
            self._model = np.load(self._save_file, allow_pickle=True)
            input_scaler_fn, output_scaler_fn = self._get_scaler_filenames()
            self._input_scaler = np.load(input_scaler_fn, allow_pickle=True)
            self._output_scaler = np.load(output_scaler_fn, allow_pickle=True)
            self._base_model = eval(model_cfg["base_model"])
        else:
            self._model = None

    @property
    def dim_state(self):
        return self._dim_state

    @property
    def fixed_input_size(self):
        return True

    def train(self, states, actions, next_statess):
        data = np.hstack([states, actions])
        self._input_scaler = StandardScaler().fit(data)
        model_predicted_next_states = np.vstack(self._base_model.predict(state, actions, return_np)for state, action in zip(states, actions)])
        residual = next_states - model_predicted_next_states
        print("residual", residual.shape)
        self._output_scaler = StandardScaler().fit(residual)
        train_data = self._input_scaler.transform(data)
        train_labels = self._output_scaler.transform(residual)
        reg = RandomForestRegressor(max_depth=20, random_state=0)
        reg.fit(train_data, train_labels)
        self._model = reg
        np.save(self._save_file, reg)
        with open(self._save_file, 'wb') as f:
            dump(self._model, f)
        input_scaler_fn, output_scaler_fn = self._get_scaler_filenames()
        with open(input_scaler_fn, 'wb') as f:
            dump(self._input_scaler, f)
        with open(output_scaler_fn, 'wb') as f:
            dump(self._output_scaler, f)

    def _get_scaler_filenames(self):
        input_scaler_fn = f"{self._save_file}_input_scaler.pkl".replace(".npy", "")
        output_scaler_fn = f"{self._save_file}_output_scaler.pkl".replace(".npy", "")
        return input_scaler_fn, output_scaler_fn

    def predict(self, state, action, unnormalized=False, return_np=False):
        cond = np.hstack([state, action])
        if len(cond.shape) == 1:
            cond = cond.reshape(1,-1)
        new_state_normalized = self._model.predict(self._input_scaler.transform(cond))[0]
        new_state = self._output_scaler.inverse_transform(new_state_normalized)
        costs = [0.01]
        if return_np:
            return new_state
        return {
            'end_states': [new_state],
            'T_exec': [1],
            'costs': costs,
            'info_plan': {
                'T_plan': [1]
            }
        }
