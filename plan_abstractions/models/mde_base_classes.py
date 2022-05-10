from pickle import dump

import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from plan_abstractions.utils import dists_and_actions_from_states_and_parameters, augment_with_dists, extract_first_and_last, identity


class DeviationModel():
    def __init__(self, cfg):
        from ..envs import FrankaRodEnv, FrankaDrawerEnv, WaterEnv2D, WaterEnv3D
        self.with_conf = False
        if 'env' in cfg.keys():
            self._env_cls = eval(cfg["env"])
        if 'data' in cfg.keys():
            self._sem_state_obj_names = cfg["data"]["sem_state_obj_names"]  # TODO get from wandb file
        if 'sem_state_obj_names' in cfg.keys():
            self._sem_state_obj_names = cfg["sem_state_obj_names"]
        if 'state_and_param_to_features' in cfg.keys():
            state_and_param_to_features = eval(cfg['state_and_param_to_features'])
        if "data_dims" in cfg.keys():
            self._data_dims = np.array(cfg["data_dims"])
        else:
            state_and_param_to_features = dists_and_actions_from_states_and_parameters
        self._is_graph_model = False
        self._train_cfg = cfg.get("train_cfg", None)
        self._c1 = cfg.get('c1',3)
        self._c2 = cfg.get('c2', 1)
        self._deviation_scaler = None
        self._state_and_parameter_scaler = None
        self._state_and_param_to_features = state_and_param_to_features 

    def evaluate_loss(self, y, y_hat):
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if len(y_hat.shape) == 1:
            y_hat = y_hat.reshape(-1, 1)
        zero_vec = np.zeros_like(y)
        return self._c1 * np.max(np.hstack([y - y_hat, zero_vec]), axis=1) ** 2 + self._c2 * np.max(
            np.hstack([y_hat - y, zero_vec]), axis=1) ** 2

    def fit_scaler(self, states_and_parameters, deviations, rescale=False):
        if rescale:
            self._state_and_parameter_scaler = StandardScaler().fit(states_and_parameters)
            self._deviation_scaler = StandardScaler().fit(deviations)

    def train(self, cfg, states_and_parameters, deviations, states_and_parameters_val, deviations_val, rescale=1,
              wandb_experiment=None):
        self._train_cfg = cfg
        self.fit_scaler(states_and_parameters, deviations, rescale=rescale)
        states_and_parameters_scaled = self._state_and_parameter_scaler.transform(states_and_parameters)
        deviations_scaled = self._deviation_scaler.transform(deviations)
        deviations_val_scaled = self._deviation_scaler.transform(deviations_val)

        states_and_parameters_val_scaled = self._state_and_parameter_scaler.transform(states_and_parameters_val)
        self._train(cfg, states_and_parameters_scaled, deviations_scaled, states_and_parameters_val_scaled,
                    deviations_val_scaled, wandb_experiment=wandb_experiment)

    def predict(self, input_state_and_parameters, already_transformed_state_vector=False,
                state_ndim=None ):
        if not already_transformed_state_vector:
            assert state_ndim is not None
        if not already_transformed_state_vector:
            if self._data_dims is not None:
                input_state_and_parameters = np.hstack([input_state_and_parameters[:,:state_ndim][:,self._data_dims], input_state_and_parameters[:,state_ndim:]])
                state_ndim = len(self._data_dims)
            state_and_parameters = self._state_and_param_to_features(input_state_and_parameters,state_ndims=state_ndim)
        else:
            state_and_parameters = input_state_and_parameters

        if self._state_and_parameter_scaler is not None:
            state_and_parameters = self._state_and_parameter_scaler.transform(state_and_parameters)

        if not self.with_conf:
            deviation_predicted = self._predict(state_and_parameters)
            stdev_pred = 0
        else:
            deviation_predicted, stdev_pred = self._predict(state_and_parameters)

        if self._deviation_scaler is not None:
            result = self._deviation_scaler.inverse_transform(deviation_predicted)
            stdev_pred = ((stdev_pred**2)*self._deviation_scaler.var_)**0.5

        else:
            assert False
            result = deviation_predicted
        return result, stdev_pred

    def predict_from_pillar_state(self, state, parameters):
        assert self._sem_state_obj_names is not None
        sem_state = self._env_cls.pillar_state_to_sem_state(state, self._sem_state_obj_names)
        return self.predict_from_np(state, parameters)

    def predict_from_np(self, state, parameters):
        state_and_parameters = np.hstack([state, parameters]).reshape(1, -1)
        if not self.with_conf:
            return self.predict(state_and_parameters, state_ndim=state.shape[0]).item()
        return self.predict(state_and_parameters, state_ndim=state.shape[0])

    def save_model(self, model_fn, deviation_scaler_fn, state_and_parameter_scaler_fn):
        self._save_model(model_fn)

        with open(deviation_scaler_fn, 'wb') as f:
            dump(self._deviation_scaler, f)

        with open(state_and_parameter_scaler_fn, 'wb') as f:
            dump(self._state_and_parameter_scaler, f)

    def load_model(self, model_fn, deviation_scaler_fn, state_and_parameter_scaler_fn):
        self._deviation_scaler = np.load(deviation_scaler_fn, allow_pickle=True)
        self._state_and_parameter_scaler = np.load(state_and_parameter_scaler_fn, allow_pickle=True)
        self._load_model(model_fn)


class SKLearnModel(DeviationModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._n_iter = 10  # default

    def _train(self, cfg, states_and_parameters, deviations, states_and_parameters_validation=None,
               deviations_validation=None, rescale=True, wandb_experiment=None, model_args={}):
        estimator = self._sklearn_model_cls(**model_args)

        def custom_loss_func(y1, y2):
            res = np.mean(self.evaluate_loss(y1, y2))
            return res

        scorer = make_scorer(custom_loss_func, greater_is_better=False)
        self.fold_cross_validation = 4
        rf_random = RandomizedSearchCV(estimator=estimator, param_distributions=self._random_grid,
                                       scoring=scorer,  # self.evaluate_loss,
                                       n_iter=self._n_iter, cv=self.fold_cross_validation, verbose=1, random_state=42,
                                       n_jobs=1)  # Fit the random search model 70 works
        rf_random.fit(states_and_parameters, deviations.flatten())
        self._model = rf_random

    def _predict(self, states_and_parameters):
        if self.with_conf:
            mean, std = self._model.best_estimator_.predict(states_and_parameters, return_std=True)
            return mean, std
        return self._model.predict(states_and_parameters)

    def _load_model(self, filename):
        try:
            self._model = np.load(filename, allow_pickle=True).item()
        except AttributeError:
            self._model = np.load(filename, allow_pickle=True)

    def _save_model(self, filename):
        try:
            del self._model.scorer_
            del self._model.scoring
        except AttributeError:
            print("Didn't have a scorer")

        with open(filename, 'wb') as f:
            dump(self._model, f)
