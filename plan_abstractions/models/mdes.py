import torch
from pytorch_lightning.loggers import WandbLogger
import torch as F
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.gaussian_process.kernels import Matern
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import pytorch_lightning as pl
import wandb
from pathlib import Path
from pickle import dump

from plan_abstractions.learning.deviation_pred_data import DeviationPredDataset
from plan_abstractions.models.mde_base_classes import DeviationModel, SKLearnModel
try:
    from plan_abstractions.models.gnn_dev_model import ECNModel
except ImportError:
    print("UNable to import gnn_dev_model")

import sys

sys.modules['plan_abstractions.models.model_validation_models'] = sys.modules[
    __name__]  # Necessary due to refactoring to get old pickled models to work


def create_deviation_wrapper_from_cfg(cfg, cache_dir='/tmp', graphs=False):
    dim_states_and_params = 12 + 5  # TODO lagrassa dont hardcode
    run_path = cfg['run_path']
    root_path = Path(cache_dir) / run_path
    model_file_name = Path(root_path) / "validation_model.pkl"
    deviation_scaler_fn = Path(root_path) / "deviation_scaler.pkl"
    state_and_parameter_scaler_fn = Path(root_path) / "state_and_parameter_scaler.pkl"
    node_state_and_parameter_scaler_fn = Path(root_path) / "state_and_parameter_scaler_node.pkl"
    edge_state_and_parameter_scaler_fn = Path(root_path) / "state_and_parameter_scaler_edge.pkl"
    if graphs:
        files_to_restore = [model_file_name, deviation_scaler_fn, node_state_and_parameter_scaler_fn, edge_state_and_parameter_scaler_fn]
    else:
        files_to_restore = [model_file_name, deviation_scaler_fn, state_and_parameter_scaler_fn]
    # files_to_restore = [deviation_scaler_fn, state_and_parameter_scaler_fn]
    for filepath in files_to_restore:
        wandb.restore(
            filepath.name,
            run_path=run_path,
            root=root_path
        )

    if cfg['type'] == "MLPModel":
        deviation_model = MLPModel(cfg, dim_states_and_params)
    elif cfg['type'] == "RFRModel":
        deviation_model = RFRModel(cfg)
    elif cfg['type'] == "KNNRegressorModel":
        deviation_model = KNNRegressorModel(cfg)
    elif cfg['type'] == "GPRModel":
        deviation_model = GPRModel(cfg)
    elif cfg['type'] == "LinearRegressionModel":
        deviation_model = LinearRegressionModel(cfg)
    elif cfg['type'] == "SVCModel":
        deviation_model = SVCModel(cfg)
    elif cfg['type'] == "ECNModel":
        deviation_model = ECNModel(cfg)
    else:
        raise ValueError("Invalid model type")
    deviation_model.load_model(model_file_name, deviation_scaler_fn, state_and_parameter_scaler_fn)
    return deviation_model


class RFRModel(SKLearnModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        n_estimators = [int(x) for x in np.linspace(start=50, stop=3000, num=20)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=20)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]  # Create the random grid
        self._random_grid = {'n_estimators': n_estimators,
                             'max_features': max_features,
                             'max_depth': max_depth,
                             'min_samples_split': min_samples_split,
                             'min_samples_leaf': min_samples_leaf,
                             'bootstrap': bootstrap}
        self._sklearn_model_cls = RFR


class MLP(pl.LightningModule):

    def __init__(self, input_dim, N, state_and_param_scaler, deviation_scaler, c1=None, c2=None, lr=1e-4, weight_decay=1e-5,
                 classification=False):
        super().__init__()
        self.loss_scale = 1
        self._c1 = c1
        self._c2 = c2  # 2 #20
        self._lr = lr
        self._weight_decay = weight_decay
        self._state_and_parameter_scaler = state_and_param_scaler
        self._deviation_scaler = deviation_scaler
        if classification:
            last_layer = nn.Linear(N, 2)
        else:
            last_layer = nn.Linear(N, 1)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, N),
            nn.ReLU(),
            nn.Linear(N, N),
            nn.Dropout(0.1),
            nn.ReLU(),
            last_layer
        )
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.))
        self.cel = nn.CrossEntropyLoss(weight=torch.tensor([1., 1.77]))
        self._classification = classification

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.layers(x)
        if self._classification:
            # loss = self.loss_scale * self.bce(y_hat, y)
            loss = self.loss_scale * self.cel(y_hat, y)
        else:
            asym_loss = self._c1 * torch.mean(torch.pow(F.relu(y - y_hat), 2)) + self._c2 * torch.mean(
                torch.pow(F.relu(y_hat - y), 2))
            loss = self.loss_scale * asym_loss
            y_unscaled = self._deviation_scaler.inverse_transform(y)
            yhat_unscaled = self._deviation_scaler.inverse_transform(y_hat.detach().numpy())
            false_negatives_3, false_negatives_5, false_positives_3, false_positives_5 = self.sensitivity_analysis_metrics(
                y_unscaled, yhat_unscaled)
            train_mean_error = np.mean(np.abs(yhat_unscaled - y_unscaled))
            self.log('train_mean_error', train_mean_error)

        self.log('train_loss', loss)
        return loss

    def sensitivity_analysis_metrics(self, y, y_hat, prefix="train"):
        diff = y - y_hat
        false_negatives_7 = np.sum(diff > 0.07) / len(diff)
        false_negatives_10 = np.sum(diff > 0.10) / len(diff)
        false_positives_7 = np.sum(diff < -0.07) / len(diff)
        false_positives_10 = np.sum(diff < -0.10) / len(diff)
        self.log(f"{prefix}_false_negatives_7", false_negatives_7)
        self.log(f"{prefix}_false_negatives_10", false_negatives_10)
        self.log(f"{prefix}_false_positives_7", false_positives_7)
        self.log(f"{prefix}_false_positives_10", false_positives_10)
        return false_negatives_7, false_negatives_10, false_positives_7, false_positives_10

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.layers(x)
        if self._classification:
            loss = self.loss_scale * self.ce(y_hat, y)
        else:
            # loss = self.loss_scale * self.mse(y_hat, y)
            asym_loss = self._c1 * torch.mean(torch.pow(F.relu(y - y_hat), 2)) + self._c2 * torch.mean(
                torch.pow(F.relu(y_hat - y), 2))
            y_unscaled = self._deviation_scaler.inverse_transform(y)
            yhat_unscaled = self._deviation_scaler.inverse_transform(y_hat)
            validation_mean_error = np.mean(np.abs(yhat_unscaled - y_unscaled))
            self.log('validation_mean_error', validation_mean_error)
            self.sensitivity_analysis_metrics(y_unscaled, yhat_unscaled, "validation")
        self.log('validation_loss', asym_loss)
        return asym_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        return optimizer


class MLPModel(DeviationModel):
    def __init__(self, cfg, dim_states_and_params):
        super().__init__(cfg)
        self._dim_states_and_params = dim_states_and_params
        self._lr = None
        if "train_cfg" in cfg.keys():
            self._lr =  cfg["train_cfg"].get("lr", 5e-3)

    def _train(self, cfg, states_and_parameters, deviations, states_and_parameters_validation=None,
               deviations_validation=None, rescale=True, wandb_experiment=None):
        N = cfg.get("N", 16)
        self._model = MLP(self._dim_states_and_params, N, self._state_and_parameter_scaler, self._deviation_scaler,
                          lr=self._lr, c1=self._c1, c2=self._c2, weight_decay =self._train_cfg.get('weight_decay'))
        dataset = DeviationPredDataset(states_and_parameters, deviations)
        validation_dataset = DeviationPredDataset(states_and_parameters_validation, deviations_validation)
        # wandb_logger = WandbLogger(project="model_validation", id="only_easy_data_longer")
        # wandb_logger = WandbLogger(project="model_validation", id="only_planning_data_longer")
        # wandb_logger = WandbLogger(project="model_validation", id="all_data")
        wandb_logger = WandbLogger(project="model_validation", experiment=wandb_experiment,
                                   id=str(np.random.randint(0, 700)))
        trainer = pl.Trainer(deterministic=True, max_epochs=int(cfg.get("max_epochs", 20)), check_val_every_n_epoch=1,
                             logger=wandb_logger)
        # trainer.fit(self._model, DataLoader(dataset, batch_size=256))
        trainer.fit(self._model, train_dataloader=DataLoader(dataset, batch_size=cfg.get("batch_size", 32)),
                    val_dataloaders=[DataLoader(validation_dataset, batch_size=256)])


    def _predict(self, states_and_parameters):
        return self._model(torch.FloatTensor(states_and_parameters)).detach().numpy()

    def _load_model(self, filename):
        self._model = torch.load(filename)

    def _save_model(self, filename):
        torch.save(self._model, filename)


class LinearRegressionModel(SKLearnModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._random_grid = {}
        self._sklearn_model_cls = LinearRegression
        self._n_iter = 1  # Nothing to optimize

    def _train_old(self, cfg, states_and_parameters, deviations, states_and_parameters_validation=None,
                   deviations_validation=None, rescale=True, wandb_experiment=None):
        self._model = LinearRegression()
        self._model.fit(states_and_parameters, deviations)

    def _predict(self, states_and_parameters):
        return self._model.predict(states_and_parameters)

    def _load_model(self, filename):
        try:
            self._model = np.load(filename, allow_pickle=True).item()
        except AttributeError:
            self._model = np.load(filename, allow_pickle=True)

    def _save_model(self, filename):
        with open(filename, 'wb') as f:
            dump(self._model, f)


class GPRModel(SKLearnModel):
    def __init__(self, cfg):
        self._sklearn_model_cls = GaussianProcessRegressor
        super().__init__(cfg)
        kernel_list = [Matern(length_scale_bounds=[0.01, 1]), Matern(length_scale_bounds=[0.1, 5]),
                       Matern(length_scale_bounds=[0.01, 1], nu=2.5)]
        self._random_grid = {'kernel': kernel_list}

    def _train(self, cfg, states_and_parameters, deviations, states_and_parameters_validation=None,
               deviations_validation=None, rescale=True, wandb_experiment=None):
        # self._model = GaussianProcessRegressor(kernel=Matern(length_scale_bounds= [0.01, 1]))
        kwargs = {'n_restarts_optimizer': 5}
        super()._train(cfg, states_and_parameters, deviations,
                       states_and_parameters_validation=states_and_parameters_validation,
                       deviations_validation=deviations_validation, rescale=rescale, wandb_experiment=wandb_experiment,
                       model_args=kwargs)
        # self._model.fit(states_and_parameters, deviations)


class KNNRegressorModel(SKLearnModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._random_grid = {'n_neighbors': [1, 20],
                             'weights': ["distance", "uniform"],
                             'p': [2]}
        self._sklearn_model_cls = KNeighborsRegressor

    def _train_old(self, states_and_parameters, deviations, states_and_parameters_validation=None,
                   deviations_validation=None, rescale=True, wandb_experiment=None):
        self._model = KNeighborsRegressor(n_jobs=10, n_neighbors=2, weights='distance', p=2, algorithm="auto")
        self._model.fit(states_and_parameters, deviations)


class SVCModel(DeviationModel):
    def __init__(self, cfg, C=10, gamma=0.01, neg_class_weight=200, kernel="rbf"):
        super().__init__(cfg)
        self._C = C
        self._gamma = gamma
        self._neg_class_weight = neg_class_weight
        self._kernel = kernel

    def _train(self, cfg, states_and_parameters, deviations, states_and_parameters_validation=None,
               deviations_validation=None, rescale=True, wandb_experiment=None):
        self._model = SVC(C=self._C, gamma=self._gamma, kernel=self._kernel,
                          class_weight={0: 1, 1: self._neg_class_weight})
        self._model.fit(states_and_parameters, deviations.flatten())

    def _predict(self, states_and_parameters):
        return self._model.predict(states_and_parameters)

    def _load_model(self, filename):
        self._model = np.load(filename, allow_pickle=True)

    def _save_model(self, filename):
        with open(filename, 'wb') as fgnn_d:
            dump(self._model, f)
