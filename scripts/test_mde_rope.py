import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import time
from typing import Any, List, Optional
import torch
import gpytorch
import matplotlib.pyplot as plt
from botorch.models import HeteroskedasticSingleTaskGP, SingleTaskGP
from botorch.fit import fit_gpytorch_model
from sklearn.preprocessing import StandardScaler
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.gp_regression import SingleTaskGP, MIN_INFERRED_NOISE_LEVEL
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import Log, OutcomeTransform
from botorch.models.utils import validate_input_scaling
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.likelihoods.gaussian_likelihood import (
    GaussianLikelihood,
    _GaussianLikelihoodBase,
)
from gpytorch.likelihoods.noise_models import HeteroskedasticNoise
from gpytorch.mlls.noise_model_added_loss_term import NoiseModelAddedLossTerm
from gpytorch.priors.smoothed_box_prior import SmoothedBoxPrior
from torch import Tensor
import os
np.random.seed(0)
import gzip
def train_gp(data, labels):
    train_x = torch.from_numpy(data).cuda()
    train_y = torch.from_numpy(labels).cuda()
    #likelihood = gpytorch.likelihoods.GaussianLikelihood()
    #gp = SingleTaskGP(train_X, train_Y)
    large_var_constraint = gpytorch.constraints.Interval(0.001, 5)
    #small_var_constraint = gpytorch.constraints.Interval(0.0000000001, 0.00001)
    #covar_module = gpytorch.kernels.LinearKernel(variance_constraint=large_var_constraint) #(RBFKernel() + LinearKernel())  #gpytorch.means.ZeroMean()
    #lengthscale_prior = gpytorch.priors.SmoothedBoxPrior(a=0.000001, b=100)
    #lengthscale_constraint = gpytorch.constraints.Interval(0.01, 0.1)
    lengthscale_constraint=None
    lengthscale_prior=gpytorch.priors.GammaPrior(3,6)
    covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(lengthscale_prior=lengthscale_prior, lengthscale_constraint=lengthscale_constraint))
    covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(
                    nu=2.5,
                    lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
                    #lengthscale_constraint=gpytorch.constraints.Interval(0.8, 10),
                ),
                outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
            )
    model = SingleTaskGP(train_X=train_x, train_Y=train_y, covar_module=covar_module).cuda()
    #model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(0.9))
    mll = ExactMarginalLogLikelihood(model.likelihood, model).cuda()
    _ = fit_gpytorch_model(mll, max_retries=10)
    print(f"Lengthscale {model.covar_module.base_kernel.lengthscale.item():>4.3f}")
    print("noise", model.likelihood.noise.item())
    with torch.no_grad():
        observed_var = torch.pow(model.posterior(train_x).mean - train_y, 2)
    model_heter = HeteroskedasticSingleTaskGP(train_x, train_y, observed_var)
    mll_heter = ExactMarginalLogLikelihood(model_heter.likelihood, model_heter)
    #options = {'max_iter': 8000, 'disp': 0, 'num_restarts':6000, 'lr':0.001}
    options=None
    _ = fit_gpytorch_model(mll_heter, options=options) #, options={'max_iter':800})
    return model_heter, mll_heter

def rope_data_to_state_and_action(data):
    state_keys = ['rope', 'right_gripper', 'left_gripper'] + ["joint_positions"]
    action_keys = ["left_gripper_position", "right_gripper_position"]
    flattened_state = []
    for state_key in state_keys:
        data_pt = data[state_key]
        flattened_state.extend(data_pt[0].flatten())
    flattened_actions =  []
    for action_key in action_keys:
        flattened_actions.extend(data[action_key].flatten())
    return flattened_state, flattened_actions


def get_data_from_dir(data_dir, n_data = 5):
    errors = []
    states = []
    actions = []
    for fn in os.listdir(data_dir):
        if "gz" in fn or "hjson" in fn or "txt" in fn:
            continue
        metadata = np.load(os.path.join(data_dir, fn), allow_pickle=1)
        data_fn = metadata["data"]
        error = np.linalg.norm(metadata['error'])
        with gzip.open(os.path.join(data_dir, data_fn)) as f:
            data = pickle.load(f)
            state, action = rope_data_to_state_and_action(data)
            actions.append(action)
            states.append(state)
        errors.append(error)
        if len(errors) > n_data:
            break
        if len(errors) % 100 == 0:
            print(len(errors))
    states_and_actions = np.hstack([np.vstack(states), np.vstack(actions)])
    errors = np.vstack(errors)
    return states_and_actions, errors




def predict_gp(data, model, likelihood):
    test_x = torch.from_numpy(data).cuda()
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #observed_pred = likelihood(model(test_x))
        observed_pred = model.posterior(test_x, observation_noise=True)
    return observed_pred

def make_scaler(data):
    ss = StandardScaler()
    ss.fit(data)
    return ss

def clean_data(data):
    #remove columns with low std
    nontrivial_std = np.std(data, axis=0) > 0.01
    data = data[:, nontrivial_std]
    return data

def fit_and_plot():
    n_data = 8400
    data_dir = "data/adaptation_data/manual_val_unadapted_1649864597"
    datas, labels = get_data_from_dir(data_dir, n_data)
    datas = clean_data(datas)
    data_scaler, label_scaler = make_scaler(datas), make_scaler(labels)
    datas_scaled = data_scaler.transform(datas)
    labels_scaled = label_scaler.transform(labels)
    train_datas, test_datas, train_labels, test_labels = train_test_split(datas_scaled, labels_scaled, test_size=0.92)
    print("Train datas shape", train_datas.shape)
    model, likelihood = train_gp(train_datas, train_labels)
    test_idxs = np.argsort(test_labels.flatten())
    test_labels = label_scaler.inverse_transform(test_labels[test_idxs])
    test_datas = test_datas[test_idxs]
    observed_pred = predict_gp(test_datas, model, likelihood)
    lower, upper = observed_pred.mvn.confidence_region()
    pred_error_unscaled = observed_pred.mean.cpu().detach().numpy().flatten()
    pred_error = label_scaler.inverse_transform(pred_error_unscaled)
    std_pred = np.sqrt(observed_pred.variance.cpu().detach().numpy())
    std = ((std_pred ** 2) * label_scaler.var_) ** 0.5
    #plt.scatter(test_labels, pred_error)
    abs_error = np.abs(pred_error_unscaled.flatten() - test_labels.flatten())
    plt.scatter(abs_error, std)
    plt.xlabel("error")
    plt.ylabel("std")
    plt.show()
    high_conf_mask = (std < 0.05).flatten()
    high_conf_samples = test_datas[high_conf_mask]
    print("Num high conf samples", len(high_conf_samples))
    plt.xlabel("d (GT)")
    plt.ylabel("dhat")
    #plt.scatter(test_labels, upper.cpu().detach().numpy())
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(test_labels[high_conf_mask], pred_error[high_conf_mask])
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()
    print("Mean error", np.mean(np.abs(test_labels-pred_error)))

    plt.plot(test_labels.flatten(), pred_error)
    plt.fill_between(test_labels.flatten(), label_scaler.inverse_transform(lower.cpu().detach().numpy()), label_scaler.inverse_transform(upper.detach().cpu().numpy()), alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("GT error")
    plt.ylabel("pred error")
    plt.show()

if __name__=="__main__":
    fit_and_plot()
