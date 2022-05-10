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

def train_gp(data, labels):
    train_x = torch.from_numpy(data).cuda()
    train_y = torch.from_numpy(labels).cuda()
    #likelihood = gpytorch.likelihoods.GaussianLikelihood()
    #gp = SingleTaskGP(train_X, train_Y)
    large_var_constraint = gpytorch.constraints.Interval(5, 10)
    small_var_constraint = gpytorch.constraints.Interval(0.0000000001, 0.00001)
    covar_module = gpytorch.kernels.LinearKernel(variance_constraint=large_var_constraint) #(RBFKernel() + LinearKernel())  #gpytorch.means.ZeroMean()
    model = SingleTaskGP(train_X=train_x, train_Y=train_y, covar_module=covar_module).cuda()
    mll = ExactMarginalLogLikelihood(model.likelihood, model).cuda()
    _ = fit_gpytorch_model(mll)
    with torch.no_grad():
        observed_var = torch.pow(model.posterior(train_x).mean - train_y, 2)
    model_heter = HeteroskedasticSingleTaskGP(train_x, train_y, observed_var)
    mll_heter = ExactMarginalLogLikelihood(model_heter.likelihood, model_heter)
    _ = fit_gpytorch_model(mll_heter, options={'max_iter':8})
    return model_heter, mll_heter


def large_underlying_process():
    #env_size = (50,60,50)
    env_size = (4,6,5)
    data = np.random.normal(size=env_size).flatten()
    label = np.random.uniform(-1,1)
    return data, label


def normal_underlying_process():
    data = np.random.uniform(low=-3,high=3)
    if data < 0:
        noise_mag = 0.1*abs(data)
    else:
        noise_mag = 0.01
    label = 0.4*np.sin(data) + noise_mag*np.random.normal()
    return data, label


def generate_data(N):
    datas = []
    labels = []
    for _ in range(N):
        data, label = normal_underlying_process()
        datas.append(data)
        labels.append(label)
    return np.vstack(datas), np.vstack(labels)

def predict_gp(data, model, likelihood):
    test_x = torch.from_numpy(data).cuda()
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #observed_pred = likelihood(model(test_x))
        observed_pred = model.posterior(test_x, observation_noise=True)
    return observed_pred


def fit_and_plot():
    n_data = 20
    datas, labels = generate_data(n_data)
    #data_scaler, label_scaler = make_scaler(datas), make_scaler(labels)
    #datas_scaled = data_scaler.transform(datas)
    #labels_scaled = label_scaler.transform(labels)
    model, likelihood = train_gp(datas, labels)
    test_datas = np.linspace(-3,3,200)
    #test_datas.sort()
    observed_pred = predict_gp(test_datas, model, likelihood)
    lower, upper = observed_pred.mvn.confidence_region()
    # Plot predictive means as blue line
    plt.plot(test_datas, observed_pred.mean.cpu().detach().numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    plt.fill_between(test_datas, lower.cpu().detach().numpy(), upper.detach().cpu().numpy(), alpha=0.5)
    plt.xlabel("Theta")
    plt.ylabel("dhat")
    plt.ylim([-1,1])
    plt.scatter(datas, labels, label="process")
    plt.plot(test_datas, 0.4*np.sin(test_datas))
    plt.show()


def time_testing():
    for n_data in [2,4,6,8,10,50, 100, 300, 500]:
        datas, labels = generate_data(n_data)
        start_time = time.time()
        model, likelihood = train_gp(datas, labels)
        end_time = time.time()
        print("n_data", n_data, "Training time", end_time-start_time)
        
        for n_test_data in [2,5,100,300,500]:
            test_datas, test_labels = generate_data(n_test_data)
            start_time = time.time()
            observed_pred = predict_gp(test_datas, model, likelihood)
            end_time = time.time()
            print("n_test_data", n_test_data, "Inference time", end_time-start_time)

        del model
        del likelihood
if __name__=="__main__":
    fit_and_plot()
