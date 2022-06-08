import pickle
from sklearn.model_selection import train_test_split
import wandb
from pathlib import Path
import numpy as np
import time
from typing import Any, List, Optional
from pickle import dump
import torch
import gpytorch
import matplotlib.pyplot as plt
from botorch.models import HeteroskedasticSingleTaskGP, SingleTaskGP
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch import fit_fully_bayesian_model_nuts
from botorch.fit import fit_gpytorch_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import NormalPrior
from botorch.models.gp_regression import SingleTaskGP, MIN_INFERRED_NOISE_LEVEL
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import Log, OutcomeTransform
from botorch.models.utils import validate_input_scaling
from botorch.optim.utils import sample_all_priors 
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.likelihoods.gaussian_likelihood import (
    GaussianLikelihood,
    _GaussianLikelihoodBase,
)
from gpytorch.likelihoods.noise_models import HeteroskedasticNoise
from gpytorch.distributions.multivariate_normal import MultivariateNormal

from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.mlls.noise_model_added_loss_term import NoiseModelAddedLossTerm
from gpytorch.priors.smoothed_box_prior import SmoothedBoxPrior
from torch import Tensor
from plan_abstractions.models.iterative_heter_gp import IterativeHeteroskedasticSingleTaskGP
import os
np.random.seed(0)
import gzip
#os.environ['WANDB_MODE'] = 'dryrun'


def train_gp(data, labels, optimize_using_val=False, val_size=0.5, experiment=None):
    train_x = torch.from_numpy(data).cuda()
    train_y = torch.from_numpy(labels).cuda()
    covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(
                    nu=2.5,
                    #eps=0.001,
                    ard_num_dims = data.shape[1],
                    #lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
                    lengthscale_constraint=gpytorch.constraints.Interval(1, 200), #9 was okay
                ),
                outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
            )
    if optimize_using_val:
        num_val = int(val_size*len(data))
        val_x = train_x[:num_val]
        val_y = train_y[:num_val]
        train_x =  train_x[num_val:]
        train_y =  train_y[num_val:]
    else:
        val_x = train_x.clone()
        val_y = train_y.clone()

    model = SingleTaskGP(train_X=train_x, train_Y=train_y, covar_module=covar_module).cuda()
    #model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(0.001))
    #saas_model = SaasFullyBayesianSingleTaskGP(train_X=train_x, train_Y=train_y)#, train_Yvar=torch.full_like(train_y, 1e-6))
    mll = ExactMarginalLogLikelihood(model.likelihood, model).cuda()
    #mll = None
    _ = fit_gpytorch_model(mll, max_retries=10, options={'disp':False})
    SMOKE_TEST=False
    WARMUP_STEPS = 16 if not SMOKE_TEST else 1
    NUM_SAMPLES = 64 if not SMOKE_TEST else 1
    THINNING = 1 if not SMOKE_TEST else 1
    #fit_fully_bayesian_model_nuts(
    #saas_model, warmup_steps=WARMUP_STEPS, num_samples=NUM_SAMPLES, thinning=THINNING, disable_progbar=False, max_tree_depth=6)
    #run_hyperparam_optimization(model, mll, train_x, train_y, val_x, val_y, experiment=experiment)
    #return train_het_gp(model, val_x, val_y)
    return model, mll

def train_het_gp_iterative(model, train_x, train_y):
    model = IterativeHeteroskedasticSingleTaskGP(train_x, train_y, False)
    model.fit()
    return model, None

def select_best_mcmc_sample(base_model, train_x, train_y):
    model_means = base_model.posterior(train_x).mean.detach().cpu().numpy()
    gt = train_y.cpu().numpy()
    losses = np.linalg.norm(model_means-gt, axis=1)

def train_het_gp_saas(base_model, train_x, train_y):
    base_model.eval()
    #best_idx = select_best_mcmc_sample(base_model, train_x, train_y)
    with torch.no_grad():
        observed_var = torch.pow(base_model.posterior(train_x).mean - train_y, 2)
    noise_model = SaasFullyBayesianSingleTaskGP(train_X=train_x, train_Y=observed_var)#, train_Yvar=torch.full_like(train_y, 1e-6))
    SMOKE_TEST=False
    WARMUP_STEPS = 16 if not SMOKE_TEST else 1
    NUM_SAMPLES = 512 if not SMOKE_TEST else 1
    THINNING = 16 if not SMOKE_TEST else 1
    fit_fully_bayesian_model_nuts(
    noise_model, warmup_steps=WARMUP_STEPS, num_samples=NUM_SAMPLES, thinning=THINNING, disable_progbar=False, max_tree_depth=12)
    return base_model, noise_model


def predict_my_het_gp(target_model,noise_model, X):
    target_mvn = target_model.posterior(X).mvn
    noise_mvn = noise_model.posterior(X).mvn
    target_mean = target_mvn.mean
    target_covar = target_mvn.covariance_matrix
    noise_covar = torch.diag_embed(noise_mvn.mean.reshape(
    target_covar.shape[:-1]).max(
    torch.tensor(MIN_INFERRED_NOISE_LEVEL)))
    try:
        mvn = MultivariateNormal(
            target_mean, target_covar + noise_covar)
    except:
        print("Invalid covariance matrix")
        mvn = MultivariateNormal(target_mean, gpytorch.add_jitter(target_covar+noise_covar,0.01))

    return GPyTorchPosterior(mvn=mvn)

def train_het_gp(model, train_x, train_y):
    model.eval()
    with torch.no_grad():
        observed_var = torch.pow(model.posterior(train_x).mean - train_y, 2)
    covar_module_het = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(
                    nu=2.5,
                    #lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
                    ard_num_dims = train_x.shape[1],
                    #lengthscale_prior=NormalPrior(model.covar_module.base_kernel.lengthscale[0].detach(), scale=0.1),
                    #lengthscale_constraint=gpytorch.constraints.Interval(2, 300),
                ),
                #outputscale_prior=NormalPrior(model.covar_module.outputscale[0].detach(), scale=0.1),
            )
    #covar_module_het.base_kernel.raw_lengthscale = model.covar_module.base_kernel.raw_lengthscale
    model_heter = HeteroskedasticSingleTaskGP(train_x, train_y, observed_var,covar_module=covar_module_het)
    mll_heter = ExactMarginalLogLikelihood(model_heter.likelihood, model_heter)
    _ = fit_gpytorch_model(mll_heter, options={'lr':0.01, 'disp':False}) #, options={'max_iter':800})
    #print(f"Lengthscale {model_heter.covar_module.base_kernel.lengthscale}")
    return model_heter, mll_heter

def run_hyperparam_optimization(model, mll, train_x, train_y, val_x, val_y, experiment=None):
    # Find optimal model hyperparameters
    training_iter = 1500
    reset_every = training_iter + 1 #9000
    val_y = val_y.flatten()
    train_y = train_y.flatten()
    model.train()
    model.likelihood.train()
    mll.train()
    old_ls = []
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
    #lengthscale_prior=gpytorch.priors.GammaPrior(1.0, 0.1)
    lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(0.001, 10)
    noise_prior=gpytorch.priors.SmoothedBoxPrior(0.001, 0.05)

    for i in range(training_iter):
        if i % reset_every == 0 and i > 0:
            old_ls = model.covar_module.base_kernel.lengthscale.clone()
            model.covar_module.base_kernel.lengthscale = torch.abs(lengthscale_prior.sample(model.covar_module.base_kernel.lengthscale.shape))
            model.likelihood.noise_covar.raw_noise.data = noise_prior.sample(model.likelihood.noise_covar.raw_noise.shape)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters
            optimizer.zero_grad(set_to_none=True)

        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        #output = model(train_x)
        # Calc loss and backprop gradients
        train_loss = -mll(model(train_x), train_y)
        val_loss = -mll(model(val_x), val_y)
        if experiment is not None:
            experiment.log({"train_loss":train_loss})
            experiment.log({"val_loss":val_loss})
            experiment.log({"noise": model.likelihood.noise.item()})
            experiment.log({"lengthscale": torch.mean(model.covar_module.base_kernel.lengthscale)})
        (val_loss+0.1*train_loss).backward()
        #print("train_loss", train_loss)
        #print("val_loss", val_loss)

        #print(
        #    i + 1, training_iter, loss.item(),
        #    model.covar_module.base_kernel.lengthscale,
        #    model.likelihood.noise,
        #)
        optimizer.step()
    #experiment.log({"lengthscale":model.covar_module.base_kernel.lengthscale})
    #experiment.log({"noise":model.likelihood.noise})


def rope_data_to_state_and_action(data):
    state_keys = ['rope', 'right_gripper', 'left_gripper'] + ["joint_positions"]
    action_keys = ["left_gripper_position", "right_gripper_position"]
    flattened_state = []
    for state_key in state_keys:
        data_pt = data[f"predicted/{state_key}"]
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




def predict_gp(data, model, likelihood, noise_model=None):
    test_x = torch.from_numpy(data).cuda()
    model.eval()
    if noise_model is not None:
        return predict_my_het_gp(model,noise_model, test_x)

    if likelihood is not None:
        likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #test_pred = likelihood(model(test_x))
        test_pred = model.posterior(test_x, observation_noise=True)
    return test_pred

def make_scaler(data):
    ss = StandardScaler()
    ss.fit(data)
    return ss


class LogScaler(StandardScaler):
    def fit(self, datas):
        self.base = 30
        return super().fit(datas)

    def transform(self, datas):
        datas = np.log(datas)/np.log(self.base)
        return super().transform(datas)

    def inverse_transform(self, datas):
        res =  super().inverse_transform(datas)
        return self.base**res

class PowerScaler(StandardScaler):
    def fit(self, datas):
        self.base = 1.1
        return super().fit(datas)

    def transform(self, datas):
        datas = np.power(datas, 1/self.base)
        return super().transform(datas)

    def inverse_transform(self, datas):
        res =  super().inverse_transform(datas)
        return np.power(res, self.base)

def make_label_scaler(data):
    ss = StandardScaler()
    ss.fit(data)
    return ss

def clean_data(data):
    #remove columns with low std
    nontrivial_std = np.std(data, axis=0) > 0.01
    data = data[:, nontrivial_std]
    data = np.hstack([data[:,:75+6],data[:,-6:]])
    #remove data that's too similar
    return data, nontrivial_std

def save_to_wandb(datas_scaled, labels_scaled, model, data_scaler, label_scaler, save_file_root_dir, nontrivial_std, upload_files = True, experiment=None):
    if not upload_files:
        return experiment
    save_filename = save_file_root_dir / 'validation_model.pkl'
    data_filename = save_file_root_dir / 'other_data.pkl'
    deviation_scaler_filename = save_file_root_dir / 'deviation_scaler.pkl'
    state_and_parameter_scaler_filename = save_file_root_dir / 'state_and_parameter_scaler.pkl'
    torch.save(model.state_dict(), save_filename)
    data_dict = {'datas_scaled':datas_scaled, 'labels_scaled':labels_scaled, 'nonzero_std':nontrivial_std}

    with open(data_filename, 'wb') as f:
        dump(data_dict, f)

    with open(deviation_scaler_filename, 'wb') as f:
        dump(label_scaler, f)

    with open(state_and_parameter_scaler_filename, 'wb') as f:
        dump(data_scaler, f)

    filenames = [save_filename, deviation_scaler_filename, state_and_parameter_scaler_filename, data_filename]
    for filename in filenames:
        wandb.save(str(filename), base_path=str(save_file_root_dir))
    wandb.save(str(save_file_root_dir / '.save_file_root' / '*.yaml'), base_path=str(save_file_root_dir))

def remove_similar(datas, labels):
    _, indices = np.unique(datas.round(2), axis=0,return_index=True)
    return datas[indices], labels[indices]

def load_data(n_data=2000):
    data_dir = "data/adaptation_data/artifacts/iterative_lowest_error_3d9e3:v0"
    datas, labels = get_data_from_dir(data_dir, n_data)
    datas, labels = remove_similar(datas, labels)
    datas, nontrivial_std = clean_data(datas)
    return datas, labels, nontrivial_std

def analyze_data(datas, labels):
    distances_between_datas = np.linalg.norm(datas[:, None, :] - datas[None, :, :], axis=-1)
    distances_between_labels = np.linalg.norm(labels[:, None, :] - labels[None, :, :], axis=-1)
    #only care about upper triangular minus diagonal
    idx_as, idx_bs  = np.triu_indices(len(datas))
    state_distances = []
    label_distances = []
    for idx_a, idx_b in zip(idx_as, idx_bs):
        if idx_a == idx_b:
            continue
        state_distances.append(distances_between_datas[idx_a][idx_b])
        label_distances.append(distances_between_labels[idx_a][idx_b])
    #plt.scatter(state_distances, label_distances)
    state_distances = np.array(state_distances)
    label_distances = np.array(label_distances)
    plt.scatter(state_distances[state_distances < 1], label_distances[state_distances < 1])

    
def remove_high_error(datas, labels, high_error = 0, experiment=None):
    mask = (labels < high_error).flatten()
    experiment.log({"#train": np.sum(mask)})
    return datas[mask],labels[mask]

def fit_and_plot():
    datas, labels, nontrivial_std = load_data()

    experiment = wandb.init(entity="lagrassa", project= "gpmde", group="alex/sparse_data", name="remove0.08", config={})
    datas, labels = remove_high_error(datas, labels, high_error=0.08, experiment=experiment)
    data_scaler, label_scaler = make_scaler(datas), make_label_scaler(labels)
    datas_scaled = data_scaler.transform(datas)
    labels_scaled = label_scaler.transform(labels)
    train_datas, test_datas, train_labels, test_labels = train_test_split(datas_scaled, labels_scaled, test_size=0.1)

    print("Train datas shape", train_datas.shape)
    #model, likelihood = train_gp(train_datas, train_labels, experiment=experiment)
    model, likelihood = train_gp(train_datas, train_labels, experiment=experiment)
    noise_model = None
    test_idxs = np.argsort(test_labels.flatten())
    test_labels = label_scaler.inverse_transform(test_labels[test_idxs])
    test_datas = test_datas[test_idxs]
    #known_good_datas, known_good_labels = get_data_from_dir("data/adaptation_data/artifacts/known_good_2_meta_1652277930:v0", 500)
    known_good_datas, known_good_labels = get_data_from_dir("data/adaptation_data/artifacts/known_good_4_iterative_lowest_error_0.08-3d9e3_1654534306:v0", 500)

    known_good_datas = clean_data(known_good_datas)[0]


    train_pred = predict_gp(train_datas, model, likelihood, noise_model=noise_model)
    train_plus_noise_pred = predict_gp(train_datas+0.05*np.random.random(), model, likelihood, noise_model=noise_model)
    test_pred = predict_gp(test_datas, model, likelihood, noise_model=noise_model)
    known_good_pred = predict_gp(data_scaler.transform(known_good_datas), model, likelihood)
    save_file_root_dir = Path("data/test_file_root_dir")
    save_to_wandb(train_datas, train_labels, model, data_scaler, label_scaler,save_file_root_dir, nontrivial_std, upload_files=False)
    lower, upper = test_pred.mvn.confidence_region()
    pred_error_unscaled = test_pred.mean.cpu().detach().numpy()
    if len(pred_error_unscaled.shape) == 3:
        pred_error = label_scaler.inverse_transform(pred_error_unscaled[0])
    else:
        pred_error = label_scaler.inverse_transform(pred_error_unscaled)
    
    std_pred = np.sqrt(test_pred.variance.cpu().detach().numpy()).flatten()
    std = ((std_pred ** 2) * label_scaler.var_) ** 0.5
    plot_d_dhat(train_pred, label_scaler.inverse_transform(train_labels), data_scaler, label_scaler, "train", experiment)
    plot_d_dhat(train_plus_noise_pred, label_scaler.inverse_transform(train_labels), data_scaler, label_scaler, "train_plus_noise", experiment)
    
    plot_d_dhat(test_pred,test_labels, data_scaler, label_scaler, "test", experiment)
    plot_d_dhat(known_good_pred, known_good_labels, data_scaler, label_scaler, "known_good", experiment)
    #plt.scatter(test_labels, pred_error)
    abs_error = np.abs(pred_error.flatten() - test_labels.flatten())
    #plt.scatter(abs_error, std)
    #plt.xlabel("error")
    #plt.ylabel("std")
    #plt.show()
    #plt.savefig("errorall.png")
    #high_conf_mask = (std < 0.1).flatten()
    #high_conf_samples = test_datas[high_conf_mask]
    #print("Num high conf samples", len(high_conf_samples))
    #plt.scatter(test_labels[high_conf_mask], pred_error[high_conf_mask])

    #plt.plot(test_labels.flatten(), pred_error)
    #plt.fill_between(pred_error.flatten()[high_conf_mask], label_scaler.inverse_transform(lower.cpu().detach().numpy()[high_conf_mask].reshape(1,-1)), label_scaler.inverse_transform(upper.detach().cpu().numpy()[high_conf_mask].reshape(-1,1)), alpha=0.5)
    #plt.gca().set_aspect('equal', adjustable='box')
    #plt.xlabel("GT error")
    #plt.ylabel("pred error")
    #plt.show()

def plot_d_dhat(botorch_pred, d_gt, data_scaler, label_scaler, label, experiment):
    plt.xlabel("d (GT)")
    plt.ylabel("dhat")
    d_hat_scaled = botorch_pred.mean.cpu().detach().numpy()
    if len(d_hat_scaled.shape) == 3:
        d_hat_scaled = d_hat_scaled[0]
    d_hat = label_scaler.inverse_transform(d_hat_scaled)
    std_pred = np.sqrt(botorch_pred.variance.cpu().detach().numpy())
    std = ((std_pred ** 2) * label_scaler.var_) ** 0.5
    plt.gca().set_aspect('equal', adjustable='box')

    high_conf_mask = (std < 0.14).flatten()
    low_error = 0.1
    abs_error = np.abs(d_hat.flatten() - d_gt.flatten())
    abs_error_low = np.abs(d_hat.flatten() - d_gt.flatten())[(d_gt < low_error).flatten()]
    dd_data = [[x, y] for (x, y) in zip(d_gt.flatten(), d_hat.flatten())]
    dd_data_low_error = [[x, y] for (x, y) in zip(d_gt.flatten(), d_hat.flatten()) if x < low_error]
    e_std_data = [[x, y] for (x, y) in zip(std.flatten(), abs_error.flatten())]
    dd_table = wandb.Table(data=dd_data, columns = ["d_gt", "d_hat"])
    dd_table_low_error = wandb.Table(data=dd_data_low_error, columns = ["d_gt", "d_hat"])
    e_std_table = wandb.Table(data=e_std_data, columns = ["std", "abs_error"])
    wandb.log({f"d_dhat:{label}" : wandb.plot.scatter(dd_table, "d_gt", "d_hat",title=f"{label} pred d (mean) v. true d")})
    wandb.log({f"d_dhat:{label} low error" : wandb.plot.scatter(dd_table_low_error, "d_gt", "d_hat",title=f"{label} dhat d_gt low error ")})
    wandb.log({f"error_std:{label}" : wandb.plot.scatter(e_std_table, "std", "abs_error",title=f"abs error / std {label}")})
    experiment.log({f"mean_error{label}":np.mean(abs_error)})
    experiment.log({f"mean_error_low{label}":np.mean(abs_error)})
    experiment.log({f"std_error{label}":np.std(abs_error)})
    #plt.scatter(d_gt, d_hat, label="all")
    #plt.scatter(d_gt[high_conf_mask], d_hat[high_conf_mask], label="std< 0.14")
    #plt.legend()
    #plt.xlim([0,1])
    #plt.ylim([0,1])
    #experiment.log({f"chart:{label}": wandb.Image(plt)})
    #experiment.log({f"d_dhat:{label}": plt})
    #plt.clf()

    #plt.xlabel("std")
    #plt.ylabel("d dhat error")
    
    #plt.scatter(std, abs_error) 
    #experiment.log({f"error_std:{label}": plt})
    #plt.clf()





if __name__=="__main__":
    fit_and_plot()
