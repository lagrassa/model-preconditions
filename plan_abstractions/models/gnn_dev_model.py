from pytorch_lightning.loggers import WandbLogger
try:
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import MessagePassing, BatchNorm
    from torch_geometric.nn.conv import NNConv
except ModuleNotFoundError:
    print("unable to import torch_geometric. GNN operations wont work")
import torch
from torch.nn import Sequential as Seq, Linear, ReLU
import torch as F
from torch_utils import get_numpy

from .mde_base_classes import DeviationModel
import numpy as np
import matplotlib.pyplot as plt
import wandb
from pickle import dump

from ..learning import copy_graph
from ..learning.deviation_pred_data import RealGraphDeviationDataset
from ..utils import append_postfix_to_filename

np.random.seed(0)
torch.manual_seed(0)

class ECNModel(DeviationModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        #does own scaling as well
        self._is_graph_model = True


    def _train(self, cfg, states_and_parameters, deviations, states_and_parameters_validation=None,
               deviations_validation=None, rescale=True, wandb_experiment=None, model_args={}):
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        train_dataset = RealGraphDeviationDataset("real", cfg, states_and_parameters=states_and_parameters, deviations=deviations, fit_scaler=True)
        self._deviation_scaler = train_dataset.deviation_scaler
        self._node_feature_scaler = train_dataset.node_features_scaler
        self._edge_feature_scaler = train_dataset.edge_features_scaler
        validation_dataset = RealGraphDeviationDataset("real", cfg, states_and_parameters=states_and_parameters_validation,
                                                       deviations=deviations_validation,
                                                       fit_scaler=False,
                                                       node_features_scaler=self._node_feature_scaler,
                                                       edge_features_scaler=self._edge_feature_scaler,
                                                       deviation_scaler=self._deviation_scaler)

        self._model = Net(train_dataset).to(device)
        run_train(self._model, train_dataset, validation_dataset, plot=False, deviation_scaler=self._deviation_scaler, train_cfg=self._train_cfg, wandb_experiment=wandb_experiment)

    def fit_scaler(self, states_and_parameters, deviations, rescale=False):
        class DummyScaler:
            def transform(self, data):
                return data
            def inverse_transform(self, data):
                return data
        self._state_and_parameter_scaler = DummyScaler()
        self._deviation_scaler = DummyScaler()

    def predict(self, states_and_parameters_unscaled, already_transformed_state_vector=False,
                state_ndim=None):

        self._model.eval()
        states_and_parameters = []
        for unscaled_data in states_and_parameters_unscaled:
            new_graph = copy_graph(unscaled_data)
            new_graph.x = torch.tensor(self._node_feature_scaler.transform(get_numpy(unscaled_data.x)), dtype=torch.float)
            new_graph.edge_attr = torch.tensor(self._edge_feature_scaler.transform(get_numpy(unscaled_data.edge_attr)), dtype=torch.float)
            states_and_parameters.append(new_graph)

        result = np.ones((len(states_and_parameters),))
        for i, graph in enumerate(states_and_parameters):
            pred_deviation = self._deviation_scaler.inverse_transform(self._model(graph).detach().cpu().numpy())
            result[i] = pred_deviation
        return result

    def save_model(self, model_fn, deviation_scaler_fn, state_and_parameter_scaler_fn):
        torch.save(self._model, model_fn)
        with open(deviation_scaler_fn, 'wb') as f:
            dump(self._deviation_scaler, f)

        with open(f"{append_postfix_to_filename(state_and_parameter_scaler_fn, 'node')}", 'wb') as f:
            dump(self._node_feature_scaler, f)

        with open(f"{append_postfix_to_filename(state_and_parameter_scaler_fn, 'edge')}", 'wb') as f:
            dump(self._edge_feature_scaler, f)

    def load_model(self, model_fn, deviation_scaler_fn, state_and_parameter_scaler_fn):
        self._deviation_scaler = np.load(deviation_scaler_fn, allow_pickle=True)
        self._node_feature_scaler = np.load(append_postfix_to_filename(state_and_parameter_scaler_fn, 'node'), allow_pickle=True)
        self._edge_feature_scaler = np.load(append_postfix_to_filename(state_and_parameter_scaler_fn, 'edge'), allow_pickle=True)
        self._model = torch.load(model_fn)
        torch.save(self._model, model_fn)

class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        in_channels = dataset.num_node_features
        num_in_node_features = dataset.num_node_features
        num_in_edge_features = dataset.num_edge_features
        nn1 = Seq(Linear(num_in_edge_features, 25), ReLU(), Linear(25, 32*num_in_node_features))
        self.conv1 = NNConv(num_in_node_features, 32, nn1, aggr="mean")
        self.batch_norm1 = BatchNorm(32)
        nn2 = Seq(Linear(num_in_edge_features,25), ReLU(), Linear(25,32*64 ))#1024*num_in_node_features))
        self.conv2 = NNConv(32, 64, nn2, aggr="mean")
        self.batch_norm2 = BatchNorm(64)
        self.fc1 = Linear(64, 128)
        self.fc2 = Linear(128, 1)



    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.nn.ELU()(self.conv1(x, edge_index, edge_attr))
        x = F.nn.ELU()(self.batch_norm1(x))
        x = F.nn.ELU()(self.conv2(x, edge_index, edge_attr))
        x = F.nn.ELU()(self.batch_norm2(x))
        x = F.nn.ELU()(self.fc1(x))
        x = F.nn.ELU()(self.fc2(x))
        num_graphs = data.num_graphs if hasattr(data, "num_graphs") else 1
        sum_across_nodes = torch.empty((num_graphs, 1), dtype=torch.float)

        if num_graphs > 1:
            for i in range(num_graphs):
                sum_across_nodes[i] = torch.sum(x[data.ptr[i]:data.ptr[i + 1]])
        else: #because data.ptr doesn't make sense outside of the context of a batch
            sum_across_nodes[0]  = torch.sum(x)
        return sum_across_nodes.flatten()

def eval_model(model, data, epoch=None, deviation_scaler=None):
    model.eval()
    pred = model(data)
    # return (pred-data.y).norm().detach().cpu().numpy(), (data.y-pred).detach().cpu().numpy()
    if epoch is not None and epoch > 300:
        import ipdb;
        ipdb.set_trace()
    if deviation_scaler is not None:
        pred_unscaled = deviation_scaler.inverse_transform(pred.detach().cpu().numpy())
        data_y_unscaled = deviation_scaler.inverse_transform(data.y.detach().cpu().numpy())
    else:
        pred_unscaled = pred.detach().cpu().numpy()
        data_y_unscaled = data.y.detach().cpu().numpy()
    return np.mean(np.abs(pred_unscaled-data_y_unscaled)), data_y_unscaled - pred_unscaled


def run_train(model, data_train_set, data_validate_set, plot=True, deviation_scaler=None, train_cfg=None, wandb_experiment=None):
    train_accuracies, test_accuracies = [], []
    test_diffs = []
    mse = torch.nn.MSELoss()
    loader = DataLoader(data_train_set, batch_size=train_cfg["batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    loss_scale = train_cfg["loss_scale"]
    if wandb_experiment is None:
        wandb_experiment = wandb.init(config=train_cfg, project="mdes", group="lagrassa/model_validation", entity="iam-lab")
    for epoch in range(train_cfg["max_num_epochs"]):
        train_acc_batch = []
        test_acc_batch = []
        test_diff_batch = []
        for data_train in loader:
            model.train()
            optimizer.zero_grad()
            out = model(data_train)
            loss = loss_scale* mse(out, data_train.y)  # + 10*(F.rELU(data_train.y-out)).view([])
            loss.backward()
            optimizer.step()
            train_acc, train_diff = eval_model(model, data_train, deviation_scaler=deviation_scaler,epoch=epoch)
            train_acc_batch.append(train_acc)

        for data_validate in data_validate_set:
            test_acc, test_diff = eval_model(model, data_validate, deviation_scaler=deviation_scaler)
            test_acc_batch.append(test_acc)
            test_diff_batch.append(test_diff)

        train_accuracies.append(np.mean(train_acc_batch))
        test_accuracies.append(np.mean(test_acc_batch))
        wandb_experiment.log({"Train MAE": train_accuracies[-1],
                   "Validation MAE": test_accuracies[-1]})
        test_diffs.append(np.max(test_diff_batch))
        print("Epoch {:03d}, Loss: {:.5f}, Train Acc: {:5f}, Test acc: {:.5f}".format(epoch, loss, train_acc, test_acc))

    if plot:
        plt.plot(train_accuracies, label="Train accuracy")
        plt.plot(test_accuracies, label="Test accuracy")
        plt.plot(test_diffs, label="Test diffs")
        plt.legend()
        plt.show()


def test(model, data_validate_set, deviation_scaler=None):
    test_acc_batch = []
    test_diff_batch = []
    for data_validate in data_validate_set:
        test_acc, test_diff = eval_model(model, data_validate, deviation_scaler=deviation_scaler)
        test_acc_batch.append(test_acc)
        test_diff_batch.append(test_diff)
    test_diff_batch = np.array(test_diff_batch)
    num_over_3 = np.sum(np.array(test_diff_batch) > 0.03)
    num_over_5 = np.sum(np.array(test_diff_batch) > 0.05)
    print("Mean error", np.mean(np.abs(test_diff_batch)))
    print("frac over 3", num_over_3 / len(test_diff_batch))
    print("frac over 5", num_over_5 / len(test_diff_batch))
