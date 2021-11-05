import torch
from itertools import combinations
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
try:
    from torch_geometric.data import InMemoryDataset, Data
except ImportError:
    class InMemoryDataset:
        pass
from plan_abstractions.learning.data_utils import get_min_dist
from pickle import dump, load
import os


class DeviationPredDataset(Dataset):
    def __init__(self, X, y, scale_X=False):
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        if not torch.is_tensor(X):
            if scale_X:
                assert False
                X = StandardScaler().fit_transform(X)
            self.X = torch.from_numpy(X)
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)  # .flatten().long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



class GraphDeviationDataset(InMemoryDataset):
    def __init__(self, root, cfg, transform=None,
                 pre_transform=None,
                 N=10,
                 node_features_scaler=None,
                 edge_features_scaler=None,
                 deviation_scaler=None,
                 fit_scaler=True):
        self._N = N
        self._data_root = root
        self.node_features_scaler = node_features_scaler
        self.edge_features_scaler = edge_features_scaler
        self.deviation_scaler = deviation_scaler
        self._node_scaler_fn = "node_scaler.pkl"
        self._edge_scaler_fn = "edge_scaler.pkl"
        self._deviation_scaler_fn = "deviation_scaler.pkl"
        self._scaler_fns = [self._node_scaler_fn, self._edge_scaler_fn, self._deviation_scaler_fn]
        self._scalers = [self.node_features_scaler, self.edge_features_scaler, self.deviation_scaler]
        self._fit_scaler = fit_scaler
        self._data_list = self._generate_data_list()
        super(GraphDeviationDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.deviation_scaler = np.load(os.path.join(root, self._deviation_scaler_fn), allow_pickle=True)

    @property
    def raw_file_names(self):
        return []  # 'some_file_1', 'some_file_2', ...]

    @property
    def data_list(self):
        return self._data_list

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        node_features = np.vstack([data.x.detach().cpu().numpy() for data in self.data_list])
        edge_features = np.vstack([data.edge_attr.detach().cpu().numpy() for data in self.data_list])
        deviations = np.vstack([data.y.detach().cpu().numpy() for data in self.data_list])
        # normalize data:
        if self._fit_scaler:
            self.node_features_scaler = StandardScaler().fit(node_features)
            self.edge_features_scaler = StandardScaler().fit(edge_features)
            self.deviation_scaler = StandardScaler().fit(deviations)
        self._scalers = [self.node_features_scaler, self.edge_features_scaler, self.deviation_scaler]

        node_features_transformed = self.node_features_scaler.transform(node_features)
        edge_features_transformed = self.edge_features_scaler.transform(edge_features)
        deviations_transformed = self.deviation_scaler.transform(deviations)

        # set back to transformed values
        node_idx = 0
        edge_idx = 0
        for i, data_pt in enumerate(self.data_list):
            num_nodes = data_pt.x.shape[0]
            num_edges = data_pt.edge_attr.shape[0]
            data_pt.x = torch.tensor(node_features_transformed[node_idx: node_idx + num_nodes], dtype=torch.float)
            data_pt.edge_attr = torch.tensor(edge_features_transformed[edge_idx: edge_idx + num_edges],
                                             dtype=torch.float)
            data_pt.y = torch.tensor(deviations_transformed[i], dtype=torch.float)
            node_idx += num_nodes
            edge_idx += num_edges

        data, slices = self.collate(self.data_list)

        torch.save((data, slices), self.processed_paths[0])
        if self._fit_scaler:
            for fn, scaler in zip(self._scaler_fns, self._scalers):
                with open(os.path.join(self._data_root, fn), "wb") as f:
                    dump(scaler, f)


class RealGraphDeviationDataset(GraphDeviationDataset):
    def __init__(self, root, cfg, transform=None,
                 pre_transform=None,
                 N=10,
                 node_features_scaler=None,
                 edge_features_scaler=None,
                 deviation_scaler=None,
                 fit_scaler=True,
                 states_and_parameters=None,
                 deviations=None
                 ):
        assert states_and_parameters is not None
        assert deviations is not None
        self._state_and_parameters = states_and_parameters
        self._deviations = deviations
        super(RealGraphDeviationDataset, self).__init__(root, cfg, transform=transform,
                                                        pre_transform=pre_transform, N=N,
                                                        node_features_scaler=node_features_scaler,
                                                        edge_features_scaler=edge_features_scaler,
                                                        deviation_scaler=deviation_scaler,
                                                        fit_scaler=fit_scaler)

    def _generate_data_list(self):
        for state_and_parameter_graph, deviation in zip(self._state_and_parameters, self._deviations):
            state_and_parameter_graph.y = torch.tensor([deviation], dtype=torch.float)
        return self._state_and_parameters

    def process(self):  # Needs to be here due to InMemoryDataset assumptions
        return super(RealGraphDeviationDataset, self).process()


class MockDeviationDataset(GraphDeviationDataset):
    def __init__(self, root, cfg, transform=None,
                 pre_transform=None,
                 N=10,
                 node_features_scaler=None,
                 edge_features_scaler=None,
                 deviation_scaler=None,
                 fit_scaler=False,
                 num_rod_possibilities=None):
        self.num_rod_possibilities = num_rod_possibilities
        td = ToyDataset("tmp")
        super(MockDeviationDataset, self).__init__(root, cfg, transform=transform,
                                                   pre_transform=pre_transform, N=N,
                                                   node_features_scaler=node_features_scaler,
                                                   edge_features_scaler=edge_features_scaler,
                                                   deviation_scaler=deviation_scaler,
                                                   fit_scaler=fit_scaler)

    def _generate_data_list(self):
        return [self.generate_data() for _ in range(self._N)]

    def process(self):  # Needs to be here due to InMemoryDataset assumptions
        return super(MockDeviationDataset, self).process()

    def generate_data(self):
        rod_pos = np.random.uniform(low=-1, high=1, size=(3,))
        num_other_rods = self.num_rod_possibilities[np.random.randint(len(self.num_rod_possibilities))]
        if np.random.randint(2):
            gripper_pos = rod_pos.copy()
            gripper_pos[2] += 0.01
            gt_deviation = 0
        else:
            gripper_pos = rod_pos.copy()
            gripper_pos[0] += 0.08
            gripper_pos[1] += 0.00
            gripper_pos[2] += 0.01
            gt_deviation = 0.1
        # random other rods

        distance = np.linalg.norm(gripper_pos - rod_pos)
        # node_features = np.vstack([np.hstack([0, rod_pos]), np.hstack([1, gripper_pos])])
        node_features = []
        gripper_id = 1
        rod_id = 0
        action = 0.01 * np.random.random(size=(3,))  # should learn action doesn't matter for this task
        node_features.append(np.hstack([rod_id, rod_pos, action]))
        node_features.append(np.hstack([gripper_id, gripper_pos, action]))
        # node_features = np.vstack([np.hstack([0, rod_pos]), np.hstack([1, gripper_pos])])
        for i in range(num_other_rods):
            other_rod_pos = np.random.uniform(low=-1, high=1, size=(3,))
            node_features.append(np.hstack([rod_id, other_rod_pos, action]))
        node_features = np.vstack(node_features)
        grasped_rod_feature = node_features[0].copy()
        np.random.shuffle(node_features)
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor([gt_deviation], dtype=torch.float)
        edge_index_np = []
        edge_attr_np = []
        all_ids = range(len(node_features))
        for id_0, id_1 in combinations(all_ids, 2):
            if node_features[id_0][0] == gripper_id and np.all(node_features[id_1] == grasped_rod_feature):
                in_contact = True
            elif node_features[id_1][0] == gripper_id and np.all(node_features[id_0] == grasped_rod_feature):
                in_contact = True
            else:
                in_contact = False

            edge_index_np.append([id_0, id_1])
            distance = np.linalg.norm(node_features[id_0][1:] - node_features[id_1][1:])
            edge_attr_np.append([distance, in_contact])
        edge_index_np = np.vstack(edge_index_np)
        edge_attr_np = np.vstack(edge_attr_np)
        edge_index = torch.tensor(edge_index_np.T, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr_np, dtype=torch.float)
        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
        return data


class ToyDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ToyDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []  # 'some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        x = torch.tensor([[-1], [0], [1], [-1]], dtype=torch.float)
        y = torch.tensor([1, 0, 1, 1], dtype=torch.long)
        edge_index = torch.tensor([[0, 1, 1, 2, 1, 3], [1, 0, 2, 1, 3, 1]], dtype=torch.long)
        edge_attr = torch.tensor([[0], [-0.1], [0.1], [0], [-0.1], [0]],
                                 dtype=torch.float)  # num edges, num edge features
        train_mask = torch.tensor([True, True, True, False], dtype=torch.bool)
        test_mask = torch.tensor([True, True, True, True], dtype=torch.bool)
        data_list = [
            Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, train_mask=train_mask, test_mask=test_mask)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
