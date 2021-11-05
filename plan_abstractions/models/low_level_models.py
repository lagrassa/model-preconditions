from abc import ABC

import quaternion

from isaacgym_utils.math_utils import quat_to_np
import torch
from torch_geometric.data import Data
import numpy as np

from plan_abstractions.learning import ASSET_NAME_TO_ID, copy_graph, update_edge_features


class LowLevelModel(ABC):
    def __init__(self, cfg):
        self._cfg = cfg

    def predict_vector(self, state, action):
        pass

    def predict_graph(self, state, action):
        pass

    def __call__(self, state, action, use_graph=False):
        if isinstance(state, Data):
            return self.predict_graph(state, action)
        elif isinstance(state, np.ndarray):
            return self.predict_vector(state, action)


class FreeSpaceMoveModel(LowLevelModel):

    def predict_vector(self, state, action):
        copied_state = state.copy()
        copied_state[:3] += action[:3]
        delta_quat = quaternion.from_float_array(action[3:7])
        current_quat = quaternion.from_float_array(copied_state[3:7])
        final_quat = delta_quat * current_quat
        copied_state[3:7] = quat_to_np(final_quat, format="wxyz")
        return copied_state

    def predict_graph(self, state, action):
        node_features_np = state.x.numpy().copy()
        for i, node_feature in enumerate(node_features_np):
            if node_feature[3] == ASSET_NAME_TO_ID["finger_right"]:
                node_features_np[i, :3] += action[:3] #TODO update this when we add rotation
        new_state = copy_graph(state)
        new_state.x = torch.tensor(node_features_np, dtype=torch.float)
        contact_distance = state.contact_distance
        num_edges = state.edge_attr.shape[0]
        dim_edge_feature = state.edge_attr.shape[1]
        edge_attr, edge_index = update_edge_features(dim_edge_feature, contact_distance, node_features_np, num_edges)
        new_state.edge_attr = edge_attr
        new_state.edge_index = edge_index
        return new_state


class FixedRelativeTransformsOnContactModel(LowLevelModel):

    def predict_vector(self, state, action):
        return state

    def predict_graph(self, state, action):
        id_idx = 3
        contact_idx = 1
        node_features_np = state.x.numpy().copy()
        edge_attr_np = state.edge_attr.numpy().copy()
        edge_index_np = state.edge_index.numpy().copy()
        #Determine which IDs are in contact with the gripper.
        gripper_node_idxs = np.arange(len(node_features_np))[node_features_np[:,id_idx] == 0]
        nodes_idxs_in_contact_with_gripper = []
        for node_idx, node_feature in enumerate(node_features_np):
            if node_feature[id_idx] == ASSET_NAME_TO_ID["finger_right"]:
                continue
            for edge_idx in range(edge_index_np.shape[1]):
                edge_pair = edge_index_np[:, edge_idx]
                if ASSET_NAME_TO_ID["finger_right"] in edge_pair and node_idx in edge_pair:
                    if not edge_attr_np[edge_idx,contact_idx]:
                        continue #not in contact
                    nodes_idxs_in_contact_with_gripper.append(node_idx)
        nodes_idxs_in_contact_with_gripper = np.unique(nodes_idxs_in_contact_with_gripper)
        for i, node_feature in enumerate(node_features_np):
            if node_feature[3] == ASSET_NAME_TO_ID["finger_right"] or i in nodes_idxs_in_contact_with_gripper:
                node_features_np[i, :3] += action[:3] #TODO update this when we add rotation

        new_state = copy_graph(state)
        new_state.x = torch.tensor(node_features_np, dtype=torch.float)
        contact_distance = state.contact_distance
        num_edges = state.edge_attr.shape[0]
        dim_edge_feature = state.edge_attr.shape[1]
        edge_attr, edge_index = update_edge_features(dim_edge_feature, contact_distance, node_features_np, num_edges)
        new_state.edge_attr = edge_attr
        new_state.edge_index = edge_index
        return new_state
