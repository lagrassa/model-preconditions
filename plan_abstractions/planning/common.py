from pillar_state import State
import numpy as np


class Action:

    def __init__(self, skill_idx, params, param_type, cost, T_exec, info_plan,
                 T_plan_coeff=0, T_exec_coeff=0, cost_coeff=1):
        self.skill_idx = skill_idx
        self.params = params
        self.cost = cost
        self.T_exec = T_exec
        self.info_plan = info_plan
        self.param_type = param_type

        self.T_plan_coeff = T_plan_coeff
        self.T_exec_coeff = T_exec_coeff
        self.cost_coeff = cost_coeff

    @property
    def T_plan(self):
        return self.info_plan['T_plan']

    @property
    def total_cost(self):
        return self.T_plan_coeff * self.T_plan + \
                self.T_exec_coeff * self.T_exec + \
                self.cost_coeff * self.cost


class PathStep:

    def __init__(self, pillar_state, action_in, depth=-1):
        self.pillar_state = pillar_state
        self.action_in = action_in
        self.depth = depth

    @property
    def exec_data(self):
        return {
            'end_states': self.pillar_state.get_serialized_string(),
            'T_exec': self.action_in.T_exec,
            'terminated': True,
            'costs': self.action_in.cost,
            'info_plan': self.action_in.info_plan
        }

    # for pickling
    def __getstate__(self):
        state_dict = self.__dict__.copy()
        state_dict['pillar_state'] = state_dict['pillar_state'].get_serialized_string()
        return state_dict

    def __setstate__(self, state_dict):
        state_dict['pillar_state'] = State.create_from_serialized_string(state_dict['pillar_state'])
        self.__dict__.update(state_dict)


class Node:
    """Class to represent an planning node."""

    def __init__(self, pillar_state, skill_idxs, parent=None, action_in=None,
                 depth=0, debug_id=0):
        assert len(skill_idxs) > 0

        if isinstance(pillar_state, bytes):
            pillar_state = State.create_from_serialized_string(pillar_state)

        self._pillar_state = pillar_state
        self._parent = parent
        self._action_in = action_in
        self._skill_idxs = skill_idxs[::] # copy the list
        self.incompletely_expanded_action_params = [ [] for _ in range(len(skill_idxs))]
        self.incompletely_expanded_action_param_types = [ [] for _ in range(len(skill_idxs))]
        self._has_been_expanded = False
        self._depth = depth
        self._children = []

        # For ease of debugging
        self._debug_id = debug_id
        # Only used in root node. Can actually store recursively, but not needed for now.
        self.num_nodes = 1

        self.n_visits = 0
        self.h = 1e6 #inf leads to issues  # need to update outside
        self.g = None  # need to update outside

    @property
    def has_been_expanded(self):
        return self._has_been_expanded
    
    def set_has_been_expanded_to_true(self):
        self._has_been_expanded = True

    @property
    def pillar_state(self):
        return self._pillar_state

    @property
    def children(self):
        return self._children

    @property
    def action_in(self):
        return self._action_in

    @action_in.setter
    def action_in(self, action):
        self._action_in = action
    
    @property
    def depth(self):
        return self._depth
    
    @property
    def debug_id(self):
        return self._debug_id

    @property
    def parent(self):
        return self._parent

    @property
    def skill_idxs(self):
        return self._skill_idxs[::]

    def set_parent(self, new_parent):
        self._parent = new_parent

    def add_children(self, children):
        self._children.extend(children)

    def is_leaf_node(self):
        return not any(self.children)

    def find_path_from_root(self):
        '''Returns path from root with root as the first element.'''
        path = []
        node = self
        while node is not None:
            path.append(PathStep(node.pillar_state, node.action_in, node.depth))
            node = node.parent
        return path[::-1]

    # for pickling
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_pillar_state'] = state['_pillar_state'].get_serialized_string()
        return state

    def __setstate__(self, state):
        state['_pillar_state'] = State.create_from_serialized_string(state['_pillar_state'])
        self.__dict__.update(state)
