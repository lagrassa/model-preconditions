import logging
import numpy as np
from time import time
from pickle import load, dump

from .common import Action, Node
from abc import ABC, abstractmethod

from ..envs.pb_franka_env import FrankaKinematicsWorld

logger = logging.getLogger(__name__)


class Planner(ABC):
    """Implements the UCT algorithm for Monte Carlo Tree Search."""

    def __init__(self, task, env, skills, cfg, root_node=None, root_dir=""):
        self._task = task
        self._model_evaluation_times =[]
        self._skills = skills
        self._skill_idxs = [i for i in range(len(skills))]
        self._env = env
        self._cfg = cfg
        self.num_model_evals = 0
        self.num_models = cfg.get("num_models", 1)
        self.model_type_per_skill_idx = {}
        for skill_idx in self._skill_idxs:
            self.model_type_per_skill_idx[skill_idx] = [0 for _ in range(self.num_models)]

        if 'num_params_per_skill' in cfg:
            self._n_params_per_skill_dict = cfg['num_params_per_skill']
        else:
            self._n_params_per_skill_dict = {'default': cfg["n_params_per_skill"] }

        self._success_per_param_types = [{param_type: [] for param_type in skill.param_types} for skill in skills]

        self._set_root_node(root_node)
        if 'check_similar' in self._cfg.keys():
            self._check_similar = self._cfg['check_similar']
        else:
            self._check_similar = True #algorithm can do whatever it wants with this info to merge similar states.
        #if cfg.get("use_multiple_models", False):
        #    self._pb_env = FrankaKinematicsWorld(root_dir = root_dir, visualize=False)
        self._pb_env=None
    
    @property
    def cfg(self):
        return self._cfg

    @property
    def n_skills(self):
        return len(self._skills)
    
    @property
    def skills(self):
        return self._skills

    def _set_root_node(self, root_node):
        # TODO start state validity checking
        self._root_node = root_node
    
    @property
    def root_node(self):
        return self._root_node

    @property
    def root_node(self):
        return self._root_node

    ############
    # Search-related methods
    ############
    @abstractmethod
    def _get_node_to_expand(self):
        pass

    def _compute_effects(self, skill_idx, node, params, param_types):
        start_states = [node.pillar_state] * len(params)
        skill = self._skills[skill_idx]

        if self._cfg["use_gt_effects"]:
            assert params.shape[0] <= self._env.n_envs
            effects = skill.gt_effects(
                self._env,
                start_states,
                params,
                self._cfg["T_plan_max"],
                self._cfg["T_exec_max"],
            )
        elif self._cfg.get("use_multiple_models", False):
            #mix based on deviation models
            hardcode = True
            model_idx_per_param = []
            if len(params) == 1:
                print("Smaller num models for some reason")
            if hardcode:
                model_idx_per_param = np.array([1]*len(params))
                #model_idx_per_param[-18:] = 1
            else:
                for param_idx, param in enumerate(params):
                    found_model = False
                    for model_idx, model in enumerate(skill.models):
                        if model.model_precondition_satisfied(start_states[param_idx], param):
                            model_idx_per_param.append(model_idx)
                            found_model = True
                            break
                    if not found_model:
                        model_idx_per_param.append(None)

            start_time = time()
            effects = skill.multiple_model_effects(self._env, start_states, params, self._cfg["T_plan_max"], self._cfg["T_exec_max"], model_idx_per_param, self._pb_env)
            if effects == -1:
                return []
            self.num_model_evals += len(model_idx_per_param)
            time_elapsed = time() - start_time
            self._model_evaluation_times.append(time_elapsed)
            #print("Model idxs", model_idx_per_param)
            #print("Time elapsed per model eval", np.mean(self._model_evaluation_times))

        else:
            effects = skill.effects_batch(start_states, params)
            # The [0]'s are b/c we do "deterministic" planning -
            # the planner does not take SEM uncertainty into account
            for i, param in enumerate(params):
                effects["end_states"][i] = effects["end_states"][i][0]
                effects["costs"][i] = effects["costs"][i][0]
                effects["T_exec"][i] = effects["T_exec"][i][0]
                effects["info_plan"][i]["T_plan"] = effects["info_plan"][i]["T_plan"][0]

        children = []
        for i, param in enumerate(params):
            if effects["end_states"][i] == None:
                print("No valid models for this set of states and params")
                continue
            action = Action(skill_idx, param,
                            param_types[i],
                            effects["costs"][i],
                            effects["T_exec"][i],
                            effects["info_plan"][i]
                            )
            child = Node(
                effects["end_states"][i],
                self._skill_idxs,
                parent=node,
                action_in=action,
                depth=node._depth+1,
                debug_id=self._root_node.num_nodes + 1
            )
            children.append(child)
            self._root_node.num_nodes += 1

        return children

    def _expand(self, node):
        assert not node.has_been_expanded
        logger.debug("  Expanding")

        for skill_idx in node.skill_idxs:
            skill = self._skills[skill_idx]

            default_params = self._n_params_per_skill_dict['default']
            num_params = self._n_params_per_skill_dict.get(skill.__class__.__name__, default_params)

            params_gen = self._task.generate_parameters(skill, self._env, node.pillar_state,
                                                        num_parameters=num_params,
                                                        return_param_types=True,
                                                        check_valid_goal_state=True,
                                                        valid_success_per_param_type=self._success_per_param_types[skill_idx]
                                                        )
            params, param_types = next(params_gen)
            if len(params) > 0:
                children = self._compute_effects(skill_idx, node, params, param_types)
                node.add_children(children)

        node.set_has_been_expanded_to_true()
        return node.children

    @abstractmethod
    def _search(self, log_every=10, timeout=1, max_expansions=10000,
                max_search_depth=float('inf')):
        pass

    def search(self, init_pillar_state=None, log_every=10, timeout=1,
               max_expansions=10000, max_search_depth=float('inf')):
        logger.debug("Search")
        logger.debug("-----------")
        start_time = time()
        if init_pillar_state is not None:
            root_node = Node(init_pillar_state, self._skill_idxs)
            self._set_root_node(root_node)

        if self._root_node is None:
            raise ValueError('Must set root node manually or pass in init_pillar_state')

        plan = self._search(log_every=log_every, timeout=timeout, max_expansions=max_expansions, max_search_depth=max_search_depth)
        end_time = time()
        elapsed_time = end_time - start_time
        logger.info(f"Planner took {elapsed_time:.2f}s")
        return plan

    def traverse_tree(self, do_print=False):
        node = self._root_node
        queue = [node]
        nodes = []
        i = 0
        if do_print:
            print("Printing the Tree")
            print("------------------")
        while queue:
            node = queue.pop(0)

            cost_to_go = self._task.evaluate(node.pillar_state)
            if do_print:
                print(f"node_id: {i}, cost-to-go: {cost_to_go}")

            nodes.append(node)
            for child in node.children:
                if child:
                    queue.append(child)

            i += 1
        return nodes

    def find_closest_to_goal_leaf_node(self):
        return self._find_closest_to_goal_leaf_node(self._root_node)

    def _find_closest_to_goal_leaf_node(self, node):
        if node is None:
            return None

        if node.is_leaf_node():
            return {
                'node': node,
                'dist': self._task.distance_to_goal_state(node.pillar_state)
            }

        best_leaf_node = None
        for child_node in node.children:
            if child_node is not None:
                subtree_leaf_node = self._find_closest_to_goal_leaf_node(child_node)
                if subtree_leaf_node is not None:
                    if best_leaf_node is None:
                        best_leaf_node = subtree_leaf_node
                    elif best_leaf_node['dist'] > subtree_leaf_node['dist']:
                        best_leaf_node = subtree_leaf_node
        return best_leaf_node
    
    def find_node_with_debug_id(self, target_node_debug_id):
        if not hasattr(self._root_node, 'debug_id'):
            raise ValueError(f"Nodes have no debug_id, cannot find node with debug id: {target_node_debug_id}")
            return None
        return self._find_node_with_debug_id(self._root_node, target_node_debug_id) 
    
    def _find_node_with_debug_id(self, node, target_debug_id):
        if node is None:
            return None
        elif node.debug_id == target_debug_id:
            return node
        
        for child_node in node.children:
            if child_node is not None:
                target_node = self._find_node_with_debug_id(child_node, target_debug_id)
                if target_node is not None:
                    return target_node
        return None

    def find_all_leaf_nodes(self):
        queue = []
        self._find_all_leaf_nodes(self._root_node, queue)
        return queue

    def _find_all_leaf_nodes(self, node, queue):
        if node is None:
            return

        if node.is_leaf_node():
            queue.append(node)
            return

        for child_node in node.children:
            if child_node is not None:
                self._find_all_leaf_nodes(child_node, queue)

    @staticmethod
    def save_plan(plan, filename):
        with open(filename, "wb") as f:
            dump(plan, f)

    @staticmethod
    def load_plan(filename):
        with open(filename, 'rb') as f:
            return load(f)

    def save(self, filename):
        with open(filename, 'wb') as f:
            dump(self._root_node, f)

    @staticmethod
    @abstractmethod
    def load(filename, task, env, skills, cfg):
        pass
