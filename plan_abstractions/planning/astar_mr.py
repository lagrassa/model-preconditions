import logging
from time import time
from pickle import load

import numpy as np

from .common import Action, Node
from .utils import remove_from_queues, add_to_queues
from ..utils.utils import to_str, get_pose_pillar_state, combine_effects
from .planner import Planner

logger = logging.getLogger(__name__)
logger.setLevel('INFO')


class MRAStarStats:
    node_expansions = 0

    def reset(self):
        self.node_expansions = 0


class MRAStar(Planner):

    ############
    # Search-related methods
    ############
    def __init__(self, task, env, skills, cfg, root_node=None, root_dir=None):
        super().__init__(task, env, skills, cfg, root_node=root_node, root_dir = root_dir)
        self._eps = cfg["eps"]
        self.wm = cfg["wm"]
        self.num_models = cfg["num_models"]
        self.ablation_type = cfg.get("ablation_type", None)
        assert self.ablation_type in [None, "random_model", "anchor_only", "all_models"]
        self._use_all_models = cfg.get("use_all_models", False)
        if self._use_all_models:
            assert self.ablation_type is None #doens't make sense to run ablations with planning w/ all models


    def _get_node_to_expand(self, max_search_depth=float('inf')):
        logger.debug("  Selecting")
        anchor_fs = [self.wm[0] * self._eps*node.h + node.g for node in self._open_anchor]# if
              #node._depth < max_search_depth]  # TODO add real cost
        fs_per_cheap_queue = []
        for i, cheap_queue in enumerate(self._open_cheaps):
            cheap_fs = [self.wm[i+1]*(self._eps*node.h + node.g) for node in cheap_queue]# if
                     #node._depth < max_search_depth]  # TODO add real cost
            fs_per_cheap_queue.append(cheap_fs)

        best_idx_anchor = np.argmin(anchor_fs) if len(anchor_fs) else None
        if sum([len(fs) for fs in fs_per_cheap_queue]) == 0:
            best_idx_cheap = None
        else:
            best_cheap_queue_idx = np.argmin([min(queue) for queue in fs_per_cheap_queue if len(queue)])
            best_idx_cheap = np.argmin(fs_per_cheap_queue[best_cheap_queue_idx])
        if best_idx_anchor is None and best_idx_cheap is None:
            logger.error("Ran out of nodes to expand")
            return None, None
        #logger.debug(f"Min cost {fs[best_idx]}")
        if self.ablation_type in ["anchor_only", "random_model", "all_models"] or best_idx_cheap is None: #only consider anchor
            return self._open_anchor[best_idx_anchor], 0
        if best_idx_anchor is None: #only consider cheap
            return self._open_cheaps[best_cheap_queue_idx][best_idx_cheap], best_cheap_queue_idx+1
        else: #compare anchor and cheap
            if anchor_fs[best_idx_anchor] < fs_per_cheap_queue[best_cheap_queue_idx][best_idx_cheap]: #anchor is better: use it
                return self._open_anchor[best_idx_anchor], 0
            else:
                return self._open_cheaps[best_cheap_queue_idx][best_idx_cheap], best_cheap_queue_idx+1

    def _partial_expand(self, node, graph_idx):
        print("Partial expansion")
        assert not node.has_been_expanded
        #DOn't need to deal with nodes with actions
        children = []
        using_partially_expanded_params = False
        for skill_idx in node.skill_idxs:
            skill = self._skills[skill_idx]
            completed_incompletes = None
            if len(node.incompletely_expanded_action_params[skill_idx]):
                param_types = node.incompletely_expanded_action_param_types[skill_idx]
                params = node.incompletely_expanded_action_params[skill_idx]
                completed_incompletes = []
                using_partially_expanded_params = True
            else:
                param_types, params = self.get_new_params(node, skill, skill_idx)
            if len(params) > 0:
                effects, model_idxs = self._compute_effects(skill_idx, node, params, param_types, model_idx_to_eval = graph_idx)

            for i, param in enumerate(params):
                if model_idxs[i] != graph_idx and not using_partially_expanded_params: #need to put off these parameters for later
                    node.incompletely_expanded_action_params[skill_idx].append(params[i])
                    node.incompletely_expanded_action_param_types[skill_idx].append(param_types[i])
                    continue
                if effects["end_states"][i] == None:
                    print("No valid models for this set of states and params")
                    continue
                if completed_incompletes is not None:
                    completed_incompletes.append(i)
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
                node.add_children(children)
                if completed_incompletes is not None:
                    node.incompletely_expanded_action_params[skill_idx] = [node.incompletely_expanded_action_params[skill_idx][idx] for idx in range(len(node.incompletely_expanded_action_params[skill_idx])) if idx not in completed_incompletes]
                    node.incompletely_expanded_action_param_types[skill_idx] = [node.incompletely_expanded_action_param_types[skill_idx][idx] for idx in
                                                            range(len(node.incompletely_expanded_action_param_types[skill_idx])) if
                                                            idx not in completed_incompletes]
                self._root_node.num_nodes += 1
        return children

    def _expand(self, node):
        if self._similar_to_node_in_queue(node, self._closed_anchor):
            return []
        assert not node.has_been_expanded
        print("Full expansion")
        logger.debug("Full Expanding")
        children = []
        for skill_idx in node.skill_idxs:
            skill = self._skills[skill_idx]
            if len(node.incompletely_expanded_action_params[skill_idx]):
                param_types = node.incompletely_expanded_action_param_types[skill_idx]
                params = node.incompletely_expanded_action_params[skill_idx]
            else:
                param_types, params = self.get_new_params(node, skill, skill_idx)

            if len(params) > 0:
                effects, _ = self._compute_effects(skill_idx, node, params, param_types)

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
                    depth=node._depth + 1,
                    debug_id=self._root_node.num_nodes + 1
                )
                children.append(child)
                node.add_children(children)
                self._root_node.num_nodes += 1

        node.set_has_been_expanded_to_true()
        return children


    def get_new_params(self, node, skill, skill_idx):
        default_params = self._n_params_per_skill_dict['default']
        num_params = self._n_params_per_skill_dict.get(skill.__class__.__name__, default_params)
        # Remember, you cannot get a node with the parameters already generated from the cheap queue.
        params_gen = self._task.generate_parameters(skill, self._env, node.pillar_state,
                                                    num_parameters=num_params,
                                                    return_param_types=True,
                                                    check_valid_goal_state=True,
                                                    valid_success_per_param_type=self._success_per_param_types[
                                                        skill_idx]
                                                    )
        params, param_types = next(params_gen)
        return param_types, params
    def _add_model_evaluation_stats_to_dict(self, skill_idx, model_idx_per_param, model_idx_to_eval=None):
        for model_idx in range(self.num_models):
            if model_idx_to_eval is not None and model_idx != model_idx_to_eval:
                continue
            self.model_type_per_skill_idx[skill_idx][model_idx] += np.sum(model_idx_per_param == model_idx)

    def _compute_effects(self, skill_idx, node, params, param_types, model_idx_to_eval=None):
        start_states = [node.pillar_state] * len(params)
        skill = self._skills[skill_idx]
        hardcode = False
        if hardcode:
            print("Warning: hardcoding parameters for MR AStar!!! For debugging only")
        model_idx_per_param = []
        if len(params) == 1:
            print("Smaller num models for some reason")
        if hardcode:
            model_idx_per_param = np.array([1] * len(params))
        elif self.ablation_type == "random_model":
            model_idx_per_param = np.random.randint(low=0, high=self.num_models, size=(len(params)))
        else:
            for param_idx, param in enumerate(params):
                model_idxs_in_precond = []
                for model_idx, model in enumerate(skill.high_level_models):
                    if model_idx_to_eval is not None and model_idx != model_idx_to_eval:
                        continue
                    if model.model_precondition_satisfied(start_states[param_idx], param):
                        model_idxs_in_precond.append(model_idx)
                if not len(model_idxs_in_precond):
                    model_idx_per_param.append(None)
                else:
                    model_idx_per_param.append(max(model_idxs_in_precond))
            model_idx_per_param = np.array(model_idx_per_param)
        start_time = time()

        if self.ablation_type == "all_models":
            effects_list = []
            for model_idx in range(self.num_models):
                model_idx_per_param = np.ones(len(params),) * model_idx
                self._add_model_evaluation_stats_to_dict(skill_idx, model_idx_per_param, model_idx_to_eval=model_idx_to_eval)
                effects_per_model = skill.multiple_model_effects(self._env, start_states, params, self._cfg["T_plan_max"],
                                                       self._cfg["T_exec_max"], model_idx_per_param,
                                                       self._pb_env, model_idx_to_eval=model_idx_to_eval)
                effects_list.append(effects_per_model)
            effects = combine_effects(effects_list)
        else:
            self._add_model_evaluation_stats_to_dict(skill_idx, model_idx_per_param,
                                                     model_idx_to_eval=model_idx_to_eval)
            effects = skill.multiple_model_effects(self._env, start_states, params, self._cfg["T_plan_max"],
                                               self._cfg["T_exec_max"], np.array(model_idx_per_param), self._pb_env, model_idx_to_eval=model_idx_to_eval)
        if effects == -1:
            return []
        self.num_model_evals += len(model_idx_per_param)
        time_elapsed = time() - start_time
        self._model_evaluation_times.append(time_elapsed)
        assert len(model_idx_per_param) == len(params)
        return effects, model_idx_per_param

    def _search(self, log_every=10, timeout=1, max_expansions=10000,
                max_search_depth=float('inf')):
        self._root_node.g = 0
        self._root_node.h = self._task.evaluate(self._root_node.pillar_state)
        self._open_anchor = [self._root_node]
        self._open_cheaps = [[self._root_node] for _ in range(self.num_models-1)]
        stats = MRAStarStats()
        start_time = time()
        iters = 0
        success = False
        self._closed_anchor = []
        self._closed_cheaps = [[] for _ in range(self.num_models-1)]
        while len(self._open_anchor) + sum([len(queue) for queue in self._open_cheaps]) and (time() - start_time) < timeout:
            node, graph_idx = self._get_node_to_expand(max_search_depth=max_search_depth)
            #print(f"Graph expanded from: {graph_idx}")
            #print(f"len open lists: {[len(queue) for queue in self._open_cheaps]}")
            #print(f"len closed lists: {[len(queue) for queue in self._closed_cheaps]}")
            if node is None:
                return None, []

            if not node:
                break

            if self._task.is_goal_state(node.pillar_state):
                goal = node
                success = True
                logger.info("  Plan found.")
                break


            # batch expansion
            if graph_idx == 0: #normal expansion
                if node in self._open_anchor: #need to check because might have removed due to be similar
                    self._open_anchor.remove(node)
                children = self._expand(node)
                if node in self._open_cheaps:
                    self._open_cheap.remove(node)
                self._closed_anchor.append(node)
                remove_from_queues(node, [self._open_anchor,] + self._open_cheaps)
                add_to_queues(node, [self._closed_anchor] + self._closed_cheaps)
            else:
                children = self._partial_expand(node, graph_idx)
                if node in self._open_cheaps[graph_idx-1]:
                    self._open_cheaps[graph_idx-1].remove(node)
                self._closed_cheaps[graph_idx-1].append(node)
                if sum([len(param_list) for param_list in node.incompletely_expanded_action_params]) == 0: #No other actions to consider
                    remove_from_queues(node, [self._open_anchor,]+self._open_cheaps)
                    add_to_queues(node, [self._closed_anchor,]+self._closed_cheaps)
                else:
                    assert node in self._open_anchor

            if len(children) > 0:
                stats.node_expansions += 1
                # child is a leaf node
                for child in children:
                    child.h = self._task.evaluate(child.pillar_state)
                    potential_g = node.g + child.action_in.total_cost #self._task.compute_edge_cost(node.pillar_state, child.pillar_state)
                    if child.g is None or potential_g < child.g: #unassigned (expand does not set this) or reconnect
                        # found better path
                        if child.g is not None:
                            pass
                        child.set_parent(node)
                        child.g = potential_g
                        if self._check_similar:
                            for open_queue, closed_list in zip([self._open_anchor,] + self._open_cheaps, [self._closed_anchor]+ self._closed_cheaps):
                                if child not in closed_list:
                                    if self._similar_to_node_in_queue(child, closed_list):
                                        continue
                                    open_queue.append(child)
                        else:
                            add_to_queues(child, [self._open_anchor,] + self._open_cheaps)

            iters += 1


            if iters % log_every == 0:
                logger.info(f"Iteration Count: {iters} | Expanded {stats.node_expansions} nodes")

            if stats.node_expansions >= max_expansions:
                logger.info('Expanded max nodes without finding a plan')
                break

        q_root = [child.h for child in self._root_node.children]
        logger.info("  Q(root) values: %s" % to_str(q_root))
        print("Timed out?", time() - start_time > timeout)
        print("Len open anchor queue", len(self._open_anchor))
        print("Len cheap open queue", [len(queue) for queue in self._open_cheaps])
        logger.info("Statistics:")
        logger.info("----------")
        logger.info("  Total Node expansions: %d" % stats.node_expansions)

        if success:
            return goal, goal.find_path_from_root()
        return None,[]
   
    def _similar_to_node_in_queue(self, node, queue):
        for other_node in queue:
            if self._task.states_similar(node.pillar_state, other_node.pillar_state):
                return True
        return False

    @staticmethod
    def load(filename, task, env, skills, cfg):
        with open(filename, 'rb') as f:
            root_node = load(f)
        return MRAStar(task, env, skills, cfg, root_node=root_node)
