import logging
from time import time
from pickle import load

import numpy as np

from ..utils.utils import to_str
from .planner import Planner

logger = logging.getLogger(__name__)
logger.setLevel('INFO')


class AStarStats:
    node_expansions = 0

    def reset(self):
        self.node_expansions = 0


class AStar(Planner):

    ############
    # Search-related methods
    ############
    def __init__(self, task, env, skills, cfg, root_node=None, root_dir=None):
        super().__init__(task, env, skills, cfg, root_node=root_node, root_dir = root_dir)
        self._eps = cfg["eps"]

    def _get_node_to_expand(self, max_search_depth=float('inf')):
        logger.debug("  Selecting")
        fs = [self._eps*node.h + node.g for node in self._open if
              node._depth < max_search_depth]  # TODO add real cost
        if fs:
            best_idx = np.argmin(fs)
            logger.debug(f"Min cost {fs[best_idx]}")
            return self._open[best_idx]
        else:
            return None

    def _search(self, log_every=10, timeout=1, max_expansions=10000,
                max_search_depth=float('inf')):
        self._root_node.g = 0
        self._open = [self._root_node]  # TODO sort
        stats = AStarStats()
        start_time = time()
        iters = 0
        success = False
        self._closed = []
        while len(self._open) and (time() - start_time) < timeout:
            node = self._get_node_to_expand(max_search_depth=max_search_depth)
            if not node:
                break

            if self._task.is_goal_state(node.pillar_state):
                goal = node
                success = True
                logger.info("  Plan found.")
                break
            
            # batch expansion
            children = self._expand(node)
            if len(children) > 0:
                stats.node_expansions += 1
                # child is a leaf node
                for child in children:
                    child.h = self._task.evaluate(child.pillar_state)
                    potential_g = node.g + child.action_in.total_cost #self._task.compute_edge_cost(node.pillar_state, child.pillar_state)
                    if child.g is None or potential_g < child.g: #unassigned (expand does not set this) or reconnect
                        # found better path
                        child.set_parent(node)
                        child.g = potential_g
                        if child not in self._closed:
                            if self._check_similar and self._similar_to_closed_node(child):
                                continue
                        self._open.append(child)
            
            iters += 1

            self._open.remove(node)
            self._closed.append(node)

            if iters % log_every == 0:
                logger.info(f"Iteration Count: {iters} | Expanded {stats.node_expansions} nodes")

            if stats.node_expansions >= max_expansions:
                logger.info('Expanded max nodes without finding a plan')
                break

        q_root = [child.h for child in self._root_node.children]
        logger.info("  Q(root) values: %s" % to_str(q_root))

        logger.info("Statistics:")
        logger.info("----------")
        logger.info("  Total Node expansions: %d" % stats.node_expansions)

        if success:
            return goal, goal.find_path_from_root()
        return None,[]
   
    def _similar_to_closed_node(self, node):
        for closed_node in self._closed:
            if self._task.states_similar(node.pillar_state, closed_node.pillar_state):
                return True
        return False

    @staticmethod
    def load(filename, task, env, skills, cfg):
        with open(filename, 'rb') as f:
            root_node = load(f)
        return AStar(task, env, skills, cfg, root_node=root_node)
