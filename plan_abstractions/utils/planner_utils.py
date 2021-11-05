import copy
import logging
import pickle as pkl

from graphviz import Digraph
from pillar_state import State
import numpy as np

logger = logging.getLogger(__name__)


def planner_graph_to_dot(
    root_node,
    plan,
    path_to_save,
    planner_state_to_viz_string,
    pillar_state_equality,
    max_depth=3,
    node_constraint=(lambda x: True),
    print_graph=False,
    viz_graph=False,
):
    """Convert a search tree generated into a dot tree by extracting relevant
    information from the nodes.

    Parameters:
    -----------
        planner_state_to_viz_string: extract the information from a planner state
            that you are interested in
        max_depth: maximum depth of states in the search tree to be rendered
        node_constraint: only nodes that satisfy the constraint will be rendered
    """

    plan = copy.deepcopy(plan)
    graph = Digraph(comment="Search Tree")

    plan_node_attr = {'color': 'lightblue2', 'style': 'filled'}
    plan_edge_attr = {'color': 'lightcoral'}

    node_id = 0
    queue = []
    queue.append((root_node, node_id))
    graph.node(
        str(node_id), planner_state_to_viz_string(root_node)[0], shape="doublecircle", **plan_node_attr,
    )
    plan.pop(0)

    depth = 0
    while depth < max_depth:
        # breadth-first
        depth_size = len(queue)
        logger.debug(f"{depth_size} nodes at depth {depth}")
        while depth_size:
            (node, parent_id) = queue.pop(0)
            children = node.children

            for child in children:
                if node_constraint(child):
                    plan_node = False
                    node_id += 1
                    node_str, edge_str = planner_state_to_viz_string(child)
                    if plan and child.depth == plan[0].depth and pillar_state_equality(
                        child.pillar_state, plan[0].pillar_state
                    ):
                        # This is an intermediate step in the plan, thus cannot be equal to a node with no children
                        if len(plan) > 1 and len(child.children) == 0:
                            graph.node(str(node_id), node_str)
                        else:
                            graph.node(str(node_id), node_str, shape="doublecircle", **plan_node_attr)
                            plan_node = True
                            plan.pop(0)
                    else:
                        graph.node(str(node_id), node_str)

                    if plan_node:
                        graph.edge(str(parent_id), str(node_id), label=edge_str, **plan_edge_attr)
                    else:
                        graph.edge(str(parent_id), str(node_id), label=edge_str)

                    queue.append((child, node_id))

            depth_size -= 1
        logger.debug(f"Visited all nodes at depth {depth}")
        depth += 1
    logger.info(f"Visited {node_id} nodes in total")
    if print_graph:
        print(graph.source)
    graph.render(path_to_save, view=viz_graph)
