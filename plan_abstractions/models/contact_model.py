#from ..learning.mde_gnn_utils import DIM_IDX, POSE_DIM
from ..utils.utils import make_shape, get_left_and_right_gripper_poses, get_numpy
import numpy as np
def node_indices_in_contact(graph, contact_distance = 0.01):
    """

    :param graph: graph to evaluate
    :param contact_distance: minimum distance between surfaces to be considered "in contact"
    But remember that not all contacts are the same. We could do something where we consider "edge
    alignment" or what portion of the surface is in contact
    :return: 1D edge_attr which represents whether the objects are in contact
    """
    #requires there to be 3D dimensionality information.
    node_attr_np = get_numpy(graph.x)
    node_dims = node_attr_np[:, DIM_IDX:]
    pose_info = node_attr_np[:, :POSE_DIM]
    edge_idx_np = get_numpy(graph.edge_index)
    fixed_quat = [1,0,0,0]
    nodes_in_contact = np.zeros(edge_idx_np.shape[1],)
    assert POSE_DIM == 3 #do not support rotation yet, assume all the same rotation
    node_to_shape = []
    eps_arr = [contact_distance, contact_distance]
    for node_attr in node_attr_np:
        #make shape using util function
        shape = make_shape(node_attr[:POSE_DIM], fixed_quat, node_attr[DIM_IDX:], eps_arr)
        node_to_shape.append(shape)

    for edge_idx in range(edge_idx_np.shape[1]):
        pair = edge_idx_np[:, edge_idx]
        #assert something about the shapes: make the distance between their surfaces
        #also check the z values of them manually: see if they should intersect.
        #just check the clips of them
        shape1 = node_to_shape[pair[0]]
        shape2 = node_to_shape[pair[1]]
        if shape1.intersects(shape2):
            nodes_in_contact[edge_idx] = 1
    return nodes_in_contact



def gripper_in_contact_with_drawer_vector(state, params, drawer_edge_dims, contact_distance = 0.01):
    """
                                              ee pose   rod 1     rod 2      drawer edge   gripper width
    :param vector with 2 rods and 1 gripper. 0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 | 15
    :param contact_distance: minimum distance between surfaces to be considered "in contact"
    But remember that not all contacts are the same. We could do something where we consider "edge
    alignment" or what portion of the surface is in contact
    :return:
    """
    #requires there to be 3D dimensionality information.
    gripper_width = state[-1]
    target_ee_pose = params[:3]
    target_ee_yaw = params[3]
    finger_shapes = [] #(assumed 0.01, 0.01, 0.01)
    finger_dims = [0.01, 0.01]
    eps_arr = [contact_distance, contact_distance]
    franka_xyzyaw = params[:4]
    gripper_thickness = contact_distance
    forward_amount = 0.02 #hard coded should come from skill params or config in future. This is the amount the franka is behind the drawer
    gripper_transforms = get_left_and_right_gripper_poses(franka_xyzyaw, gripper_width, gripper_thickness).values()
    for transform in gripper_transforms:
        #make shape using util function
        shape = make_shape(transform[:3], transform[3:], finger_dims, eps_arr)
        finger_shapes.append(shape)
    #make drawer shape
    fixed_quat = [1,0,0,0]
    drawer_edge_pose = state[12:15]
    drawer_edge_pose_shifted_by_forward_amount = drawer_edge_pose.copy()
    drawer_edge_pose_shifted_by_forward_amount[1] += forward_amount
    drawer_shape = make_shape(drawer_edge_pose_shifted_by_forward_amount, fixed_quat, drawer_edge_dims, eps_arr)
    for finger_shape in finger_shapes:
        if finger_shape.intersects(drawer_shape):
            return True
    return False






