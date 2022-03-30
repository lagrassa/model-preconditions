import logging
import datetime
import os
import re

import numpy as np
import quaternion
from autolab_core import RigidTransform
from matplotlib import pyplot as plt
from pyquaternion import Quaternion
from scipy.stats import rv_discrete
from shapely.geometry import Polygon, LineString

from isaacgym import gymapi
from isaacgym_utils.math_utils import angle_axis_between_axes, np_to_quat, np_quat_to_quat, np_to_vec3, quat_to_np, \
    rpy_to_quat, transform_to_np, vec3_to_np, transform_to_RigidTransform, np_to_transform
from pillar_state import State
import torch

import logging
logger = logging.getLogger(__name__)


def object_name_to_asset_name(object_name):
    if "franka:" in object_name:
        object_name = object_name.split(':')[1]  # remove franka part to get asset name
    number_pattern = r'[0-9]'  # remove numbers from body_name
    asset_name = re.sub(number_pattern, '', object_name)
    return asset_name


def pillar_state_to_shapes(pillar_state, body_names, asset_name_to_eps_arr=None, plot=False):
    shapes = []
    if asset_name_to_eps_arr is None:
        asset_name_to_eps_arr = {}
        for body_name in body_names:
            asset_name_to_eps_arr[object_name_to_asset_name(body_name)] = [0,0]
        logger.warning("Warning using default asset name to eps arr")
    for body_name in body_names:
        asset_name = object_name_to_asset_name(body_name)
        sizes = pillar_state.get_values_as_vec([f"constants/{asset_name}_dims"])[:2]
        pos = pillar_state.get_values_as_vec([f"frame:{body_name}:pose/position"])
        quat = pillar_state.get_values_as_vec([f"frame:{body_name}:pose/quaternion"])  # wxyz
        eps_arr = asset_name_to_eps_arr[asset_name]

        shape = make_shape(pos, quat, sizes, eps_arr)
        shapes.append(shape)

    if plot:
        plot_shapes(shapes, body_names)

    return shapes

def plot_shapes(shapes, body_names):
    for shape, body_name in zip(shapes, body_names):
        coords = shape.boundary.coords.xy
        plt.plot(coords[0], coords[1], label=body_name)
    plt.gca().set_aspect('equal')
    plt.show()


def make_shape(pos, quat, sizes, eps_arr):
    # assume rectangular bounding box
    O_bbox_pts_np = (np.array([
        [1, 1, 0],
        [1, -1, 0],
        [-1, -1, 0],
        [-1, 1, 0]
    ]) * [sizes[0] / 2 + eps_arr[0], sizes[1] / 2 + eps_arr[1], 0])
    O_bbox_pts = [gymapi.Transform(p=np_to_vec3(pt)) for pt in O_bbox_pts_np]
    W_T_O = gymapi.Transform(
        p=np_to_vec3(pos),
        r=np_to_quat(quat, format='wxyz')
    )
    W_bbox_pts = [W_T_O * pt for pt in O_bbox_pts]
    shape = Polygon([vec3_to_np(pt.p) for pt in W_bbox_pts])
    return shape


def point_in_box(point, box_coords, box_dims):
    """

    Args:
        point: (x,y) tuple
        box_coords: center coords
        box_dims: width (x) height (y)

    Returns:

    """
    for dim in [0, 1]:
        if point[dim] < box_coords[dim] - box_dims[dim] / 2:
            return False  # too low
        if point[dim] > box_coords[dim] + box_dims[dim] / 2:
            return False
    return True


def generate_random_pose_and_quat(object_height, low, high, eps=1e-3):
    random_vec = np.random.uniform(low=low, high=high)
    random_yaw = np.deg2rad(random_vec[2])
    random_quat = quat_to_np(rpy_to_quat((0, 0, random_yaw)), format="wxyz")
    random_pose = np.hstack([random_vec[:2], [object_height / 2 + eps]])
    return random_pose, random_quat


def create_param_ddist(param_cfg):
    fields = list(param_cfg.keys())
    values = []
    probabilities = []
    for field in fields:
        values.append(field)
        probabilities.append(param_cfg[field])
    assert np.allclose(sum(probabilities), 1., atol=1e-4)
    ddist = rv_discrete(values=(range(len(values)), probabilities))
    ddist.sample = lambda: fields[ddist.rvs()]  # scipy rv_discrete doesnt support string fields
    return ddist

def normalize0tau(angle):
    if angle < 0:
        return angle + 2*np.pi
    if angle > 2*np.pi:
        return angle - 2*np.pi
    return angle

def min_distance_between_angles(x,y):
    x = normalize0tau(x)
    y = normalize0tau(y)
    inputs = [x,y]
    for input_val in inputs:
        assert input_val > 0 and input_val < 2*np.pi
    dist1 = abs((x-y) % 2*np.pi)
    dist2 = abs((y-x) % 2*np.pi)
    return min(dist1, dist2)



def shapes_in_collision(shapes, body_names, plot=False, return_one=False):
    collisions = []
    for body1, shape1 in zip(body_names, shapes):
        for body2, shape2 in zip(body_names, shapes):
            if body1 == body2:
                continue
            is_collision = shape1.intersects(shape2)
            disjoint = shape1.disjoint(shape2)
            assert is_collision == (not disjoint)
            if is_collision:
                collisions.append((body1, body2))
            if return_one:
                break
    if plot:
        if len(collisions) == 0:
            print("No collisions")
        plt.legend()
        plt.show()
    return collisions


def to_str(seq, decimal=3):
    return str(np.around(seq, decimal))


def is_debug(logger):
    return logging.getLevelName(logger.level) == "DEBUG"


def yaw_from_quat(quat):
    return Quaternion(x=quat.x, y=quat.y, z=quat.z, w=quat.w) \
        .yaw_pitch_roll[0]  # the math_utils version jumps around, in the range `[-pi, pi]`


def yaw_from_np_quat(quat):
    return Quaternion(quat).yaw_pitch_roll[0]  # the math_utils version jumps around, in the range `[-pi, pi]`


def ee_yaw_to_np_quat(yaw):
    c_yaw, s_yaw = np.cos(yaw), np.sin(yaw)
    quat = quaternion.from_rotation_matrix(np.array([
        [c_yaw, s_yaw, 0],
        [s_yaw, -c_yaw, 0],
        [0, 0, -1]
    ]))
    return quat


def yaw_to_np_quat(yaw):
    c_yaw, s_yaw = np.cos(yaw), np.sin(yaw)
    quat = quaternion.from_rotation_matrix(np.array([
        [c_yaw, -s_yaw, 0],
        [s_yaw, c_yaw, 0],
        [0, 0, 1]
    ]))
    return quat


def xy_yaw_to_transform(xy_yaw, for_ee=False):
    if for_ee:
        quat = ee_yaw_to_np_quat(xy_yaw[2])
    else:
        quat = quaternion.from_euler_angles([0, 0, xy_yaw[2]])

    return gymapi.Transform(
        p=gymapi.Vec3(xy_yaw[0], xy_yaw[1], 0),
        r=np_quat_to_quat(quat)
    )


def transform_to_xy_yaw(transform):
    return np.array([
        transform.p.x,
        transform.p.y,
        yaw_from_np_quat(quat_to_np(transform.r, 'wxyz'))
    ])


def xyz_yaw_to_transform(xyz_yaw, for_ee=False):
    if for_ee:
        quat = ee_yaw_to_np_quat(xyz_yaw[3])
    else:
        quat = quaternion.from_euler_angles([0, 0, xyz_yaw[3]])

    return gymapi.Transform(
        p=np_to_vec3(xyz_yaw[:3]),
        r=np_quat_to_quat(quat)
    )

def xyz_yaw_to_rt(xyz_yaw, for_ee=False, from_frame='', to_frame=''):
    gymapi_transform = xyz_yaw_to_transform(xyz_yaw, for_ee=for_ee)
    return transform_to_RigidTransform(gymapi_transform, from_frame=from_frame, to_frame=to_frame)


def transform_to_xyz_yaw(transform):
    return np.array([
        transform.p.x,
        transform.p.y,
        transform.p.z,
        yaw_from_np_quat(quat_to_np(transform.r, 'wxyz'))
    ])


# for making pillar_state's gripper/finger rotations be "z-down"
r_flip_yz = np_quat_to_quat(quaternion.from_rotation_matrix(np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]
])))


def angle_axis_between_quats(q0, q1):
    '''
    Finds dq s.t. dq * q1 = q0
    '''
    if quaternion.as_float_array(q1) @ quaternion.as_float_array(q0) < 0:
        q0 = -q0
    dq = q0 * q1.inverse()
    return quaternion.as_rotation_vector(dq)


def seven_dim_internal_state_to_pose(internal_state, parameters, env_idx):
    curr_pose = internal_state[:7]
    goal_quat = quat_to_np(ee_yaw_to_np_quat(parameters[env_idx][3]), format="wxyz")
    goal_pose = np.hstack([parameters[env_idx][:3], goal_quat])
    return curr_pose, goal_pose

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def get_numpy(tensor):
    # not sure if I should do detach or not here
    return tensor.to('cpu').detach().numpy()


def get_rod_rel_goal_RigidTransforms_x_in(x_th, y_th, sy_rod):
    return [
        # positive x
        RigidTransform(
            rotation=np.array([
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ]),
            translation=np.array([x_th, sy_rod / 2, 0])
        ),
        RigidTransform(
            rotation=np.array([
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ]),
            translation=np.array([x_th, 0, 0])
        ),
        RigidTransform(
            rotation=np.array([
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ]),
            translation=np.array([x_th, -sy_rod / 2, 0])
        ),
        # negative x
        RigidTransform(
            rotation=np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]),
            translation=np.array([-x_th, sy_rod / 2, 0])
        ),
        RigidTransform(
            rotation=np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]),
            translation=np.array([-x_th, 0, 0])
        ),
        RigidTransform(
            rotation=np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]),
            translation=np.array([-x_th, -sy_rod / 2, 0])
        ),
        # positive y
        RigidTransform(
            rotation=np.array([
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1]
            ]),
            translation=np.array([0, y_th, 0])
        ),
        # negative y
        RigidTransform(
            rotation=np.array([
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1]
            ]),
            translation=np.array([0, -y_th, 0])
        ),
    ]


def get_rod_grasps_transforms(sy_rod, sz_rod, grasp_offset=0):
    # Object to EE
    if isinstance(grasp_offset, list):
        grasp_offsets = grasp_offset
    else:
        grasp_offsets = [grasp_offset]

    angles = [np.pi / 2, -np.pi / 2]
    quats = [rpy_to_quat([0, 0, angle]) for angle in angles]
    grasp_height = sz_rod / 2
    translations = []
    for _grasp_offset in grasp_offsets:
        translations.append([0,0,grasp_height])
        translations.append([0, -(sy_rod/2 - _grasp_offset), grasp_height])
        translations.append([0, sy_rod / 2 - _grasp_offset, grasp_height])
    grasp_transforms = []
    for quat in quats:
        for translation in translations:
            new_transform = gymapi.Transform(p=np_to_vec3(translation), r=quat)
            grasp_transforms.append(new_transform)
    return grasp_transforms

def make_save_dir_and_get_plan_results_filename(plan_results_dirname):
    if not os.path.isdir(plan_results_dirname):
        os.mkdir(plan_results_dirname)
    for i in range(200):
        plan_results_filename_option = os.path.join(plan_results_dirname, f"plan_results_{i}.npy")
        if os.path.isfile(plan_results_filename_option):
            continue
        return plan_results_filename_option

def get_min_dist(states_and_params):
    rod0_dist = np.linalg.norm(states_and_params[:,:2]- states_and_params[:,4:6], axis=1)
    rod1_dist = np.linalg.norm(states_and_params[:,:2]- states_and_params[:,8:10], axis=1)
    min_dist = np.min(np.vstack([rod0_dist, rod1_dist]), axis=0)
    return min_dist

def yaw_distance(yaw1, yaw2):
    return min((yaw1-yaw2) % 2*np.pi, (yaw2-yaw1)% 2*np.pi)

def pillar_state_to_franka_xyzyaw(pillar_state):
    pose = get_pose_pillar_state(pillar_state, "franka:ee")
    pose_tf = np_to_transform(pose, format="wxyz")
    xy_yaw = transform_to_xy_yaw(pose_tf)
    xyz_yaw = np.hstack([xy_yaw[:2], pose[2], xy_yaw[2]])
    return xyz_yaw

def set_fingers_and_visualize_pillar_state(pillar_state, franka_cls):
    body_names = ["franka:finger_left", "franka:finger_right", "rod0", "rod1"]
    xyz_yaw = pillar_state_to_franka_xyzyaw(pillar_state)
    asset_name_to_eps_arr, franka_body_names, obj_body_names, potential_pillar_state = place_grippers_in_pillar_state(pillar_state, 0, xyz_yaw)
    franka_cls.is_in_collision(potential_pillar_state, body_names=body_names,
                                  asset_name_to_eps_arr=asset_name_to_eps_arr, plot=1)


def params_cause_collision_franka(state, parameters, franka_cls, plot=False, ignore_in_hand=None, gripper_eps=1e-2, return_state = False):
    franka_xyzyaw = parameters[:4]
    asset_name_to_eps_arr, franka_body_names, obj_body_names, potential_pillar_state = place_grippers_in_pillar_state(state, gripper_eps,
                                                                                                                      franka_xyzyaw)
    for obj_body_name in obj_body_names:
        if ignore_in_hand is not None and ignore_in_hand == obj_body_name:
            continue
        body_names = franka_body_names + [obj_body_name]
        if franka_cls.is_in_collision(potential_pillar_state, body_names=body_names, 
                        asset_name_to_eps_arr=asset_name_to_eps_arr, plot=plot):
            if return_state:
                return True, potential_pillar_state
            else:
                return True
    if return_state:
        return False, potential_pillar_state
    return False




def place_grippers_in_pillar_state(state, gripper_eps, franka_xyzyaw=None):
    if franka_xyzyaw is None:
        franka_xyzyaw = pillar_state_to_franka_xyzyaw(state)
        franka_xyzyaw[1] *= -1
    gripper_thickness = state.get_values_as_vec(["constants/finger_left_dims"])[0]
    gripper_width = state.get_values_as_vec(["frame:franka:gripper/width"])[0]
    T_gripper_to_world_nps = get_left_and_right_gripper_poses(franka_xyzyaw, gripper_width, gripper_thickness)
    potential_pillar_state = State.create_from_serialized_string(state.get_serialized_string())
    for key, T_gripper_to_world_np in T_gripper_to_world_nps.items():
        potential_pillar_state.update_property(f"frame:franka:finger_{key}:pose/position", T_gripper_to_world_np[:3])
        potential_pillar_state.update_property(f"frame:franka:finger_{key}:pose/quaternion", T_gripper_to_world_np[3:])
    asset_name_to_eps_arr = {
        "finger_left": [gripper_eps, gripper_eps],
        "finger_right": [gripper_eps, gripper_eps],
        "drawer": [1e-4, 1e-4],
        "rod": [1e-4, 1e-4]
    }
    # only care about objects and fingers
    obj_body_names = [body_name for body_name in get_object_names_in_pillar_state(state) if "franka" not in body_name]
    franka_body_names = [body_name for body_name in get_object_names_in_pillar_state(state) if "finger" in body_name]
    return asset_name_to_eps_arr, franka_body_names, obj_body_names, potential_pillar_state


def get_left_and_right_gripper_poses(franka_xyzyaw, gripper_width, gripper_thickness):
    distance_to_center = gripper_width + gripper_thickness
    T_ee_to_world = xyz_yaw_to_transform(franka_xyzyaw)
    T_gripper_to_ee_left = gymapi.Transform(p=gymapi.Vec3(0, distance_to_center, 0))
    T_gripper_to_ee_right = gymapi.Transform(p=gymapi.Vec3(0, -distance_to_center, 0))
    T_gripper_to_world_nps = {
        'left': transform_to_np(T_ee_to_world * T_gripper_to_ee_left, format="wxyz"),
        'right': transform_to_np(T_ee_to_world * T_gripper_to_ee_right, format="wxyz")
    }
    return T_gripper_to_world_nps


def get_object_names_in_pillar_state(pillar_state):
    object_names = []
    for prop_name in pillar_state.get_prop_names():
        if "frame" in prop_name and "position" in prop_name and "joint" not in prop_name:
            parts = prop_name.split(":")
            if len(parts) == 4:
                object_names.append(f"{parts[1]}:{parts[2]}")
            elif len(parts) == 3:
                object_names.append(parts[1])
            else:
                raise ValueError("Unexpected number of parts in prop_name {prop_name}")
    return object_names


def states_similar_within_tol(pillar_state_1, pillar_state_2, yaw_tol, pos_tol):
    object_names = get_object_names_in_pillar_state(pillar_state_1)
    for object_name in object_names:
        pose_1 = np.array(get_pose_pillar_state(pillar_state_1, object_name))
        pose_2 = np.array(get_pose_pillar_state(pillar_state_2, object_name))
        if np.linalg.norm(pose_1 - pose_2) > pos_tol:
            return False
        yaw_1 = yaw_from_np_quat(pose_1[3:])
        yaw_2 = yaw_from_np_quat(pose_1[3:])
        if abs(yaw_1 - yaw_2) > yaw_tol:
            return False

    return True


def get_pose_pillar_state(pillar_state, object_name):
    prop_names = [f"frame:{object_name}:pose/position", f"frame:{object_name}:pose/quaternion"]
    pose = pillar_state.get_values_as_vec(prop_names)
    return pose

def get_color_pillar_state(pillar_state, object_name):
    asset_name = object_name_to_asset_name(object_name)
    color_string = f"constants/{asset_name}:color"
    if color_string not in pillar_state.get_prop_names():
        print(f"No color found for object {asset_name} in pillar_state")
        color = [0,0,0]
    else:
        color =  pillar_state.get_values_as_vec([color_string])
    assert len(color) == 3
    return color


def pillar_state_obj_to_RigidTransform(pillar_state, obj_name, from_frame, to_frame, align_z=False):
    # if align_z is True, then will "flush" the z-axis of the rotation to be 
    # either [0, 0, 1] or [0, 0, -1], whichever is closest.
    # This is useful for getting ee transforms but only caring about yaw.

    pos = pillar_state.get_values_as_vec([f'frame:{obj_name}:pose/position'])
    quat = pillar_state.get_values_as_vec([f'frame:{obj_name}:pose/quaternion'])
    T = RigidTransform(
        translation=pos,
        rotation=quaternion.as_rotation_matrix(quaternion.from_float_array(quat)),
        from_frame=from_frame, to_frame=to_frame
    )

    if align_z:
        R = T.rotation
        r_proj = angle_axis_between_axes(R[:, 2], np.array([0, 0, np.sign(R[2, 2])]))
        q_proj = quaternion.from_rotation_vector(r_proj)
        R_proj = quaternion.as_rotation_matrix(q_proj)
        T.rotation = R_proj @ R

    return T


def pillar_state_obj_to_transform(pillar_state, obj_name, align_z=False):
    # if align_z is True, then will "flush" the z-axis of the rotation to be 
    # either [0, 0, 1] or [0, 0, -1], whichever is closest.
    # This is useful for getting ee transforms but only caring about yaw.
    try:
        pos = pillar_state.get_values_as_vec([f'frame:{obj_name}:pose/position'])
    except:
        import ipdb; ipdb.set_trace()
    quat = pillar_state.get_values_as_vec([f'frame:{obj_name}:pose/quaternion'])
    T = gymapi.Transform(p=np_to_vec3(pos), r=np_to_quat(quat, 'wxyz'))

    if align_z:
        q = quaternion.from_float_array(quat)
        R = quaternion.as_rotation_matrix(q)
        r_proj = angle_axis_between_axes(R[:, 2], np.array([0, 0, np.sign(R[2, 2])]))
        q_proj = quaternion.from_rotation_vector(r_proj)
        T.r = np_quat_to_quat(q_proj * q)

    return T


def set_franka_pillar_properties_from_init_states_arr(franka_name, franka_init_states, init_state_idx,
                                                      potential_pillar_state):
    potential_pillar_state.update_property(f"frame:{franka_name}:ee:pose/position",
                                           franka_init_states['ee_poses'][init_state_idx, :3])
    potential_pillar_state.update_property(f"frame:{franka_name}:ee:pose/quaternion",
                                           franka_init_states['ee_poses'][init_state_idx, 3:])
    potential_pillar_state.update_property(f"frame:{franka_name}:ee:pose/angular_velocity", [0] * 3)
    potential_pillar_state.update_property(f"frame:{franka_name}:ee:pose/linear_velocity", [0] * 3)
    potential_pillar_state.update_property(f"frame:{franka_name}:joint/position",
                                           franka_init_states['joints'][init_state_idx])
    potential_pillar_state.update_property(f"frame:{franka_name}:joint/velocity", np.zeros(7))
    potential_pillar_state.update_property(f"frame:{franka_name}:gripper/width",
                                           franka_init_states['gripper_widths'][init_state_idx])
    potential_pillar_state.update_property(f"frame:{franka_name}:finger_left:pose/position",
                                           franka_init_states['finger_left_poses'][init_state_idx, :3])
    potential_pillar_state.update_property(f"frame:{franka_name}:finger_left:pose/quaternion",
                                           franka_init_states['finger_left_poses'][init_state_idx, 3:])
    potential_pillar_state.update_property(f"frame:{franka_name}:finger_right:pose/position",
                                           franka_init_states['finger_right_poses'][init_state_idx, :3])
    potential_pillar_state.update_property(f"frame:{franka_name}:finger_right:pose/quaternion",
                                           franka_init_states['finger_right_poses'][init_state_idx, 3:])

def extract_xy_dims_and_height(asset_name, cfg, finger_dims=None):
    if 'finger' in asset_name:
        assert finger_dims is not None
        object_height = finger_dims[2]
        xy_dims = finger_dims[:2]
    elif 'radius' in cfg[asset_name]["dims"].keys():
        object_height = cfg[asset_name]['dims']['radius']
        xy_dims = [cfg[asset_name]['dims'][dim] for dim in ['radius', 'width']]
    else:
        object_height = cfg[asset_name]['dims']['sz']
        xy_dims = [cfg[asset_name]['dims'][dim] for dim in ['sx', 'sy']]
    return xy_dims, object_height

def extract_effects_dict_as_arrays(env, effects, obj_names, initial_states, anchor_obj_name=None, as_numpy=False):
    # TODO lagrassa generalize
    end_states = effects["end_states"]
    if isinstance(end_states[0], bytes):
        end_states = [State.create_from_serialized_string(end_state) for end_state in end_states]
    end_states_vec = np.vstack([
        env.pillar_state_to_sem_state(end_state, obj_names, anchor_obj_name=anchor_obj_name,
                                      ref_pillar_state=init_state)
        for (init_state, end_state) in zip(initial_states, end_states)
    ])
    costs = effects["costs"]
    T_exec = effects["T_exec"]
    info_plan = effects["info_plan"]
    if isinstance(info_plan, list):
        T_plans = [info["T_plan"] for info in info_plan]
    else:
        T_plans = info_plan["T_plan"]
    if as_numpy:
        # Return everything as a 2D array.
        return np.array(end_states_vec), np.array(costs).reshape(-1, 1), np.array(T_exec).reshape(-1, 1), np.array(
            T_plans).reshape(-1, 1)
    else:
        return end_states_vec, costs, T_exec, T_plans


def is_pos_B_btw_A_and_C(pos_A, pos_B, pos_C):
    # checking for 2D positions A, B, and C if B is between A and C
    # project B along line joining A and C
    # projection = A + dot(AB,AC) / dot(AC,AC) * AC
    # for example, used for checking if rod is between gripper and goal:
    # A: gripper, C: goal, B: rod
    
    dot_AP_AB = np.dot(pos_B - pos_A, pos_C - pos_A)
    dot_AB_AB = np.dot(pos_C - pos_A, pos_C - pos_A)
    if dot_AB_AB == 0 and dot_AP_AB == 0:
        return False, np.linalg.norm(pos_C-pos_B) #Can't be between, A and C same point.
    try:
        projection = pos_A + dot_AP_AB / dot_AB_AB * (pos_C - pos_A)
    except FloatingPointError:
        print("Problem")

    # if 0<eps<1 projection lies on line joining gripper and goal
    epsx = (projection[0] - pos_A[0]) / (pos_C[0] - pos_A[0]) if (pos_C[0] - pos_A[0]) != 0.0 else 0.5 
    # if 0<eps<1 projection lies on line joining gripper and goal
    epsy = (projection[1] - pos_A[1]) / (pos_C[1] - pos_A[1]) if (pos_C[1] - pos_A[1]) != 0.0 else 0.5  
    
    in_between = (epsx > 0 and epsx < 1) and (epsy > 0 and epsy < 1)
    
    return in_between, projection


def rod_intersects_fingers(rod_pos, rod_quat, rod_dim_y, left_finger_pos, right_finger_pos):
    rod_y_axis = quaternion.as_rotation_matrix(rod_quat)[:2, 1]
    rod_tip_0 = rod_pos + rod_y_axis * rod_dim_y / 2
    rod_tip_1 = rod_pos - rod_y_axis * rod_dim_y / 2

    rod = LineString([rod_tip_0, rod_tip_1])
    fingers = LineString([left_finger_pos, right_finger_pos])
    return rod.intersects(fingers)


def pretty_print_param_infos(success_per_param_type):
    print("----------------------")
    for param_type in success_per_param_type.keys():
        if len(success_per_param_type[param_type]) == 0:
            print(f"{param_type} was not sampled")
        else:
            success_rate = np.mean(success_per_param_type[param_type])
            num_successes = np.sum(success_per_param_type[param_type])
            num_tries = len(success_per_param_type[param_type])
            print(
                f"{param_type} satisfied termination conditions and preconditions at a rate of {success_rate} or {num_successes} out of {num_tries}")
    print("----------------------")

def get_num_rods_from_pillar_state(pillar_state):
    obj_names = get_object_names_in_pillar_state(pillar_state)
    return len([obj_name for obj_name in obj_names if "rod" in obj_name])



def is_obj_in_gripper(state, max_height_diff=0.02, gripper_err=1e-3):
    """

    Args:
        state:
        max_height_diff:
        gripper_err:

    Returns:
        Whether an object is between the gripper and the width if any
    """
    ee_pose = get_pose_pillar_state(state, "franka:ee")[:3]
    object_names = get_object_names_in_pillar_state(state)
    for object_name in object_names:
        if "franka" in object_name:
            continue
        if "drawer" in object_name:
            continue
        asset_name = object_name_to_asset_name(object_name)
        prop_pose = get_pose_pillar_state(state, object_name)[:3]
        dims_name = f"constants/{asset_name}_dims"
        if dims_name not in state.get_prop_names():
            logger.warning("No dimensions found for object. Assuming not graspable")
            continue
        largest_object_dim = max(state.get_values_as_vec([dims_name]))
        max_distance = largest_object_dim / 2  # TODO lagrassa base this on the object orientation.
        # Just using distance from center doesn't allow grasps on the end of the rods however
        if np.linalg.norm(np.array(prop_pose)[:2] - np.array(ee_pose)[:2]) < max_distance:
            if abs(prop_pose[2] - ee_pose[2]) > max_height_diff:
                continue
            # in between grippers (probably)
            smallest_object_dim = min(state.get_values_as_vec([f"constants/{asset_name}_dims"]))
            return True, smallest_object_dim, object_name
    return False, None, None



def pretty_print_state_with_params(pos, yaw):
    '''Pretty print a list or array of positions and a scalar yaw.'''
    assert type(pos) is np.ndarray or type(pos) is list
    pos_str = np.array_str(np.array(pos), precision=3, suppress_small=True)
    yaw_str = f'{yaw:.3f} ({np.rad2deg(yaw):.1f} deg)'
    return 'pos: ' + pos_str + ', yaw: ' + yaw_str 


def pretty_print_array(pos, prefix=''):
    assert type(pos) is np.ndarray or type(pos) is list
    pos_str = np.array_str(np.array(pos), precision=3, suppress_small=True, max_line_width=200)
    if len(prefix) > 0:
        return f'{prefix}: {pos_str}'
    else:
        return f'{pos_str}'

def plot_devs(deviations, label):
    count, bins_count = np.histogram(deviations, bins=20)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    plt.plot(bins_count[1:], cdf, label=label)
    plt.xlim([0,0.1])



def compute_deviation_helper(pred_deviations, actual_deviations):
    return np.mean(np.abs(pred_deviations - actual_deviations))

def append_postfix_to_filename(og_path, postfix):
    all_but_ext_name, ext_name = os.path.splitext(og_path)
    return f"{all_but_ext_name}_{postfix}{ext_name}"


def get_formatted_time():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def dists_and_actions_from_states_and_parameters(states_and_parameters, state_ndims=None, only_dists=True):
    assert state_ndims is not None
    parameters = states_and_parameters[:, state_ndims:]
    transformed_features = augment_with_dists(states_and_parameters, only_dists=only_dists)
    dists_and_actions = np.hstack([transformed_features, parameters])
    return dists_and_actions

def identity(states_and_parameters, state_ndims=None):
    return states_and_parameters

def extract_first_and_last(states_and_parameters, state_ndims=3, only_dists=True):
    """
    Pretty special purpose to the water world where the first and last are the most important. eventually this
    should be learned
    """
    first_only = states_and_parameters[:,0:2]
    last_only = states_and_parameters[:,state_ndims-1].reshape(-1,1)
    parameters = states_and_parameters[:, state_ndims:]
    state_transformed_and_actions = np.hstack([first_only, last_only, parameters])
    return state_transformed_and_actions


def combine_effects(effects_list):
    combined_effects = {}
    if len(effects_list) == 0:
        logger.warning("No effects in effects list")
    for key in effects_list[0].keys(): #assuming structure is the same
        combined_effects[key] = []
        for effect in effects_list:
            combined_effects[key].extend(effect[key])
    return combined_effects

def augment_with_dists(states_and_params, only_dists=False):
    rod0_dist = np.linalg.norm(states_and_params[:,:2]- states_and_params[:,4:6], axis=1)
    rod1_dist = np.linalg.norm(states_and_params[:,:2]- states_and_params[:,8:10], axis=1)
    between_rod_dist = np.linalg.norm(states_and_params[:,4:6]- states_and_params[:,8:10], axis=1)
    if only_dists:
        return np.hstack([rod0_dist.reshape(-1, 1), rod1_dist.reshape(-1, 1),
                          between_rod_dist.reshape(-1, 1)])
    else:
        return np.hstack([states_and_params, rod0_dist.reshape(-1,1), rod1_dist.reshape(-1,1), between_rod_dist.reshape(-1,1)])
