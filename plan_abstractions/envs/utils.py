import numpy as np
import quaternion
from pyquaternion import Quaternion
from isaacgym_utils.math_utils import np_to_transform, angle_axis_between_axes


def get_pose_pillar_state(pillar_state, object_name):
    # pillar state only makes sense in the context of an env so makes sense to evaluate based on the env
    prop_names = [f"frame:{object_name}:pose/position", f"frame:{object_name}:pose/quaternion"]
    pose = pillar_state.get_values_as_vec(prop_names)
    return pose


def get_joint_position_pillar_state(pillar_state, robot_name):
    # pillar state only makes sense in the context of an env so makes sense to evaluate based on the env
    return pillar_state.get_values_as_vec([f"frame:{robot_name}:joint/position"])


def get_joint_velocity_pillar_state(pillar_state, robot_name):
    # pillar state only makes sense in the context of an env so makes sense to evaluate based on the env
    return pillar_state.get_values_as_vec([f"frame:{robot_name}:joint/velocity"])


def get_gripper_width_pillar_state(pillar_state, robot_name):
    # pillar state only makes sense in the context of an env so makes sense to evaluate based on the env
    return pillar_state.get_values_as_vec([f"frame:{robot_name}:gripper/width"])


def is_pose_of_object_close(state1, state2, object_name, position_tol=5e-3, angle_tol=5e-3, velocity_tol=0.01, show_diff=False, yaw_only=False):
    """
    each pose array of form [x,y,z, w, x, y, z]

    if yaw_only is True, will first align the z-axis of the object in state1 to that in state2,
    then, angle_distance is only computed as the yaw difference
    """
    pose_arr1 = get_pose_pillar_state(state1, object_name)
    pose_arr2 = get_pose_pillar_state(state2, object_name)
    pos_close = np.allclose(pose_arr1[:3], pose_arr2[:3], atol=position_tol)
    if yaw_only:
        q1 = quaternion.from_float_array(pose_arr1[3:])
        q2 = quaternion.from_float_array(pose_arr2[3:])

        R1 = quaternion.as_rotation_matrix(q1)
        R2 = quaternion.as_rotation_matrix(q2)

        r_proj = angle_axis_between_axes(R1[:, 2], R2[:, 2])
        q_proj = quaternion.from_rotation_vector(r_proj)
        q1_proj = q_proj * q1

        dq = q2 * q1_proj.inverse()
        dR = quaternion.as_rotation_matrix(dq)
        angle_distance = np.arccos(dR[0, 0])
    else:
        angle_distance = Quaternion.absolute_distance(Quaternion(pose_arr1[3:]), Quaternion(pose_arr2[3:]))
    angle_close = angle_distance < angle_tol

    if show_diff:
        print(f"Angle distance was {angle_distance}")
        print(f"Pose 1 was {pose_arr1} Pose 2 was {pose_arr2}")

    return pos_close and angle_close


def is_state_of_robot_close(state1, state2, robot_name, position_tol=5e-3, velocity_tol=5e-3, width_tol=5e-3):
    q1 = get_joint_position_pillar_state(state1, robot_name)
    q2 = get_joint_position_pillar_state(state2, robot_name)

    qd1 = get_joint_velocity_pillar_state(state1, robot_name)
    qd2 = get_joint_velocity_pillar_state(state2, robot_name)

    w1 = get_gripper_width_pillar_state(state1, robot_name)
    w2 = get_gripper_width_pillar_state(state2, robot_name)

    position_close = np.allclose(q1, q2, atol=position_tol)
    velocity_close = np.allclose(qd1, qd2, atol=velocity_tol)
    width_close = np.allclose(w1, w2, atol=width_tol)

    return position_close and velocity_close and width_close


def set_pose(pose, name, object_ref, env_idx):
    """
    Throw error if no id exists in this environment with that name
    return transform
    """
    transform = np_to_transform(np.hstack(pose), format="wxyz")
    object_ref.set_rb_transforms(name=name, transforms=[transform], env_idx=env_idx)
    return transform


def make_env_with_init_states(env_cls, cfg, init_state_kwargs={}, setup_callbacks=[]):
    env = env_cls(cfg, setup_callbacks=setup_callbacks)
    [env._scene.render() for _ in range(100)]
    initial_state_gen = env.generate_init_states(cfg, **init_state_kwargs)
    initial_state = next(initial_state_gen)
    env.set_all_states([initial_state] * env.n_envs, n_steps=10)
    return env
