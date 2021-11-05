from isaacgym import gymapi
from plan_abstractions.models.analytical_models import SEMAnalyticalDrawerAndRobot, SEMSimpleFreeSpace, SEMAnalyticalRodsAndRobot
from isaacgym_utils.math_utils import angle_axis_between_quats
from plan_abstractions.utils import  xyz_yaw_to_rt
import numpy as np

ATOL = 1e-3


def test_drawer_sem():
    drawer_edge_dims =  [-1.25, 0.01, 0.07]
    dim_state = 16
    sem = SEMAnalyticalDrawerAndRobot(1, dim_state, drawer_edge_dims)
    bin_pose = [-1.5, 0.3-drawer_edge_dims[1]/2., drawer_edge_dims[2]/2]
    ee_pose = np.array([-1.6, 0.1, 0.4, 0])
    rod_pose = np.array([7,0,0, 0]) #doesn't really matter
    finger_width = 0.02
    input_state = np.hstack([ee_pose, rod_pose, rod_pose, bin_pose, finger_width])
    end_y = -0.1

    params = [-1.5, 0.3, drawer_edge_dims[2]/2, np.pi/2, end_y]
    cond = np.hstack([input_state, params])
    res = sem.sample(cond, 0)
    drawer_diff = res['x_hats'][-1][12:15]
    computed_cost = res['x_hats'][-1][-2]
    first_part_cost = np.linalg.norm([ee_pose[:2]-params[:2]]) + abs(ee_pose[2]-params[2])
    opening_cost = abs(params[-1]-params[1])
    #params_reminder : [x, start_y, z, theta, end_y]
    true_cost =  first_part_cost + opening_cost
    assert(np.isclose(true_cost, computed_cost, atol=0.001))

    desired_open_drawer_diff = np.array([0, end_y-params[1]+sem._gripper_space, 0])
    assert np.allclose(desired_open_drawer_diff, drawer_diff)

    #and when the drawer should not move
    too_far_params = [-1.5, 0.5, drawer_edge_dims[2]/2, np.pi/2, end_y]
    too_far_cond = np.hstack([input_state, too_far_params])
    too_far_res = sem.sample(too_far_cond, 0)
    drawer_diff = too_far_res['x_hats'][-1][12:15]

    assert np.allclose(drawer_diff, np.zeros_like(drawer_diff))

def test_freespace_no_contact():
    sem = SEMSimpleFreeSpace(2)
    ee_pose = np.array([0.6, 0.1, 0.4, 0])
    rod_pose = np.array([0.6,0.4,0.02,0.3]) #somewhat far away
    input_state = np.hstack([ee_pose, rod_pose, rod_pose])
    target_pose = rod_pose[:].copy()
    params = target_pose
    params[2] += 0.02
    cond = np.hstack([input_state, params])
    res = sem.sample(cond, 0)
    computed_cost = res['x_hats'][-1][-2]
    true_cost = np.linalg.norm([ee_pose[:2]-params[:2]]) + abs(ee_pose[2]-params[2])
    assert(np.isclose(true_cost, computed_cost, atol=0.001))
    ee_diff, rod1_diff, rod2_diff = _extract_diffs(res)
    desired_ee_diff = params - ee_pose
    desired_ee_diff[3] = -params[3]- ee_pose[3]
    assert np.allclose(ee_diff, desired_ee_diff, atol=ATOL)
    assert np.allclose(rod1_diff, 0, atol=ATOL)
    assert np.allclose(rod2_diff, 0, atol=ATOL)

def test_place_in_contact_with_one_no_rotate():
    sem = SEMAnalyticalRodsAndRobot(2)
    other_rod_pose = np.array([-1.6,0.4,0.02,0.3])#somewhat far away
    target_rod_pose = np.array([0.5,0.2,0.02,0.3])#close
    ee_pose = target_rod_pose.copy()
    ee_pose[3] = target_rod_pose[-1] + np.pi/2
    input_state = np.hstack([ee_pose, other_rod_pose, target_rod_pose])
    params =  [0.6, 0.1, 0.02, 0.2]
    cond = np.hstack([input_state, params])
    res = sem.sample(cond, 0)
    ee_diff, other_rod_diff, target_rod_diff = _extract_diffs(res)
    computed_cost = res['x_hats'][-1][-2]
    true_cost = np.linalg.norm([ee_pose[:2]-params[:2]]) + abs(ee_pose[2]-params[2])
    desired_ee_diff = params - ee_pose
    desired_ee_diff[3] = -params[3]- ee_pose[3]
    desired_target_rod_diff = desired_ee_diff.copy()
    assert(np.isclose(true_cost, computed_cost, atol=ATOL))
    assert np.allclose(ee_diff, desired_ee_diff, atol=ATOL)
    assert np.allclose(target_rod_diff, desired_target_rod_diff, atol=ATOL)
    assert np.allclose(other_rod_diff, 0, atol=ATOL)


def test_place_in_contact_with_rotate_and_offset():
    sem = SEMAnalyticalRodsAndRobot(2)
    other_rod_pose = [-1.6,0.4,0.02,0.3] #somewhat far away
    target_rod_pose = np.array([0.5,0.2,0.02,0.3]) #close
    ee_pose = target_rod_pose.copy()
    ee_pose[3] = target_rod_pose[-1] + np.pi/2
    x_diff = 0.01
    y_diff = -0.03
    ee_pose[1] += y_diff
    ee_pose[0] += x_diff
    input_state = np.hstack([ee_pose, other_rod_pose, target_rod_pose])
    params =  [0.6, 0.1, 0.02, 0.2]
    cond = np.hstack([input_state, params])
    res = sem.sample(cond, 0)
    ee_diff, other_rod_diff, target_rod_diff = _extract_diffs(res)
    #distance and relative angle the same
    distance_before = np.linalg.norm(ee_pose[:2] - target_rod_pose[:2])
    ee_pose_after = ee_pose + ee_diff
    target_pose_after = target_rod_pose + target_rod_diff
    distance_after =  np.linalg.norm(ee_pose_after[:2]-target_pose_after[:2])
    angle_before = (ee_pose[3]-target_rod_pose[3]) % (2*np.pi)
    angle_after = (ee_pose_after[3]-target_pose_after[3]) % (2*np.pi)
    assert np.allclose(distance_before, distance_after, atol=ATOL)
    assert np.allclose(angle_before, angle_after, atol=ATOL)
    assert np.allclose(other_rod_diff, 0, atol=ATOL)
    computed_cost = res['x_hats'][-1][-2]
    true_cost = np.linalg.norm([ee_pose[:2]-params[:2]]) + abs(ee_pose[2]-params[2])
    assert(np.isclose(true_cost, computed_cost, atol=ATOL))


def _extract_diffs(res):
    ee_diff = res['x_hats'][-1][0:4]
    rod1_diff = res['x_hats'][-1][4:8]
    rod2_diff = res['x_hats'][-1][8:12]
    return ee_diff, rod1_diff, rod2_diff
