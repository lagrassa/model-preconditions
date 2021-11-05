import pybullet as p
from autolab_core import RigidTransform
import numpy as np
from pyquaternion import Quaternion
import plan_abstractions.envs.pb_utils as ut
import os
from pillar_state import State

from isaacgym_utils.math_utils import project_to_line
from plan_abstractions.utils import get_pose_pillar_state

np.random.seed(0)

class FrankaKinematicsWorld():
    def __init__(self, visualize=False, root_dir= "", bullet=None, load_previous=False):
        if visualize:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setGravity(0,0,-9.8)
        self.in_hand = []
        self.steps_taken = 0
        self.z_transform = 0.1034 #do this property. This number comes from the gripper transform when offset=True
        self.elbow_joint = 3
        self.root_dir = root_dir
        self.setup_robot()
        self.setup_workspace()

    """
    spawns a franka arm, eventually a FrankaArm object
    """
    def setup_robot(self):
        #self.robot = p.loadURDF(os.environ["HOME"]+"/ros/src/franka_ros/franka_description/robots/model.urdf") #fixme, point somewhere less fragile
        #self.robot_old = p.loadURDF(self.root_dir + "assets/robots/model.urdf")
        self.robot = p.loadURDF(self.root_dir + "assets/robots/franka_panda_dynamics.urdf")
        ut.set_point(self.robot, (0,0,1e-2))
        start_joints = [2.28650215, -1.36063288, -1.4431576, -1.93011263,0.23962597,  2.6992652,  0.82820212,0.0,0.0]
        self.grasp_joint = 7 #panda_hand
        assert ut.get_link_name(self.robot, self.grasp_joint) == "panda_hand"
        p.changeDynamics(self.robot, -1, mass=0)
        ut.set_joint_positions(self.robot, ut.get_movable_joints(self.robot), start_joints)
        self.joints_to_plan_with = ut.get_movable_joints(self.robot)[:-2]

    """
    table with a hollow and solid cylinder on it
    """
    def setup_workspace(self):
        self.rod0 = ut.create_box(0.01, 0.17, 0.01, color=(1,0,0,1))
        self.rod1 = ut.create_box(0.01, 0.17, 0.01, color =(0.2, 0.9, 0.9, 1))

    def forward_kinematics(self, joint_positions):
        ut.set_joint_positions(self.robot, self.joints_to_plan_with, joint_positions )
        link_pose = ut.get_link_pose(self.robot, self.grasp_joint)
        link_pos = np.array(link_pose[0])
        link_pos[2] -= self.z_transform
        link_quat = np.hstack([link_pose[1][-1], link_pose[1][:-1]])
        link_pose_ig = np.hstack([link_pos, link_quat])
        return link_pose_ig


    def inverse_kinematics(self, ee_pose_input, rod0_pose=None, rod1_pose=None, width=0.02, max_iter=100, good_num_solutions=10):
        if rod0_pose is not None:
            ut.set_pose(self.rod0, ig_pose_to_pb_pose(rod0_pose))
            ut.set_pose(self.rod1, ig_pose_to_pb_pose(rod1_pose))
        ee_pose = ee_pose_input.copy()
        ee_pose[2] += self.z_transform
        pb_quat = tuple(ee_pose[4:]) + (ee_pose[3],)
        pos = (ee_pose[:3], pb_quat)
        #quick_sol = ut.inverse_kinematics(self.robot, self.grasp_joint, pos,movable_joints = self.joints_to_plan_with,null_space=None, ori_tolerance=0.03)
        ut.set_joint_positions(self.robot,  [8,9],[0.02, 0.02])
        quick_sol = None #the elbow cost is important
        collision = False
        if quick_sol is None:
            slow_sol = self._compute_IK(pos, self.joints_to_plan_with, max_iter = max_iter, good_num_solutions=good_num_solutions)
            self.set_joints(slow_sol)
            if ut.link_pairs_collision(self.robot, list(range(11)), self.rod0):
                print("COllides with rod0")
                collision = True
            if ut.link_pairs_collision(self.robot, list(range(11)), self.rod1):
                print("COllides with rod0")
            collision = True
            return slow_sol, collision
        else:
            return quick_sol

    def set_joints(self, joint_vals):
        ut.set_joint_positions(self.robot, ut.get_movable_joints(self.robot)[:len(joint_vals)], joint_vals)

    def _compute_IK(self, goal_pose, joints_to_plan_for, cur_pose=None, maintain_elbow = True, max_iter = 100, good_num_solutions=10):
        solutions = []
        lower = ut.get_min_limits(self.robot, joints_to_plan_for)
        upper = ut.get_max_limits(self.robot, joints_to_plan_for)
        rest = np.mean(np.vstack(
            [ut.get_min_limits(self.robot, joints_to_plan_for), ut.get_max_limits(self.robot, joints_to_plan_for)]),
            axis=0)
        rest = [ut.get_joint_positions(self.robot, ut.get_movable_joints(self.robot))]
        for i in range(max_iter):
            lower = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
            upper = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
            null_space = None
            end_conf = ut.inverse_kinematics(self.robot, self.grasp_joint, goal_pose, movable_joints=joints_to_plan_for, null_space=null_space) #end conf to be in the goal loc
            random_jts = []
            for i in range(7):
                random_jts.append(np.random.uniform(low=lower[i], high=upper[i]))
            self.set_joints(random_jts)
            #ranges = np.array(upper)-np.array(lower)#2*np.ones(len(joints_to_plan_for))
            #null_space = [lower, upper, ranges, [rest]]
            if end_conf is not None:
                solutions.append(end_conf)
            if len(solutions) > good_num_solutions:
                break
        costs = []
        for solution in solutions:
            #distance_cost = np.linalg.norm(np.array(cur_pose)-np.array(solution[:-2]))
            self.set_joints(solution[:-2])
            elbow_pose =  ut.get_link_pose(self.robot, self.elbow_joint)
            curr_elbow_xy = np.array(elbow_pose[0][:2])
            #elbow_cost = ut.quat_angle_between(elbow_pose[2], quaternion_about_axis(2))
            u0 = np.array(ut.get_link_pose(self.robot, 0)[0][:2])
            u1 = np.array(ut.get_link_pose(self.robot, 8)[0][:2])
            goal_elbow_xy = project_to_line(curr_elbow_xy, u0, u1)
            elbow_cost = np.linalg.norm(goal_elbow_xy - curr_elbow_xy)
            costs.append(elbow_cost)
        return solutions[np.argmin(costs)][:-2]

    def show_effects(self, end_states, pause = False):
        for state in end_states:
            try:
                state = State.create_from_serialized_string(state)
            except RuntimeError:
                print(" COunt not initialized from pilalr state")
            except TypeError:
                state = state
            ee_pose = get_pose_pillar_state(state, "franka:ee")
            rod0_pose = get_pose_pillar_state(state, "rod0")
            rod1_pose = get_pose_pillar_state(state, "rod1")
            ik_sol = self.inverse_kinematics(ee_pose_input=ee_pose, rod0_pose=rod0_pose, rod1_pose=rod1_pose, max_iter = 40, good_num_solutions=2)
            if pause:
                input("OK?")


def ig_pose_to_pb_pose(pos_array):
    pos = pos_array[:3]
    quat = tuple(pos_array[4:]) + (pos_array[3],)
    return (pos, quat)

def test_set_yaw(pw, yaw_deg):
    yaw = np.deg2rad(yaw_deg)
    T_camera_world = RigidTransform.load("/home/lagrassa/git/plan-abstractions/data/calibration/overheadparams/kinect2_overhead_to_world.tf")
    position = np.array([-0.05,0, 0.828])
    position = np.array([-0.15,0, 0.828])
    #position = np.array([0,0,0])
    input_rotation = RigidTransform.z_axis_rotation(yaw + np.pi/2)#-np.pi/2)
    #input_rotation = (RigidTransform(rotation=RigidTransform.x_axis_rotation(yaw)) * RigidTransform(rotation=RigidTransform.y_axis_rotation(np.pi/2))).rotation
    #input_rotation = np.array([0,1,0,0,0,-1,-1,0,0]).reshape((3,3))
    #quaternion = RigidTransform(rotation=RigidTransform.z_axis_rotation(yaw)).quaternion
    T_tag_camera = RigidTransform(rotation=input_rotation, translation=position)
    T_tag_camera.to_frame="kinect2_overhead"
    T_tag_world = T_camera_world * T_tag_camera
    #rod0_pos = np.hstack([position, quaternion])
    rod0_pos = np.hstack([T_tag_world.translation, T_tag_world.quaternion])
    ut.set_pose(pw.rod0, ig_pose_to_pb_pose(rod0_pos))


def test_ik_and_fk(pw):
    ee_pose2 = [0.46572980284690857, 0.06284715235233307, 0.009861186146736145, -0.00021559835295192897,
                0.9846376180648804, -0.1746046394109726, -0.0013848436065018177]
    joint_positions2 = [0.1344347596168518, 0.5030147433280945, 0.0003378019609954208, -2.436507225036621,
                        0.005740350112318993, 2.9372458457946777, 1.2654527425765991]
    joint_positions3 = [-0.11227073520421982, 0.7405218482017517, -0.0002620779268909246, -1.9120864868164062,
                        0.0018544427584856749, 2.651613473892212, 1.1806566715240479]
    ee_pose3 = np.array(
        [0.6111786365509033, -0.0690915510058403, 0.009850315749645233, -4.9290043534711e-05, 0.9677436947822571,
         -0.25193583965301514, -0.0006004265160299838])

    joint_positions = np.array(
        [0.13464535772800446, 0.5029193162918091, 8.78758801263757e-05, -2.4366228580474854, 0.006471884436905384,
         2.937086820602417, 1.2647905349731445])
    ee_pose = np.array(
        [0.46572980284690857, 0.06284715235233307, 0.009861186146736145, -0.00021559835295192897, 0.9846376180648804,
         -0.1746046394109726, -0.0013848436065018177])
    fk_result = pw.forward_kinematics(joint_positions3)
    print("Fk result", fk_result)
    print("EE pose", ee_pose3.round(2))
    print("FK distance", np.linalg.norm(ee_pose3[:3] - fk_result[:3]))
    ik_res, collision = pw.inverse_kinematics(ee_pose)
    print("IK res error", np.linalg.norm(np.array(ik_res) - np.array(joint_positions3)))

def test_yaw_rod_setting(pw):
    position = [0.4, 0, 0.02]
    yaw = np.deg2rad(-88)
    quaternion = RigidTransform.z_axis_rotation(yaw).quaternion
    rod0_pose = np.hstack([position, quaternion])
    ut.set_pose(self.rod0, ig_pose_to_pb_pose(rod0_pose))
    import ipdb; ipdb.set_trace()




if __name__ == "__main__":
    #pw = PegWorld( visualize=False, handonly/do=False)
    pw = FrankaKinematicsWorld(visualize=True, load_previous=False)
    #test_ik_and_fk(pw)
    test_yaw_rod_setting(pw)
    #end_states = np.load("data/pillar_state_recons_pb.npy")
    #pw.show_effects(end_states)

