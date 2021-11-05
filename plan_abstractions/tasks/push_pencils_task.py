import logging

import numpy as np

from .base_task import BaseTask
from ..utils import point_in_box, yaw_from_quat, np_to_quat, get_pose_pillar_state, \
    states_similar_within_tol
from ..skills import FreeSpaceLQRMove, FreeSpacePDMove, LQRWaypointsXYYaw
from ..utils.utils import pretty_print_array

logger = logging.getLogger(__name__)


class PushRodsTask(BaseTask):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._pos_same_tol = cfg['goal']['position_tol']
        self._yaw_same_tol = cfg['yaw_same_tol']
        if cfg['goal']['randomize']:
            self._goal_pos = np.random.uniform(low=cfg["goal"]["goal_pose_ranges"]["low"],
                                               high=cfg["goal"]["goal_pose_ranges"]["high"])
        else:
            self._goal_pos = np.array(cfg['goal']['pos'])

    @property
    def goal_pos(self):
        return self._goal_pos

    @property
    def position_tol(self):
        return self._pos_same_tol

    def resample_goal(self, env=None):
        return self.resample_goal_from_range(self._cfg["goal"]["goal_pose_ranges"]["low"], 
                                             self._cfg["goal"]["goal_pose_ranges"]["high"])
    
    def resample_goal_from_range(self, low, high, env=None):
        assert self._cfg['goal']['randomize']
        old_goal_pos = self._goal_pos
        new_goal_pos = np.random.uniform(low, high)
        self._goal_pos = new_goal_pos
        return old_goal_pos, new_goal_pos

    def pillar_state_to_internal_state(self, pillar_state):
        rod0_pos = np.array(
            pillar_state.get_values_as_vec(["frame:rod0:pose/position"])
        )[:2]
        rod1_pos = np.array(
            pillar_state.get_values_as_vec(["frame:rod1:pose/position"])
        )[:2]
        return np.array([rod0_pos, rod1_pos])

    def states_similar(self, pillar_state_1, pillar_state_2):
        return states_similar_within_tol(pillar_state_1, pillar_state_2, self._pos_same_tol, self._yaw_same_tol)

    def is_goal_state(self, pillar_state):
        rod0_pos, rod1_pos = self.pillar_state_to_internal_state(
            pillar_state)
        desired_position = self._goal_pos

        position_tol = self._pos_same_tol

        pos_close = (
                      np.linalg.norm(rod0_pos - desired_position) < position_tol
                    ) and (np.linalg.norm(rod1_pos - desired_position) < position_tol)

        return pos_close

    def distance_to_goal_state(self, pillar_state):
        rod0_pos, rod1_pos = self.pillar_state_to_internal_state(pillar_state)
        desired_pos = self._goal_pos
        total_dist_to_goal = np.linalg.norm(rod0_pos - desired_pos) ** 2 + np.linalg.norm(
            rod1_pos - desired_pos) ** 2
        return np.sqrt(total_dist_to_goal)

    def is_valid_state(self, pillar_state, skills):
        # TODO
        # Check for the preconditions of all the skills
        return True

    def is_terminal_state(self, pillar_state, skills):
        return self.is_goal_state(pillar_state) or not self.is_valid_state(
            pillar_state, skills
        )

    def compute_edge_cost(self, pillar_state_1, pillar_state_2):
        pusher_pose1 = get_pose_pillar_state(pillar_state_1, "pusher")
        pusher_pose2 = get_pose_pillar_state(pillar_state_2, "pusher")
        return np.linalg.norm(np.array(pusher_pose1[:3]) - np.array(pusher_pose2[:3]))  # TODO add yaw?

    def evaluate(self, pillar_state):
        rod0_pos, rod1_pos = self.pillar_state_to_internal_state(
            pillar_state)
        desired_position = self._goal_pos

        error = np.linalg.norm(desired_position - rod0_pos) + np.linalg.norm(
            desired_position - rod1_pos
        )
        return np.linalg.norm(error)

    def pretty_print_goal_params(self):
        return pretty_print_array(self._goal_pos, prefix='Goal position') 
    
    def pretty_print_with_reference_to_pillar_state(self, pillar_state):
        goal_str = self.pretty_print_goal_params()
        rod0_pos, rod1_pos = self.pillar_state_to_internal_state(pillar_state)
        rod0_str = pretty_print_array(rod0_pos, 'Rod 0:')
        rod1_str = pretty_print_array(rod1_pos, 'Rod 1:')

        return f'     Goal: {goal_str}\n' \
               f' Rod 0: {rod0_str}\n' \
               f' Rod 1: {rod1_str}\n'


class PushRodsInBoxTask(BaseTask):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._pos_same_tol = cfg['goal']['position_tol']
        self._yaw_same_tol = cfg['yaw_same_tol']
        self._goal_dims = np.array(list(cfg['goal']['dims'].values()))
        self._setup_callbacks.append(lambda env, scene, env_idx: self.add_goal_box_to_env_cb(env, scene, env_idx))

        if cfg['goal']['randomize']:
            self._goal_pose = np.random.uniform(low=cfg["goal"]["goal_pose_ranges"]["low"],
                                                high=cfg["goal"]["goal_pose_ranges"]["high"])

        else:
            self._goal_pose = np.array(cfg['goal']['pose'] + [0, ])

        def free_space_lqr_gen(env, state):
            while True:
                pos = self.get_goal_positions_around_goal_region()
                x, y = pos
                parameters = np.array([x, y])
                yield parameters

        def free_space_pd_gen(env, state):
            curr_quat = np_to_quat(state.get_values_as_vec(['frame:pusher:pose/quaternion']), format="wxyz")
            curr_yaw = yaw_from_quat(curr_quat)
            while True:
                pos = self.get_goal_positions_around_goal_region()
                x, y = pos
                parameters = np.array([x, y, curr_yaw])
                yield parameters

        def lqr_waypoint_xyyaw_gen(env, state):
            curr_quat = np_to_quat(state.get_values_as_vec(['frame:pusher:pose/quaternion']), format="wxyz")
            curr_yaw = yaw_from_quat(curr_quat)
            while True:
                pos = self.get_goal_positions_around_goal_region()
                x, y = pos
                rods_to_push = np.random.randint(0, 2, env.num_rods)
                parameters = np.array([x, y, curr_yaw] + rods_to_push.tolist())
                yield parameters

        self._skill_specific_param_generators[FreeSpaceLQRMove.__name__] = free_space_lqr_gen
        self._skill_specific_param_generators[FreeSpacePDMove.__name__] = free_space_pd_gen
        self._skill_specific_param_generators[LQRWaypointsXYYaw.__name__] = lqr_waypoint_xyyaw_gen
    
    def resample_goal(self, env=None):
        return self.resample_goal_from_range(self._cfg["goal"]["goal_pose_ranges"]["low"], 
                                             self._cfg["goal"]["goal_pose_ranges"]["high"])
    
    def resample_goal_from_range(self, low, high, env=None):
        assert self._cfg['goal']['randomize']
        old_goal_pose = self._goal_pose
        # Note that goal centric generators(in __init__) should still be valid even after goal resampling.
        new_goal_pose = np.random.uniform(low=low, high=high)
        self._goal_pose = new_goal_pose

        if env is not None:
            env.reset_visual_box(self._goal_pose, self.goal_dims)

        return old_goal_pose, new_goal_pose

    @property
    def goal_pos(self):
        return self._goal_pose

    @property
    def goal_dims(self):
        return self._goal_dims

    def add_goal_box_to_env_cb(self, env, scene, env_idx):
        """
        Callback that adds a visual box to the environment.
        useful for debugging
        """
        env.add_visual_box_cb(self.goal_pos, self.goal_dims)

    def pillar_state_to_internal_state(self, pillar_state):
        rod0_pos = np.array(
            pillar_state.get_values_as_vec(["frame:rod0:pose/position"])
        )[:2]
        rod1_pos = np.array(
            pillar_state.get_values_as_vec(["frame:rod1:pose/position"])
        )[:2]
        return np.array([rod0_pos, rod1_pos])

    def is_goal_state(self, pillar_state):
        rod0_pos, rod1_pos = self.pillar_state_to_internal_state(
            pillar_state)
        res = point_in_box(rod0_pos[:2], self.goal_pos, self.goal_dims[:2]) and \
              point_in_box(rod1_pos[:2], self.goal_pos, self.goal_dims[:2])
        return res

    def states_similar(self, pillar_state_1, pillar_state_2):
        return states_similar_within_tol(pillar_state_1, pillar_state_2, self._pos_same_tol, self._yaw_same_tol)

    def distance_to_goal_state(self, pillar_state):
        rod0_pos, rod1_pos = self.pillar_state_to_internal_state(pillar_state)
        desired_pos = self._goal_pos[:2]
        total_dist_to_goal = np.linalg.norm(rod0_pos - desired_pos) ** 2 + np.linalg.norm(
            rod1_pos - desired_pos) ** 2
        return np.sqrt(total_dist_to_goal)

    def get_goal_positions_around_goal_region(self):
        return np.random.uniform(low=self.goal_pos[:2] - self.goal_dims[:2] / 2,
                                 high=self.goal_pos[:2] + self.goal_dims[:2] / 2)

    def is_valid_state(self, pillar_state, skills):
        # TODO
        # Check for the preconditions of all the skills
        return True

    def evaluate(self, pillar_state):
        rod0_pos, rod1_pos = self.pillar_state_to_internal_state(
            pillar_state)
        desired_position = self.goal_pos[:2]

        error = np.linalg.norm(desired_position - rod0_pos) \
                + np.linalg.norm(desired_position - rod1_pos)
        return np.linalg.norm(error)

    def pretty_print_goal_params(self):
        self.goal_pos[0], self.goal_pos[1], self.goal_dims[0], self.goal_dims
        goal_pos = pretty_print_array(self.goal_pos[:2], prefix='Goal pos')
        goal_dims = pretty_print_array(self.goal_dims[:2], prefix='Goal dims')
        return goal_pos + ', ' + goal_dims
    
    def pretty_print_with_reference_to_pillar_state(self, pillar_state):
        goal_str = self.pretty_print_goal_params()
        rod0_pos, rod1_pos = self.pillar_state_to_internal_state(pillar_state)
        rod0_str = pretty_print_array(rod0_pos, 'Rod 0:')
        rod1_str = pretty_print_array(rod1_pos, 'Rod 1:')

        return f'     Goal: {goal_str}\n' \
               f' Rod 0: {rod0_str}\n' \
               f' Rod 1: {rod1_str}\n'