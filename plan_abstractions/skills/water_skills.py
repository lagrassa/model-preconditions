import numpy as np

from plan_abstractions.skills.skills import Skill
from plan_abstractions.controllers.water_controllers import WaterTransportController, PourController


class WaterTransport2D(Skill):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.param_shape = (2,)
        self._terminate_on_timeout=True
        self.total_horizon = 10
        self._position_tol = 0.005
        self._no_water = 0.005
        self._min_above_dist = 0.01
        self._min_near_cup_dist = 0.05

    def pillar_state_to_internal_state(self, state):
        return state

    def state_precondition_satisfied(self, state):
        #The most important: is it 1) above the other cup 2) within enough distance?
        if state[-1] < self._no_water:
            return False
        height_control_cup = state[1]
        height_target = state[7]
        pos_control = state[0]
        pos_target = state[6]
        if height_control_cup - height_target < self._min_above_dist:
            return False
        if abs(pos_control-pos_target) > self._min_near_cup_dist:
            return False
        print("State precondition true")
        return True

    def precondition_satisfied(self, state, parameters, check_valid_goal_state=False, env=None):
        return True

    def _gen_object_centric_parameters(self, env, state):
        # Should be irrelevant, raise an error if sampled
        while True:
            yield np.array([0.1])

    def _gen_random_parameters(self, env, state):
        while True:
            #random_dist = np.random.uniform(low=0.1, high=0.2)
            random_dist_x = np.random.uniform(low=0.05, high=0.4)
            random_dist_z = np.random.uniform(low=0.05, high=0.2)
            random_dist = np.array([random_dist_x, random_dist_z])
            curr_state = state[0:2]
            yield curr_state + random_dist

    def _gen_relation_centric_parameters(self, env, state):
        # Should be irrelevant, raise an error if sampled
        while True:
            yield
            raise NotImplementedError


    def apply_action(self, env, env_idx, action):
        assert env_idx == 0
        env.save_action(action)

    def make_controllers(self, initial_states, parameters, T_plan_max=1, t=0, dt=0.01,total_horizon=5, real_robot=False, avoid_obstacle_height=True):
        info_plans = []
        controllers = []
        for env_idx, initial_state in enumerate(initial_states):
            controller = WaterTransportController()
            goal_pose = parameters[env_idx]
            info_plan = controller.plan(curr_pos = initial_state[0:2], goal_pose = goal_pose, total_horizon = total_horizon)
            controllers.append(controller)
            info_plans.append(info_plan)
        return controllers, info_plans

    def check_termination_condition(self, internal_state, parameters, t, controller=None, env_idx=None):
        timeout = False
        if controller is not None and np.linalg.norm(internal_state[0:2] - controller.goal_pose) <  self._position_tol:
            return True
        if self._terminate_on_timeout and controller is not None:
            timeout = t >= controller.horizon
        return timeout



class Pour(Skill):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.param_shape = (3,)
        self._terminate_on_timeout=True
        self.total_horizon = 10
        self._position_tol = 0.005

    def pillar_state_to_internal_state(self, state):
        return state

    def state_precondition_satisfied(self, state):
        return True

    def precondition_satisfied(self, state, parameters, check_valid_goal_state=False, env=None):
        return True

    def _gen_object_centric_parameters(self, env, state):
        # Should be irrelevant, raise an error if sampled
        while True:
            yield np.array([0.1])

    def _gen_random_parameters(self, env, state):
        while True:
            #random_dist = np.random.uniform(low=0.1, high=0.2)
            random_theta = np.random.uniform(low=0.05, high=0.4)
            yield np.array([random_theta])

    def _gen_relation_centric_parameters(self, env, state):
        # Should be irrelevant, raise an error if sampled
        while True:
            yield
            raise NotImplementedError


    def apply_action(self, env, env_idx, action):
        assert env_idx == 0
        env.save_action(action)

    def make_controllers(self, initial_states, parameters, T_plan_max=1, t=0, dt=0.01,total_horizon=5, real_robot=False, avoid_obstacle_height=True):
        info_plans = []
        controllers = []
        for env_idx, initial_state in enumerate(initial_states):
            controller = PourController()
            goal_angle = parameters[env_idx][0]
            info_plan = controller.plan(curr_pose = initial_state[:2], curr_angle = initial_state[2], goal_angle = goal_angle, total_horizon = total_horizon)
            controllers.append(controller)
            info_plans.append(info_plan)
        return controllers, info_plans

    def check_termination_condition(self, internal_state, parameters, t, controller=None, env_idx=None):
        timeout = False
        if controller is not None and np.linalg.norm(internal_state[0:2] - controller.goal_pose) <  self._position_tol:
            return True
        if self._terminate_on_timeout and controller is not None:
            timeout = t >= controller.horizon
        return timeout



