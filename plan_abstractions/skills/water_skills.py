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
        self._no_water = 0.05

    def pillar_state_to_internal_state(self, state):
        return state

    def state_precondition_satisfied(self, state):
        if state[-1] < self._no_water:
            print("Water full constraint violated")
            return False
        return True

    def precondition_satisfied(self, state, parameters, check_valid_goal_state=False, env=None):
        return True

    def _gen_object_centric_parameters(self, env, state):
        # Samples parameters near the lip of the other object
        while True:
            height_target = state[7]
            glass_x = state[0]
            glass_distance = state[6] - state[0]
            noisy_object_centric_x = glass_distance - glass_x + np.random.uniform(low=0.2, high=0.5)
            noisy_object_centric_y = height_target + np.random.uniform(low=0.09, high=0.3)
            yield np.array([noisy_object_centric_x, noisy_object_centric_y])

    def _gen_random_parameters(self, env, state):
        while True:
            #random_dist = np.random.uniform(low=0.1, high=0.2)
            random_dist_x = np.random.uniform(low=0.02, high=0.2)
            random_dist_z = np.random.uniform(low=0.02, high=0.2)
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

    def make_controllers(self, initial_states, parameters, T_plan_max=1, t=0, dt=0.01,total_horizon=10, real_robot=False, avoid_obstacle_height=True):
        info_plans = []
        controllers = []
        for env_idx, initial_state in enumerate(initial_states):
            controller = WaterTransportController()
            goal_pos = parameters[env_idx]
            info_plan = controller.plan(curr_pos = initial_state[0:2], goal_pos = goal_pos, total_horizon = total_horizon)
            controllers.append(controller)
            info_plans.append(info_plan)
        return controllers, info_plans

    def check_termination_condition(self, internal_state, parameters, t, controller=None, env_idx=None):
        timeout = False
        if controller is not None and np.linalg.norm(internal_state[0:2] - controller.goal_pos) <  self._position_tol:
            return True
        if self._terminate_on_timeout and controller is not None:
            timeout = t >= controller.horizon
        return timeout



class Pour(Skill):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.param_shape = (1,)
        self._terminate_on_timeout=True
        self._termination_buffer_time=20
        self.total_horizon = 10
        self._position_tol = 0.005
        self._no_water = 0.05
        self._min_above_dist = 0.09
        self._target_overlap = 0.36
        self._max_volume = 0.8 #this should be a parameter
        self._min_near_cup_edge_dist = 0.06

    def pillar_state_to_internal_state(self, state):
        return state

    def state_precondition_satisfied(self, state):
        #The most important: is it 1) above the other cup 2) within enough distance?
        if state[-1] < self._no_water:
            #print("Water full constraint violated")
            return False
        height_control_cup = state[1]
        height_target = state[7]
        glass_x = state[0]
        pos_control = state[0] #x coordinate of box, starts at 0
        glass_distance = state[6]-state[0] 
        pos_target = glass_distance-glass_x  + self._target_overlap
        if height_control_cup - height_target < self._min_above_dist:
            return False
        if abs(pos_control-pos_target) > self._min_near_cup_edge_dist:
            return False
        print("State precondition true")
        return True

    def precondition_satisfied(self, state, parameters, check_valid_goal_state=False, env=None):
        return True

    def _gen_object_centric_parameters(self, env, state):
        # Should be irrelevant, raise an error if sampled
        while True:
            yield np.array([0.6])

    def _gen_random_parameters(self, env, state):
        while True:
            #random_dist = np.random.uniform(low=0.1, high=0.2)
            #random_theta = np.random.uniform(low=0.9, high=3.1)
            random_theta = np.random.uniform(low=0.8, high=2.5)
            yield np.array([random_theta])

    def _gen_relation_centric_parameters(self, env, state):
        # Should be irrelevant, raise an error if sampled
        while True:
            yield
            raise NotImplementedError


    def apply_action(self, env, env_idx, action):
        assert env_idx == 0
        env.save_action(action)

    def make_controllers(self, initial_states, parameters, T_plan_max=1, t=0, dt=0.01,total_horizon=20, real_robot=False, avoid_obstacle_height=True):
        info_plans = []
        controllers = []
        for env_idx, initial_state in enumerate(initial_states):
            controller = PourController()
            goal_angle = parameters[env_idx][0]
            info_plan = controller.plan(curr_pos = initial_state[:2], curr_angle = initial_state[2], goal_angle = goal_angle, max_volume=self._max_volume, total_horizon = total_horizon)
            controllers.append(controller)
            info_plans.append(info_plan)
        return controllers, info_plans

    def check_termination_condition(self, internal_state, parameters, t, controller=None, env_idx=None):
        timeout = False
        water_in_control_cup = internal_state[-1]
        water_in_target_cup = internal_state[-2]
        #print("water in control cup", water_in_control_cup)
        #print("water in target cup", water_in_target_cup)
        if water_in_target_cup < self._max_volume:
            return False
        if controller is not None and np.linalg.norm(internal_state[2] - controller.start_angle) <  self._position_tol:
            return True
        if self._terminate_on_timeout and controller is not None:
            timeout = t >= controller.horizon + self._termination_buffer_time
        return timeout



