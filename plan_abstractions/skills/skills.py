from abc import ABC, abstractmethod
import logging

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union
from pillar_state import State
from tqdm import trange
import quaternion

from isaacgym_utils.math_utils import np_to_quat
from .skill_exec_cb import SkillExecCb
from ..envs import  FrankaRodEnv
from ..models import create_sem_wrapper_from_cfg, create_deviation_wrapper_from_cfg
from ..models.model import SEMModel, SimModel
try:
    from ..models.low_level_models import FreeSpaceMoveModel, FixedRelativeTransformsOnContactModel
except (ModuleNotFoundError, ImportError):
    print("Unable to import low level models")
from ..utils import yaw_from_quat, pillar_state_to_shapes, yaw_from_np_quat, pillar_state_obj_to_transform, \
    xy_yaw_to_transform, transform_to_xy_yaw, get_pose_pillar_state, angle_axis_between_quats, create_param_ddist, \
    is_pos_B_btw_A_and_C, pretty_print_param_infos

logger = logging.getLogger(__name__)


class Skill(ABC):

    def __init__(self, param_dist_cfg=None,
                 cache_dir="/tmp",
                 sem_cfg=None,
                 deviation_cfg=None,
                 replan_cfg=None,
                 models_cfg=None,
                 low_level_models_cfg=None,
                 use_delta_actions=False,
                 real_robot=False):
        self._has_sem_model = False
        self._should_reset_to_viewable = False
        self._use_delta_actions = use_delta_actions
        if param_dist_cfg is not None:
            self._param_ddist = create_param_ddist(param_dist_cfg)
            self._param_types = param_dist_cfg.keys()
        else:
            default_param_dist_dict = {"object_centric": 0.9, "relation_centric": 0, "random": 0.1, "task_oriented": 0}
            self._param_ddist = create_param_ddist(default_param_dist_dict)
            self._param_types = default_param_dist_dict.keys()
        if sem_cfg is not None:
            self._has_sem_model = True
            self._sem_wrapper = create_sem_wrapper_from_cfg(sem_cfg, skill_cls=self.__class__, sem_state_obj_names=sem_cfg.get("sem_state_obj_names", None), cache_dir=cache_dir)

        if deviation_cfg is not None:
            self._has_deviation_model = True
            self._deviation_wrapper = create_deviation_wrapper_from_cfg(deviation_cfg, cache_dir=cache_dir)
            self._is_deviation_classification = deviation_cfg.get("is_classification", False)
            if not self._is_deviation_classification:
                self._acceptable_deviation = deviation_cfg["acceptable_deviation"]
            self._acceptable_deviation = deviation_cfg["acceptable_deviation"]
            self._beta = deviation_cfg.get("beta", None)
            self._param_rejection_sampling = deviation_cfg.get("param_rejection_sampling", False)
        else:
            self._param_rejection_sampling = False
        self._high_level_models = []
        if models_cfg is not None:
            for model_name, model_cfg in models_cfg.items():
                model_type = model_cfg["type"]
                if model_type == "SEMModel":
                    new_model = eval(model_type)(model_cfg, cache_dir=cache_dir)
                elif model_type == "SimModel":
                    new_model = eval(model_type)(model_cfg)
                else:
                    raise ValueError("Invalid model type")

                self._high_level_models.append(new_model)
            self._has_deviation_model = False
        self._low_level_models = []
        if low_level_models_cfg is not None:
            for low_level_model_type, low_level_model_cfg in low_level_models_cfg.items():
                new_low_level_model = eval(low_level_model_type)(low_level_model_cfg)
                self._low_level_models.append(new_low_level_model)


        self._do_replan = False
        self._replan_interval = -1
        if replan_cfg is not None:
            self._do_replan = replan_cfg['use']
            self._replan_interval = replan_cfg['interval']

        self.num_discrete_states = None
        self.param_shape = None

    @property
    def has_sem_model(self):
        return self._has_sem_model


    @property
    def param_types(self):
        return self._param_types

    @property
    def do_replan(self):
        return self._do_replan

    @property
    def replan_interval(self):
        return self._replan_interval

    @abstractmethod
    def pillar_state_to_internal_state(self, state):
        """
        converts state into lower dimensional representation that can be used by the DynamicsModel
        likely by indexing the relevant dimensions from pillar_state_py object
        """
        pass

    def state_precondition_satisfied(self, state):
        """
        Checks preconditions that depend only on the initial state and not the
        skill parameters.
        """
        return True

    @abstractmethod
    def precondition_satisfied(self, state, parameters, check_valid_goal_state=False, env=None):
        pass

    def effects(self, state, parameters):
        """
        given state returns next state after applying this action with the parameters
        uses SEM, used for high level planner
        parameters is an NxM array where N is the number of environments to rollout forward model on
        and M is the dimensionality of the parameters
        """
        assert self.has_sem_model
        sem_effects = self._sem_wrapper(state, parameters)
        effects = self._post_process_sem_effects(sem_effects)

        return effects

    def effects_batch(self, states, parameters_matrix):
        # TODO(jacky): do explicitly batching w/ neural network
        effects_list = [
            self.effects(state, parameters)
            for state, parameters in zip(states, parameters_matrix)
        ]
        effects = {k: [] for k in effects_list[0]}
        for i in range(len(effects_list)):
            for k in effects:
                effects[k].append(effects_list[i][k])

        return effects

    def multiple_model_effects(self, env, states, params, T_plan_max, T_exec_max, model_idxs, pb_env, model_idx_to_eval=None):
        """

        Args:
            env: env to use for sims
            states: start states
            params: params
            model_idx: model idxs to use for each state, param pair

        Returns: effects where each was run using a particular model idx

        """
        effects_all = {}
        effects_all["end_states"] = [None for _ in states] 
        effects_all["costs"] = np.array([0]*len(states), dtype=object)
        effects_all["T_exec"] = np.array([0]*len(states), dtype=object)
        effects_all["info_plan"] = np.array([{}]*len(states), dtype=object)
        collision_eps = 0.002
        asset_name_to_eps_arr = {"finger_left": [2 * collision_eps, 2 * collision_eps],"finger_right": [2 * collision_eps, 2 * collision_eps],"rod": [collision_eps, collision_eps]}
        debug = False
        possible_model_idxs = [None] + list(range(len(self.high_level_models)))
        for model_idx in possible_model_idxs:
            if model_idx_to_eval is not None and model_idx != model_idx_to_eval:
                continue
            if model_idx is None:
                continue #not useful
            idx_mask = model_idxs == model_idx
            states_for_model_idx = np.array(states)[idx_mask]
            if not isinstance(params, np.ndarray):
                params = np.array(params)
            params_for_model_idx = params[idx_mask]
            if len(states_for_model_idx) == 0:
                continue
            effects = self.high_level_models[model_idx].apply(states_for_model_idx, params_for_model_idx, env, T_plan_max, T_exec_max, self, pb_env)
            if effects == -1:
                return -1
            if debug:
                body_names = ["franka:finger_left", "franka:finger_right", "rod0", "rod1"]
                franka_cls = FrankaRodEnv
                for end_state in effects["end_states"]:
                    if isinstance(end_state, bytes):
                        end_state = State.create_from_serialized_string(end_state)
                    franka_cls.is_in_collision(end_state, body_names = body_names, asset_name_to_eps_arr=asset_name_to_eps_arr, plot=1)

            if debug:
                pb_env.show_effects(effects["end_states"])
            #add effects with that mask
            _set_end_states_given_mask(effects_all, effects, idx_mask)
            effects_all["costs"][idx_mask] = effects["costs"]
            if "T_exec" in effects.keys():
                effects_all["T_exec"][idx_mask] = effects["T_exec"]
            if "info_plan" in effects.keys():
                effects_all["info_plan"][idx_mask] = [plan_dict for plan_dict in effects["info_plan"]]
        return effects_all




    def _post_process_sem_effects(self, sem_effects):
        ''' Can overwrite this if special post processing is needed for child skill
        '''
        return sem_effects

    def gt_effects(self, env, state, parameters, T_plan_max, T_exec_max):
        return self.execute(env, state, parameters, T_plan_max, T_exec_max)

    @property
    def high_level_models(self):
        return self._high_level_models

    @property
    def low_level_models(self):
        return self._low_level_models

    @property
    def has_sem_model(self):
        return self._has_sem_model

    @property
    def sem_wrapper(self):
        return self._sem_wrapper

    def update_sem_wrapper(self, sem_wrapper):
        assert sem_wrapper is not None
        self._has_sem_model = True
        self._sem_wrapper = sem_wrapper

    def generate_parameters(self, env, state, num_parameters=1, task_oriented_sampler_gen=None, max_attempts=50,shuffle=True,
                            return_param_types=False, debug=False, valid_success_per_param_type=None, 
                            check_valid_goal_state=True):
        """
        given state returns generator of valid parameters according to a pre-specified distribution
        will try for max_attempts before throwing a stopiteration

        @param valid_success_per_param_type a dictionary of all possible param types that might be encountered while sampling
        from the distribution for this function. It will be mutated.
        @debug if true, success_per_param type statistics will be printed each time

        Use cases:
        1) for old behaviour, leave debug parameters as defaults
        2) To log without printing, pass in valid_success_per_param_type but leave debug=False
        3) To print without saving, debug=True, leave valid_success_per_param_type as default
        4) To print and save, debug=True, and pass in valid_success_per_param_type
        """
        if self.state_precondition_satisfied(state):
            internal_state = self.pillar_state_to_internal_state(state)
            param_gens = {
                'object_centric': self._gen_object_centric_parameters(env, state),
                'relation_centric': self._gen_relation_centric_parameters(env, state),
                'random': self._gen_random_parameters(env, state),
                'task_oriented': task_oriented_sampler_gen
            }
            param_types = [None] * num_parameters
            if valid_success_per_param_type is None:
                valid_success_per_param_type = {k: [] for k in param_gens}
            while True:
                parameter_array = np.zeros((num_parameters,) + self.param_shape)
                num_valid_params_found = 0
                num_param_attempts = 0
                while num_valid_params_found < num_parameters:
                    if num_param_attempts > max_attempts:
                        logging.info(f"Skill {self.__class__.__name__} | Max param gen attmpt reached: {max_attempts}. Returning params found so far")
                        if return_param_types:
                            yield parameter_array[:num_valid_params_found], param_types[:num_valid_params_found]
                        else:
                            yield parameter_array[:num_valid_params_found]
                        raise StopIteration

                    if num_param_attempts % 50 == 0:
                        logging.debug(f"Num attmepts to gen params: {num_param_attempts}")

                    param_type = self._param_ddist.sample()
                    parameter_option = next(param_gens[param_type])

                    if self._param_rejection_sampling and self.predicted_to_deviate(state, parameter_option):
                        valid_success_per_param_type[param_type].append(0)
                    elif self.precondition_satisfied(state, parameter_option,
                                                     check_valid_goal_state=check_valid_goal_state, env=env) \
                            and not self.check_termination_condition(internal_state, parameter_option, 0):

                        parameter_array[num_valid_params_found, :] = parameter_option
                        param_types[num_valid_params_found] = param_type
                        num_valid_params_found += 1
                        valid_success_per_param_type[param_type].append(1)
                    else:
                        valid_success_per_param_type[param_type].append(0)


                    num_param_attempts += 1
                

                if debug:
                    pretty_print_param_infos(valid_success_per_param_type)
                order = np.arange(len(parameter_array))
                random_order = np.random.permutation(order)
                if return_param_types:
                    yield parameter_array[random_order], np.array(param_types)[random_order]
                else:
                    yield parameter_array[random_order]
        else:
            if return_param_types:
                yield [], []
            else:
                yield []

    @abstractmethod
    def _gen_random_parameters(self, env, state):
        """
        totally random parameter
        """
        pass

    def _gen_relation_centric_parameters(self, env, state):
        """
        Complete list of relation_centric parameters
        """
        pass

    def predicted_to_deviate(self, state, parameters):
        assert self._has_deviation_model
        predicted_deviation, dev_stdev = self._deviation_wrapper.predict_from_pillar_state(state, parameters, is_classification = self._is_deviation_classification, with_conf=1)
        #print("predicted dev", predicted_deviation)
        if self._is_deviation_classification:
            return predicted_deviation
        elif self._beta is not None:
            return predicted_deviation + self._beta*dev_stdev > self._acceptable_deviation 
        else:
            return predicted_deviation > self._acceptable_deviation

    @abstractmethod
    def _gen_object_centric_parameters(self, env, state):
        """
        Returns: complete list of object centric parameters
        """
        pass

    @abstractmethod
    def make_controllers(self, initial_states, parameters, T_plan_max, t, dt):
        """

        :param state: N states
        :param parameters: N x M list of parameters
        :return:  (controllers (list), plan_infos (list))
        """
        pass

    @abstractmethod
    def check_termination_condition(self, internal_state, parameters, t, controller=None, env_idx=None):
        """
        Args:
            internal_state:  State to check if terminated
            parameters: skill parameters
            t: execution time step
            controller
            env_idx

            The controller and env_idx args are optional

        Returns:
            Whether the skill should terminate for that step
        """
        pass

    @abstractmethod
    def apply_action(self, env, env_idx, action):
        """

        Args:
            env: IG environment  to step
            env_idx: idx within the IG env
            action: action in whatever action space is meaningful to the skill and comes from the policy

        Returns:
            infos {}

        """
        pass

    def unpack_parameters(self, parameters):
        '''
        Returns N arguments that represent semantically useful parts of the parameters. Utility function
        parameters are M x 1 vector
        ex) A, B, Q, R, T  = skill.unpack_parameters(parameters) for LQR
        goal_pose, goal_velocity = skill.unpack_parameters(parameters) for a general position/velocity controller
        '''
        pass

    def get_exec_cb(self, *args, **kwargs):
        return SkillExecCb(self, *args, **kwargs)

    def execute(self, env, initial_states, parameters, T_plan_max, T_exec_max=1002, plot=False, dims_to_plot=None,
                set_state=True,
                save_low_level_transitions=False):
        """
        Runs the skill starting at all paralell envs until termination, returns a dict w/ recorded exec data
        Args:
            env:
            initial_states: initial state list of N states
            parameters: parameters N x M
            T_plan_max: Maximum amount of time allowed for planning
            dims_to_plot: np.array of dimensions that should be plotted

        Returns:
            Dict w/ data of all parallel skill executions
        """
        print("Executing", self, parameters)
        assert len(initial_states) <= env.n_envs
        assert not save_low_level_transitions
        env_idxs = np.arange(len(initial_states))

        if set_state:
            env.set_all_states(initial_states, env_idxs=env_idxs, n_steps=10)

        initial_settled_states = env.get_all_states(env_idxs=env_idxs)
        controllers, info_plans = self.make_controllers(initial_settled_states, parameters, T_plan_max, 0, dt=env.dt, real_robot=env._real_robot)

        data = {
            'end_states': [None] * len(env_idxs),
            'low_level_states': [[] for idx in env_idxs],
            'low_level_actions':[[] for idx in env_idxs],
            'T_exec': np.zeros(len(env_idxs)),
            'terminated': [False] * len(env_idxs),
            'costs': np.zeros(len(env_idxs)),
            'info_plan': info_plans,
            'dt': env.dt,
            'initial_settled_states': initial_settled_states,
        }

        if plot:
            log_incurred_states = []
            log_actions = []
        total_costs = np.zeros(len(env_idxs))
        for t in range(T_exec_max): #, desc=f"Executing Skill {self.__class__.__name__}", leave=False):
            log_incurred_states_per_t = [None] * len(env_idxs)
            log_actions_per_t = [None] * len(env_idxs)
            for env_idx in env_idxs:
                if data['terminated'][env_idx]:
                    continue

                if t == 0 and hasattr(env, 'set_skill_params'):
                    env.set_skill_params(env_idx, parameters[env_idx])
                pillar_state = env.get_sem_state(env_idx)
                #internal_state = self.pillar_state_to_internal_state(pillar_state)
                internal_state = pillar_state
                controller = controllers[env_idx]
                if self.check_termination_condition(internal_state, parameters[env_idx], t, controller, env_idx):
                    if env._real_robot:
                        data['end_states'][env_idx] = env.get_end_state(should_reset_to_viewable = self._should_reset_to_viewable).get_serialized_string()
                    else:
                        data['end_states'][env_idx] = pillar_state #pillar_state.get_serialized_string()
                    data['terminated'][env_idx] = True
                    #assert len(data["low_level_states"][env_idx]) == int(t)
                    data['T_exec'][env_idx] = t
                    data['costs'][env_idx] = total_costs[env_idx]
                    logger.debug(f"      Env {env_idx:} Termination condition met")
                    desired_position = self.unpack_parameters(parameters[env_idx])
                    logger.debug(f"       pusher pos: {internal_state[:2]}")
                    logger.debug(f"       desired pos: {desired_position}")
                    continue
                if save_low_level_transitions and not data['terminated'][env_idx]:
                    data["low_level_states"][env_idx].append(pillar_state.get_serialized_string())

                action = controller(internal_state, t, delta=self._use_delta_actions)
                if save_low_level_transitions and not data['terminated'][env_idx]:
                    data["low_level_actions"][env_idx].append(action)
                self.apply_action(env, env_idx, action)
                if plot:
                    log_incurred_states_per_t[env_idx] = internal_state
                    log_actions_per_t[env_idx] = action

            if plot:
                log_incurred_states.append(log_incurred_states_per_t)
                log_actions.append(log_actions_per_t)


            if np.all(data['terminated']):
                break

            step_costs = env.step()[env_idxs]
            non_terminated_mask = np.logical_not(data['terminated'])
            total_costs[non_terminated_mask] += step_costs[non_terminated_mask]

            if self._do_replan and t % self._replan_interval == 0:
                current_pillar_states = env.get_all_states(env_idxs=env_idxs)
                # TODO(jacky): do something about the unused info_plans
                controllers, _ = self.make_controllers(current_pillar_states, parameters, T_plan_max, t, env.dt, env._real_robot)

        for env_idx, terminated in enumerate(data['terminated']):
            if not terminated:
                if env._real_robot:
                    pillar_state = env.get_end_state(should_reset_to_viewable = self._should_reset_to_viewable)
                else:
                    pillar_state = env.get_sem_state(env_idx)
                data['end_states'][env_idx] = pillar_state #.get_serialized_string()
                if save_low_level_transitions:
                    data["low_level_states"][env_idx].pop() #remove last element because this data field is only supposed to include
                #the first and up to the last
                data['costs'][env_idx] = total_costs[env_idx]
                data['T_exec'][env_idx] = t
                if save_low_level_transitions:
                    assert len(data["low_level_states"][env_idx]) == int(t)

        if plot and len(log_actions) > 0:
            self.plot_logged_info(dims_to_plot, env, internal_state, log_actions, log_incurred_states, parameters)

        return data

    def plot_logged_info(self, dims_to_plot, env, internal_state, log_actions, log_incurred_states, parameters):
        dim_internal_state = len(internal_state)
        dim_action = len(log_actions[0][0])
        num_states_to_show = len(dims_to_plot) if dims_to_plot is not None else dim_internal_state
        _, axes = plt.subplots(max(num_states_to_show, dim_action), 2, figsize=(8, 8))
        ts = np.arange(len(log_incurred_states))
        for i in range(dim_internal_state):
            if dims_to_plot is not None and i not in dims_to_plot:
                continue
            for env_idx in range(env.n_envs):
                values_to_plot = [data[env_idx][i] for t, data in enumerate(log_incurred_states)
                                  if data[env_idx] is not None
                                  and None not in data[env_idx]]
                axes[i, 0].plot(ts[:len(values_to_plot)], values_to_plot, label=env_idx)

            idx_to_plot = i if dims_to_plot is None else dims_to_plot.index(i)
            axes[idx_to_plot, 0].set_ylabel(f"state {i}")
        for i in range(dim_action):
            for env_idx in range(env.n_envs):
                values_to_plot = [data[env_idx][i] for t, data in enumerate(log_actions)
                                  if data[env_idx] is not None and
                                  None not in data[env_idx]]
                axes[i, 1].plot(ts[:len(values_to_plot)], values_to_plot, label=env_idx)
            axes[i, 1].set_ylabel("action")
        # plotting goal xy
        axes[0, 0].plot(ts[:len(values_to_plot)], [parameters[0, 0]] * len(values_to_plot), label='goalx')
        axes[1, 0].plot(ts[:len(values_to_plot)], [parameters[0, 1]] * len(values_to_plot), label='goaly')
        plt.legend()
        plt.show()


class FreeSpacePDMove(Skill):

    def __init__(self, num_rods=2, **kwargs):
        super().__init__(**kwargs)
        self.num_discrete_states = 17  # fill in
        self.num_rods = num_rods
        self.param_shape = (3,)  # x,y,theta
        self.position_tol = 5e-3
        self.yaw_tol = np.deg2rad(10)

        self._max_goal_dist = 0.3
        self._max_goal_ang = np.pi

    def pillar_state_to_internal_state(self, state):
        curr_pos = np.array(state.get_values_as_vec(['frame:pusher:pose/position']))[:2]
        curr_quat = np_to_quat(state.get_values_as_vec(['frame:pusher:pose/quaternion']), format="wxyz")
        curr_yaw = yaw_from_quat(curr_quat)

        curr_lin_vel = np.array(state.get_values_as_vec(['frame:pusher:pose/linear_velocity']))[:2]
        curr_ang_vel = np.array(state.get_values_as_vec(['frame:pusher:pose/angular_velocity']))[2]
        return np.hstack([curr_pos, [curr_yaw], curr_lin_vel, [curr_ang_vel]])

    def precondition_satisfied(self, state, parameters, check_valid_goal_state=False, env=None):
        curr_pos = np.array(state.get_values_as_vec(['frame:pusher:pose/position']))[:2]
        curr_quat = np_to_quat(state.get_values_as_vec(['frame:pusher:pose/quaternion']), format="wxyz")
        curr_yaw = yaw_from_quat(curr_quat)

        close_dist = np.linalg.norm(parameters[:2] - curr_pos) < self._max_goal_dist
        close_angle = np.abs((parameters[2] - curr_yaw + np.pi) % (2 * np.pi) - np.pi) < self._max_goal_ang
        logger.debug(f"Skill: {self.__class__.__name__} close dist: {close_dist} close angle: {close_angle} ")

        return close_dist and close_angle

    def _gen_object_centric_parameters(self, env, state):
        goal_transforms = env.get_goal_transforms_around_objects(state, plot=False)
        while True:
            for i, goal_transform in enumerate(goal_transforms):
                x = goal_transform.p.x
                y = goal_transform.p.y
                theta = yaw_from_quat(goal_transform.r)
                parameters = np.array([x, y, theta])
                yield parameters

    def _gen_random_parameters(self, env, state):
        while True:
            curr_pos = np.array(get_pose_pillar_state(state, PushRodEnv.robot_name))
            curr_yaw = yaw_from_np_quat(curr_pos[3:])
            random_dir = np.random.uniform(low=0, high=2 * np.pi)
            random_dist = np.random.uniform(low=0, high=self._max_goal_dist)
            parameters = (random_dist * np.cos(random_dir), random_dist * np.sin(random_dir))
            parameters = np.hstack([curr_pos[:2] + parameters, np.random.uniform(low=curr_yaw - self._max_goal_dist,
                                                                                 high=curr_yaw + self._max_goal_dist)])
            yield parameters

    def _gen_relation_centric_parameters(self, env, state):
        while True:
            curr_pos = np.array(get_pose_pillar_state(state, PushRodEnv.robot_name))
            goal_transforms = env.get_goal_transforms_around_objects(state, in_only=True, plot=False)
            for i, goal_transform in enumerate(goal_transforms):
                x = goal_transform.p.x
                y = goal_transform.p.y
                des_theta = yaw_from_quat(goal_transform.r)
                curr_dist = np.linalg.norm(curr_pos[:2] - np.array([x, y]))
                if curr_dist > self._max_goal_dist:
                    random_dist = 0  # this parameter will fail the precondition check but that shouldn't be handled at this phase
                else:
                    random_dist = np.random.uniform(low=curr_dist, high=self._max_goal_dist - curr_dist)
                # go farther in that direction
                dir = np.arctan2((y - curr_pos[1]), (x - curr_pos[0]))
                parameters = np.array([x + random_dist * np.cos(dir), y + random_dist * np.sin(dir), des_theta])
                yield parameters

    @staticmethod
    def parameters_to_relative_parameters(parameters, ref_pillar_state, anchor_obj_name):
        a_T_w = pillar_state_obj_to_transform(ref_pillar_state, anchor_obj_name).inverse()

        w_T_o = xy_yaw_to_transform(parameters)
        a_T_o = a_T_w * w_T_o

        relative_parameters = transform_to_xy_yaw(a_T_o)
        return relative_parameters

    @staticmethod
    def relative_parameters_to_parameters(relative_parameters, ref_pillar_state, anchor_obj_name):
        w_T_a = pillar_state_obj_to_transform(ref_pillar_state, anchor_obj_name)

        a_T_o = xy_yaw_to_transform(relative_parameters)
        w_T_o = w_T_a * a_T_o

        parameters = transform_to_xy_yaw(w_T_o)
        return parameters

    def apply_action(self, env, env_idx, action):
        env.apply_force_torque_to_pusher(action, env_idx)

    def unpack_parameters(self, parameters):
        goal_pose = parameters[:2]
        goal_yaw = parameters[2]
        return np.array(goal_pose), goal_yaw

    def make_controllers(self, initial_states, parameters, T_plan_max, t, dt):
        info_plans = []
        controllers = []
        for parameter in parameters:
            desired_position, desired_yaw = self.unpack_parameters(parameter)
            controller = PositionController()
            info_plan = controller.plan(desired_position, desired_yaw)
            controllers.append(controller)
            info_plans.append(info_plan)
        return controllers, info_plans

    def check_termination_condition(self, internal_state, parameters, t, controller=None, env_idx=None):
        desired_position, desired_yaw = self.unpack_parameters(parameters)
        error_pos = desired_position - internal_state[:2]
        error_yaw = desired_yaw - internal_state[2]

        pos_close = np.isclose(np.linalg.norm(error_pos), 0, atol=self.position_tol)
        yaw_close = np.isclose(error_yaw, 0, atol=self.yaw_tol)

        lin_vel_zero = np.isclose(np.linalg.norm(internal_state[3:5]), 0, atol=2e-3)
        ang_vel_zero = np.isclose(np.abs(internal_state[5]), 0, atol=np.deg2rad(1))

        return pos_close and yaw_close and lin_vel_zero and ang_vel_zero


class FreeSpaceLQRMove(Skill):

    def __init__(self, num_rods=2, **kwargs):
        super().__init__(**kwargs)
        self.num_discrete_states = 17  # fill in
        self.num_rods = num_rods
        self.param_shape = (2,)
        self.position_tol = 5e-3
        self.yaw_tol = np.deg2rad(1)
        self._terminate_on_timeout = False

        self._max_goal_dist = 0.3
        self._p_random_parameter = 0.1

    def check_termination_condition(self, internal_state, parameters, t, controller=None, env_idx=None):
        desired_position = parameters
        error_pos = desired_position[:2] - internal_state[:2]
        pos_close = np.isclose(np.linalg.norm(error_pos), 0, atol=self.position_tol)

        lin_vel_zero = np.allclose(np.linalg.norm(internal_state[2:4]), 0, atol=1e-3)

        timed_out = False
        if self._terminate_on_timeout and env_idx is not None:
            timed_out = t >= self._traj_timesteps[env_idx]

        return (pos_close and lin_vel_zero) or timed_out

    def pillar_state_to_internal_state(self, state):
        pose = state.get_values_as_vec(['frame:pusher:pose/position'])[:2]
        pusher_vel = state.get_values_as_vec(['frame:pusher:pose/linear_velocity'])[:2]
        return np.hstack([pose, pusher_vel])

    def precondition_satisfied(self, state, parameters):
        curr_pos = np.array(state.get_values_as_vec(['frame:pusher:pose/position']))[:2]
        close_dist = np.linalg.norm(parameters[:2] - curr_pos) < self._max_goal_dist
        logger.debug(f"Skill: {self.__class__.__name__} close dist: {close_dist}")
        return close_dist

    def apply_action(self, env, env_idx, action):
        # TODO(jacky): LQR should add rotation
        env.apply_force_torque_to_pusher(np.hstack([action, [0]]), env_idx)

    def _gen_object_centric_parameters(self, env, state):
        goal_transforms = env.get_goal_transforms_around_objects(state, plot=False)
        while True:
            for i, goal_transform in enumerate(goal_transforms):
                x = goal_transform.p.x
                y = goal_transform.p.y
                parameters = np.array([x, y])
                yield parameters

    def _gen_random_parameters(self, env, state):
        while True:
            curr_pos = np.array(get_pose_pillar_state(state, PushRodEnv.robot_name))
            random_dir = np.random.uniform(low=0, high=2 * np.pi)
            random_dist = np.random.uniform(low=0, high=self._max_goal_dist)
            parameters = (
            curr_pos[0] + random_dist * np.cos(random_dir), curr_pos[1] + random_dist * np.sin(random_dir))
            yield parameters

    def _gen_relation_centric_parameters(self, env, state):
        curr_pos = np.array(get_pose_pillar_state(state, PushRodEnv.robot_name))
        goal_transforms = env.get_goal_transforms_around_objects(state, in_only=True, plot=False)
        while True:
            for i, goal_transform in enumerate(goal_transforms):
                x = goal_transform.p.x
                y = goal_transform.p.y
                curr_dist = np.linalg.norm(curr_pos[:2] - np.array([x, y]))
                if curr_dist > self._max_goal_dist:
                    random_dist = 0  # this parameter will fail the precondition check but that shouldn't be handled at this phase
                else:
                    random_dist = np.random.uniform(low=curr_dist, high=self._max_goal_dist - curr_dist)
                # go farther in that direction
                dir = np.arctan2((y - curr_pos[1]), (x - curr_pos[0]))
                parameters = np.array([x + random_dist * np.cos(dir), y + random_dist * np.sin(dir)])
                yield parameters

    @staticmethod
    def parameters_to_relative_parameters(parameters, ref_pillar_state, anchor_obj_name):
        a_T_w = pillar_state_obj_to_transform(ref_pillar_state, anchor_obj_name).inverse()

        xy_yaw = np.array([parameters[0], parameters[1], 0])
        w_T_o = xy_yaw_to_transform(xy_yaw)
        a_T_o = a_T_w * w_T_o

        relative_parameters = transform_to_xy_yaw(a_T_o)[:2]
        return relative_parameters

    @staticmethod
    def relative_parameters_to_parameters(relative_parameters, ref_pillar_state, anchor_obj_name):
        w_T_a = pillar_state_obj_to_transform(ref_pillar_state, anchor_obj_name)

        xy_yaw = np.array([relative_parameters[0], relative_parameters[1], 0])
        a_T_o = xy_yaw_to_transform(xy_yaw)
        w_T_o = w_T_a * a_T_o

        parameters = transform_to_xy_yaw(w_T_o)[:2]
        return parameters

    def unpack_parameters(self, parameters):
        return parameters

    def make_controllers(self, initial_states, parameters, T_plan_max, t, dt):
        info_plans = []
        controllers = []
        self._traj_timesteps = [None] * len(initial_states)
        for env_idx, initial_state in enumerate(initial_states):
            initial_internal_state = self.pillar_state_to_internal_state(initial_state)
            goal_pos = self.unpack_parameters(parameters[env_idx])
            mass = initial_state.get_values_as_vec(['constants/pusher_mass'])[0]
            dt = initial_state.get_values_as_vec(["constants/dt"])[0]
            controller = LQRWaypointControllerXYOld()
            info_plan = controller.plan(initial_internal_state, goal_pos, mass, dt)
            self._traj_timesteps[env_idx] = controller.horizon
            controllers.append(controller)
            info_plans.append(info_plan)
        return controllers, info_plans


class LQRWaypointsXY(Skill):

    def __init__(self, num_rods=2, **kwargs):
        super().__init__(**kwargs)
        self.num_discrete_states = 17  # fill in
        self.num_rods = num_rods
        self.param_shape = (2 + num_rods,)  # x,y, rods

        self.position_tol = 5e-3
        self.yaw_tol = np.deg2rad(1)
        self._terminate_on_timeout = False

        self._max_goal_dist = 0.3
        self._rod_perp_offset = 0.1
        self._yaw_diff_thresh = 30 * np.pi / 180

        self._p_random_parameter = 0.1

    def check_termination_condition(self, internal_state, parameters, t, controller=None, env_idx=None):
        goal_pos, _ = self.unpack_parameters(parameters)
        error_pos = goal_pos - internal_state[:2]
        pos_close = np.isclose(np.linalg.norm(error_pos), 0, atol=self.position_tol)

        lin_vel_zero = np.allclose(np.linalg.norm(internal_state[2:4]), 0, atol=1e-3)

        timed_out = False
        if self._terminate_on_timeout and env_idx is not None:
            timed_out = t >= self._traj_timesteps[env_idx]

        return (pos_close and lin_vel_zero) or timed_out

    def pillar_state_to_internal_state(self, state):
        pose = state.get_values_as_vec(['frame:pusher:pose/position'])[:2]
        pusher_vel = state.get_values_as_vec(['frame:pusher:pose/linear_velocity'])[:2]
        return np.hstack([pose, pusher_vel])

    def precondition_satisfied(self, state, parameters, check_valid_goal_state=False):
        """
        Args:
            state:
            parameters:
        Returns: The evaluation of 4 conditions:
            - if the goals are close enough to the current poses
            - if the rods are between the pusher and each goal, so it can perform the action with sweeping
            - yaw should be a resonable value 
        """
        goal_pos, rods_to_push = self.unpack_parameters(parameters)
        internal_state = self.pillar_state_to_internal_state(state)
        curr_pos = internal_state[:2]
        close_dist = np.linalg.norm(goal_pos - curr_pos) < self._max_goal_dist

        good_rods = []
        yaw_pusher_to_goal = np.arctan2(goal_pos[0] - curr_pos[0], goal_pos[1] - curr_pos[1])
        des_quat = quaternion.from_euler_angles([0, 0, yaw_pusher_to_goal - np.pi / 2])

        for i in range(rods_to_push.shape[0]):
            rod_pos = state.get_values_as_vec([f'frame:rod{rods_to_push[i]}:pose/position'])[:2]
            rod_quat = quaternion.from_float_array(
                state.get_values_as_vec([f'frame:rod{rods_to_push[i]}:pose/quaternion']))
            # checking if the rods are between the pusher and each goal
            in_between, projection = is_pos_B_btw_A_and_C(curr_pos, rod_pos, goal_pos)

            # perpendicular dist. of rod from line joining gripper and goal
            perp_dist = np.linalg.norm(rod_pos - projection)

            # yaw of rod realtive to the line joining gripper and goal
            _del = angle_axis_between_quats(rod_quat, des_quat)[2]
            yaw_delta = _del if abs(_del) < np.pi / 2 \
                else _del - np.sign(_del) * np.pi

            good_rods.append(
                (in_between and perp_dist < self._rod_perp_offset and yaw_delta < self._yaw_diff_thresh))

        logger.debug(f"Skill: {self.__class__.__name__} close dist: {close_dist} rods in goal path: {good_rods} ")
        return close_dist and np.all(good_rods)

    def apply_action(self, env, env_idx, action):
        env.apply_force_torque_to_pusher(np.hstack([action, [0]]), env_idx)

    def _gen_object_centric_parameters(self, env, state):
        goal_transforms = env.get_goal_transforms_around_objects(state, plot=False)
        while True:
            for i, goal_transform in enumerate(goal_transforms):
                x = goal_transform.p.x
                y = goal_transform.p.y
                rods_to_push = np.random.randint(0, 2, self.num_rods)
                parameters = np.append(np.array([x, y]), rods_to_push)
                yield parameters

    def _gen_random_parameters(self, env, state):
        while True:
            curr_pos = np.array(get_pose_pillar_state(state, PushRodEnv.robot_name))
            random_dir = np.random.uniform(low=0, high=2 * np.pi)
            random_dist = np.random.uniform(low=0, high=self._max_goal_dist)
            x = curr_pos[0] + random_dist * np.cos(random_dir)
            y = curr_pos[1] + random_dist * np.sin(random_dir)
            rods_to_push = np.random.randint(0, 2, self.num_rods)
            parameters = np.append(np.array([x, y]), rods_to_push)
            yield parameters

    def _gen_relation_centric_parameters(self, env, state):
        while True:
            curr_pos = np.array(get_pose_pillar_state(state, PushRodEnv.robot_name))
            goal_transforms = env.get_goal_transforms_around_objects(state, in_only=True, plot=False)
            for i, goal_transform in enumerate(goal_transforms):
                x = goal_transform.p.x
                y = goal_transform.p.y
                curr_dist = np.linalg.norm(curr_pos[:2] - np.array([x, y]))
                if curr_dist > self._max_goal_dist:
                    random_dist = 0  # this parameter will fail the precondition check but that shouldn't be handled at this phase
                else:
                    random_dist = np.random.uniform(low=curr_dist, high=self._max_goal_dist - curr_dist)
                # go farther in that direction
                dir = np.arctan2((y - curr_pos[1]), (x - curr_pos[0]))
                rods_to_push = np.random.randint(0, 2, self.num_rods)
                parameters = np.append(np.array([x + random_dist * np.cos(dir), y + random_dist * np.sin(dir)]),
                                       rods_to_push)

                yield parameters

    @staticmethod
    def parameters_to_relative_parameters(parameters, ref_pillar_state, anchor_obj_name):
        a_T_w = pillar_state_obj_to_transform(ref_pillar_state, anchor_obj_name).inverse()

        xy_yaw = np.array([parameters[0], parameters[1], 0])
        w_T_o = xy_yaw_to_transform(xy_yaw)
        a_T_o = a_T_w * w_T_o

        relative_parameters = np.hstack((transform_to_xy_yaw(a_T_o)[:2], parameters[2:]))
        return relative_parameters

    @staticmethod
    def relative_parameters_to_parameters(relative_parameters, ref_pillar_state, anchor_obj_name):
        w_T_a = pillar_state_obj_to_transform(ref_pillar_state, anchor_obj_name)

        xy_yaw = np.array([relative_parameters[0], relative_parameters[1], 0])
        a_T_o = xy_yaw_to_transform(xy_yaw)
        w_T_o = w_T_a * a_T_o

        parameters = np.hstack((transform_to_xy_yaw(w_T_o)[:2], relative_parameters[2:]))
        return parameters

    def unpack_parameters(self, parameters):
        goal_pos = parameters[:2]
        rods_to_push = np.where(parameters[2:] == 1)[0]
        return np.array(goal_pos), rods_to_push

    def make_controllers(self, initial_states, parameters, T_plan_max, t, dt):
        info_plans = []
        controllers = []
        self._traj_timesteps = [None] * len(initial_states)
        for env_idx, initial_state in enumerate(initial_states):
            internal_state = self.pillar_state_to_internal_state(initial_state)
            goal_pos, rods_to_push = self.unpack_parameters(parameters[env_idx])

            controller = LQRWaypointControllerXY()

            waypoints = np.zeros((4, rods_to_push.shape[0] + 1))
            waypoints[:2, -1] = goal_pos
            for i in range(rods_to_push.shape[0]):
                waypoints[:2, i] = initial_state.get_values_as_vec([f'frame:rod{rods_to_push[i]}:pose/position'])[
                                   :2]

            # sort waypoints
            dist_vec = np.linalg.norm(waypoints[:2, :-1] - internal_state[:2].reshape(2, 1), axis=0)
            indx = np.append(np.argsort(dist_vec), -1)
            waypoints = waypoints[:, indx]

            mass = initial_state.get_values_as_vec(['constants/pusher_mass'])[0]
            dt = initial_state.get_values_as_vec(["constants/dt"])[0]
            info_plan = controller.plan(internal_state, waypoints, dt, mass)
            self._traj_timesteps[env_idx] = controller.horizon
            controllers.append(controller)
            info_plans.append(info_plan)
        return controllers, info_plans


class LQRWaypointsXYYaw(Skill):

    def __init__(self, num_rods=2, **kwargs):
        super().__init__(**kwargs)
        self.num_discrete_states = 17  # fill in
        self.num_rods = num_rods
        self.param_shape = (3 + num_rods,)  # x,y

        self.position_tol = 5e-3
        self.yaw_tol = np.deg2rad(1)
        self._terminate_on_timeout = False

        self._max_goal_dist = 0.3
        self._max_goal_ang = np.deg2rad(60)

        self._rod_perp_offset = 0.1
        self._yaw_diff_thresh = np.deg2rad(30)

        self._p_random_parameter = 0.1

    def check_termination_condition(self, internal_state, parameters, t, controller=None, env_idx=None):

        goal_pos, goal_yaw, _ = self.unpack_parameters(parameters)
        error_pos = goal_pos - internal_state[:2]
        error_yaw = goal_yaw - internal_state[2]

        del_yaw = goal_yaw - internal_state[2]
        del_yaw = (del_yaw + np.pi) % (2 * np.pi) - np.pi  # wrap b/w -pi and pi
        error_yaw = np.linalg.norm(del_yaw if abs(del_yaw) < np.pi / 2 \
                                       else del_yaw - np.sign(del_yaw) * np.pi)

        pos_close = np.isclose(np.linalg.norm(error_pos), 0, atol=self.position_tol)
        yaw_close = np.isclose(error_yaw, 0, atol=self.yaw_tol)

        lin_vel_zero = np.isclose(np.linalg.norm(internal_state[3:5]), 0, atol=2e-3)
        ang_vel_zero = np.isclose(np.abs(internal_state[5]), 0, atol=np.deg2rad(1))

        timed_out = False
        if self._terminate_on_timeout and env_idx is not None:
            timed_out = t >= self._traj_timesteps[env_idx]

        return (pos_close and yaw_close and lin_vel_zero and ang_vel_zero) or timed_out

    def pillar_state_to_internal_state(self, state):
        curr_pos = np.array(state.get_values_as_vec(['frame:pusher:pose/position']))[:2]
        curr_quat = np_to_quat(state.get_values_as_vec(['frame:pusher:pose/quaternion']), format="wxyz")
        curr_yaw = yaw_from_quat(curr_quat)

        curr_lin_vel = np.array(state.get_values_as_vec(['frame:pusher:pose/linear_velocity']))[:2]
        curr_ang_vel = np.array(state.get_values_as_vec(['frame:pusher:pose/angular_velocity']))[2]
        return np.hstack([curr_pos, [curr_yaw], curr_lin_vel, [curr_ang_vel]])

    def precondition_satisfied(self, state, parameters, check_valid_goal_state=False):
        """
        Args:
            state:
            parameters:
        Returns: The evaluation of 4 conditions:
            - if the goals are close enough to the current poses
            - if the rods are between the pusher and each goal, so it can perform the action with sweeping
            - yaw from rods to pusher are not too big
            - yaw from goal to pusher are not too big
        """
        goal_pos, goal_yaw, rods_to_push = self.unpack_parameters(parameters)
        internal_state = self.pillar_state_to_internal_state(state)
        pusher_pos, pusher_yaw = internal_state[:2], internal_state[2]

        close_dist = np.linalg.norm(goal_pos - pusher_pos) < self._max_goal_dist
        close_angle = np.abs((goal_yaw - pusher_yaw + np.pi) % (2 * np.pi) - np.pi) < self._max_goal_ang

        yaw_pusher_to_goal = np.arctan2(goal_pos[0] - pusher_pos[0], goal_pos[1] - pusher_pos[1])
        des_quat = quaternion.from_euler_angles([0, 0, yaw_pusher_to_goal - np.pi / 2])

        good_rods = []
        for rod_idx in rods_to_push:
            rod_pos = state.get_values_as_vec([f'frame:rod{rod_idx}:pose/position'])[:2]
            rod_quat = quaternion.from_float_array(
                state.get_values_as_vec([f'frame:rod{rod_idx}:pose/quaternion']))

            # checking if the rods are between the pusher and each goal
            in_between, projection = is_pos_B_btw_A_and_C(pusher_pos, rod_pos, goal_pos)
            # perpendicular dist. of rod from line joining gripper and goal
            perp_dist = np.linalg.norm(rod_pos - projection)

            # yaw of rod realtive to the line joining gripper and goal
            _del = angle_axis_between_quats(rod_quat, des_quat)[2]
            _del = (_del + np.pi) % (2 * np.pi) - np.pi
            yaw_delta = _del if abs(_del) < np.pi / 2 else _del - np.sign(_del) * np.pi

            good_rods.append(
                in_between and perp_dist < self._rod_perp_offset and abs(yaw_delta) < self._yaw_diff_thresh
            )

        logger.debug(f"Skill: {self.__class__.__name__} close dist: {close_dist}  close angle: {close_angle} rods in goal path: {good_rods} ")
        return close_dist and close_angle and np.all(good_rods)

    def apply_action(self, env, env_idx, action):
        env.apply_force_torque_to_pusher(action, env_idx)

    def _gen_object_centric_parameters(self, env, state):
        goal_transforms = env.get_goal_transforms_around_objects(state, plot=False)
        while True:
            for i, goal_transform in enumerate(goal_transforms):
                x = goal_transform.p.x
                y = goal_transform.p.y
                theta = yaw_from_quat(goal_transform.r)
                rods_to_push = np.array([int(b) for b in f'{np.random.randint(1, self.num_rods + 2):02b}'])
                parameters = np.append(np.array([x, y, theta]), rods_to_push)
                yield parameters

    def _gen_random_parameters(self, env, state):
        curr_pos = np.array(get_pose_pillar_state(state, PushRodEnv.robot_name))
        curr_yaw = yaw_from_np_quat(curr_pos[3:])
        while True:
            random_dir = np.random.uniform(low=0, high=2 * np.pi)
            random_dist = np.random.uniform(low=0, high=self._max_goal_dist)
            pos = (curr_pos[0] + random_dist * np.cos(random_dir), curr_pos[1] + random_dist * np.sin(random_dir))

            # always push at least rod
            rods_to_push = np.array([int(b) for b in f'{np.random.randint(1, self.num_rods + 2):02b}'])
            parameters = np.hstack([pos, np.random.uniform(low=curr_yaw - self._max_goal_ang,
                                                           high=curr_yaw + self._max_goal_ang),
                                    rods_to_push])
            yield parameters

    def _gen_relation_centric_parameters(self, env, state):
        while True:
            curr_pos = np.array(get_pose_pillar_state(state, PushRodEnv.robot_name))
            goal_transforms = env.get_goal_transforms_around_objects(state, in_only=True, plot=False)
            for i, goal_transform in enumerate(goal_transforms):
                x = goal_transform.p.x
                y = goal_transform.p.y
                curr_dist = np.linalg.norm(curr_pos[:2] - np.array([x, y]))
                yaw = yaw_from_quat(goal_transform.r)
                if curr_dist > self._max_goal_dist:
                    random_dist = 0  # this parameter will fail the precondition check but that shouldn't be handled at this phase
                else:
                    random_dist = np.random.uniform(low=curr_dist, high=self._max_goal_dist - curr_dist)
                # go farther in that direction
                direction = np.arctan2((y - curr_pos[1]), (x - curr_pos[0]))
                pos = (random_dist * np.cos(direction), random_dist * np.sin(direction))

                # always push at least rod
                rods_to_push = np.array([int(b) for b in f'{np.random.randint(1, self.num_rods + 2):02b}'])
                parameters = np.hstack([pos, yaw, rods_to_push])
                yield parameters

    @staticmethod
    def parameters_to_relative_parameters(parameters, ref_pillar_state, anchor_obj_name):
        a_T_w = pillar_state_obj_to_transform(ref_pillar_state, anchor_obj_name).inverse()

        xy_yaw = np.array(parameters[:3])
        w_T_o = xy_yaw_to_transform(xy_yaw)
        a_T_o = a_T_w * w_T_o

        relative_parameters = np.hstack((transform_to_xy_yaw(a_T_o), parameters[3:]))
        return relative_parameters

    @staticmethod
    def relative_parameters_to_parameters(relative_parameters, ref_pillar_state, anchor_obj_name):
        w_T_a = pillar_state_obj_to_transform(ref_pillar_state, anchor_obj_name)

        xy_yaw = np.array(relative_parameters[:3])
        a_T_o = xy_yaw_to_transform(xy_yaw)
        w_T_o = w_T_a * a_T_o

        parameters = np.hstack((transform_to_xy_yaw(w_T_o), relative_parameters[3:]))
        return parameters

    def unpack_parameters(self, parameters):
        goal_pos = parameters[:2]
        goal_yaw = parameters[2]
        rods_to_push = np.where(parameters[3:] == 1)[0]
        return np.array(goal_pos), goal_yaw, rods_to_push

    def make_controllers(self, initial_states, parameters, T_plan_max, t, dt):
        info_plans = []
        controllers = []
        self._traj_timesteps = []
        for env_idx, initial_state in enumerate(initial_states):
            internal_state = self.pillar_state_to_internal_state(initial_state)
            pusher_pos = internal_state[:2]
            goal_pos, goal_yaw, rods_to_push = self.unpack_parameters(parameters[env_idx])

            goal_wp = np.r_[goal_pos, [goal_yaw], [0] * 3]
            waypoints_list = []
            for rod_idx in rods_to_push:
                rod_pos = initial_state.get_values_as_vec([f'frame:rod{rod_idx}:pose/position'])[:2]

                # only add waypoint if it is between pusher and goal
                v_rod_to_goal = goal_pos - rod_pos
                v_rod_to_pusher = pusher_pos - rod_pos
                if v_rod_to_goal @ v_rod_to_pusher < 0:
                    rod_yaw = yaw_from_np_quat(
                        initial_state.get_values_as_vec([f'frame:rod{rod_idx}:pose/quaternion']))
                    rod_wp = np.r_[rod_pos, [rod_yaw], [0] * 3]
                    waypoints_list.append(rod_wp)
            waypoints_list.append(goal_wp)
            waypoints = np.array(waypoints_list).T

            # sort waypoints
            dist_vec = np.linalg.norm(waypoints[:2, :-1] - internal_state[:2].reshape(2, 1), axis=0)
            indx = np.append(np.argsort(dist_vec), -1)
            waypoints = waypoints[:, indx]

            mass = initial_state.get_values_as_vec(['constants/pusher_mass'])[0]
            dims = initial_state.get_values_as_vec(['constants/pusher_dims'])
            dt = initial_state.get_values_as_vec(["constants/dt"])[0]

            controller = LQRWaypointControllerXYYaw()
            T = np.clip(4 - t * dt, 1, 4)
            info_plan = controller.plan(internal_state, waypoints, dt, dims, mass=mass, T=T, t_start=t)

            self._traj_timesteps.append(controller.horizon)
            controllers.append(controller)
            info_plans.append(info_plan)

        return controllers, info_plans


class LQRSweep2Objects(FreeSpaceLQRMove):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_rods = 2
        self.pusher_position_tol = 0.01  # Big because this skill isn't good right now
        self.rod_position_tol = 0.01
        self.yaw_diff_thresh = 0.05
        self._max_goal_dist = 0.3
        self._terminate_on_timeout = False

    def check_termination_condition(self, internal_state, parameters, t, controller=None, env_idx=None):
        pusher_goal, rod_poses = self.unpack_parameters(parameters)
        all_poses = [pusher_goal] + rod_poses
        poses_close = []
        for i, object_pose in enumerate(all_poses):
            start_idx = 2 * i
            desired_position = parameters[start_idx:start_idx + 2]
            error_pos = desired_position - internal_state[:2]
            position_tol = self.pusher_position_tol if i == 0 else self.rod_position_tol
            pos_close = np.isclose(np.linalg.norm(error_pos), 0, atol=position_tol)
            poses_close.append(pos_close)
            if not pos_close:
                break

        timed_out = False
        if self._terminate_on_timeout and env_idx is not None:
            timed_out = t >= self._traj_timesteps[env_idx]

        return np.all(poses_close) or timed_out

    def pillar_state_to_internal_state(self, state):
        pusher_pose = state.get_values_as_vec(['frame:pusher:pose/position'])[:2]
        pusher_vel = state.get_values_as_vec(['frame:pusher:pose/linear_velocity'])[:2]
        rod_states = []
        for i in range(self.num_rods):
            rod_pose = state.get_values_as_vec([f'frame:rod{i}:pose/position'])[:2]
            rod_vel = state.get_values_as_vec([f'frame:rod{i}:pose/linear_velocity'])[:2]
            rod_states = np.hstack([rod_states, rod_pose, rod_vel])
        return np.hstack([pusher_pose, pusher_vel, rod_states]).flatten()

    def precondition_satisfied(self, state, parameters, check_valid_goal_state=False):
        """

        Args:
            state:
            parameters:

        Returns: The evaluation of 3 conditions:
            - if the pusher and rods are aligned
            - if the goals are close enough to the current poses
            - if the rods are between the pusher and each goal, so it can perform the action with sweeping

        """
        curr_pos = np.array(state.get_values_as_vec(['frame:pusher:pose/position']))[:2]
        close_dist = np.linalg.norm(parameters[:2] - curr_pos.reshape(-1, 1)) < self._max_goal_dist
        body_names = ['pusher', 'rod0', 'rod1']
        yaws = []
        for body_name in body_names:
            yaws.append(yaw_from_np_quat(state.get_values_as_vec([f"frame:{body_name}:pose/quaternion"])))

        rod_shapes = pillar_state_to_shapes(state, ['rod0', 'rod1'], eps=(-0.002, -0.05))
        pusher_shape = pillar_state_to_shapes(state, ['pusher'])[0]
        # yaws can't be too different. assume 1 degree of symmetry
        # check if each rod is in a region between the pusher and the goal for all goals
        rod_intersects = []
        yaws_similar = []
        goal_points = []
        for i in range(self.num_rods):
            start_idx = 2 + 2 * i
            goal_points.append(Point(parameters[start_idx:start_idx + 2]))
        pusher_yaw = yaws[0]
        for i in range(self.num_rods):
            rod_shape = rod_shapes[i]
            rod_yaw = yaws[1 + i]
            yaw_diff = min(abs(rod_yaw - pusher_yaw), abs(rod_yaw - pusher_yaw) - np.pi)  # assume same symmetries
            yaws_similar.append(yaw_diff < self.yaw_diff_thresh)
            for goal in goal_points:
                push_region = unary_union([pusher_shape, goal]).convex_hull
                rod_intersects.append(rod_shape.intersects(push_region))

        rods_in_push_region = np.all(rod_intersects)
        return close_dist and yaws_similar and rods_in_push_region

    def apply_action(self, env, env_idx, action):
        env.apply_force_torque_to_pusher(np.hstack([action, [0, ]]), env_idx)

    @staticmethod
    def parameters_to_relative_parameters(parameters, ref_pillar_state, anchor_obj_name):
        a_T_w = pillar_state_obj_to_transform(ref_pillar_state, anchor_obj_name).inverse()

        xy_yaw = np.array([parameters[0], parameters[1], 0])
        w_T_o = xy_yaw_to_transform(xy_yaw)
        a_T_o = a_T_w * w_T_o

        relative_parameters = transform_to_xy_yaw(a_T_o)[:2]
        return relative_parameters

    @staticmethod
    def relative_parameters_to_parameters(relative_parameters, ref_pillar_state, anchor_obj_name):
        w_T_a = pillar_state_obj_to_transform(ref_pillar_state, anchor_obj_name)

        xy_yaw = np.array([relative_parameters[0], relative_parameters[1], 0])
        a_T_o = xy_yaw_to_transform(xy_yaw)
        w_T_o = w_T_a * a_T_o

        parameters = transform_to_xy_yaw(w_T_o)[:2]
        return parameters

    def unpack_parameters(self, parameters):
        # rod_poses = []
        # for i in range(self.num_rods):
        #     start_idx = 2 + 2 * i
        #     rod_poses.append(parameters[start_idx:start_idx + 2])
        return np.array(parameters)

    def make_controllers(self, initial_states, parameters, T_plan_max, t, dt):
        info_plans = []
        controllers = []
        self._traj_timesteps = [None] * len(initial_states)
        for env_idx, initial_state in enumerate(initial_states):
            controller = LQRModeSwitch2ObjectsController()
            initial_pusher_pose = initial_state.get_values_as_vec(['frame:pusher:pose/position'])[:2]
            initial_rod_poses = []
            for i in range(self.num_rods):
                rod_pose = initial_state.get_values_as_vec([f'frame:rod{i}:pose/position'])[:2]
                initial_rod_poses.append(rod_pose)
            pusher_mass = initial_state.get_values_as_vec(['constants/pusher_mass'])[0]
            rod_mass = initial_state.get_values_as_vec(['constants/rod_mass'])[0]
            pusher_dims = initial_state.get_values_as_vec(['constants/pusher_dims'])
            rod_dims = initial_state.get_values_as_vec(['constants/rod_dims'])

            pusher_goal = self.unpack_parameters(parameters[env_idx])
            dt = initial_state.get_values_as_vec(["constants/dt"])[0]
            info_plan = controller.plan(initial_pusher_pose, initial_rod_poses, pusher_goal, pusher_dims,
                                        rod_dims,
                                        pusher_mass,
                                        rod_mass, dt, T_plan_max)
            self._traj_timesteps[env_idx] = len(controller.tau.T)
            controllers.append(controller)
            info_plans.append(info_plan)
        return controllers, info_plans

def _set_end_states_given_mask(effects_all, effects, idx_mask):
    """
    Util function: modifier. Effects can be none of model has no effects (ex. model precond not satisfied)
    """
    effect_idx = 0
    for i, use_idx in enumerate(idx_mask):
        if use_idx:
            if effects is not None:
                effects_all["end_states"][i] = effects["end_states"][effect_idx]
            else:
                effects_all["end_states"][i] = None
            effect_idx += 1
