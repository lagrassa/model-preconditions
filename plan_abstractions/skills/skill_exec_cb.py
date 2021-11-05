"""
Author: Jacky Liang
jackyliang@cmu.edu
"""

class SkillExecCb:

    def __init__(self, skill, initial_state, parameter, T_plan_max, set_state=True):
        self._skill = skill
        self._initial_state = initial_state
        # If we do not pass initial state we should set initial state from env. 
        # Separating these vars for now.
        self._initial_state_from_env = None
        self._parameter = parameter
        self._T_plan_max = T_plan_max
        self._set_state = set_state

        self._had_first_pre_env_step = False

        self._controller = None

        self._exec_data = {
            'end_states': None,
            'T_exec': 0,
            'terminated': False,
            'costs': 0,
            'info_plan': None
        }
            
    @property
    def terminated(self):
        return self._exec_data['terminated']

    @property
    def exec_data(self):
        return self._exec_data

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def do_set_state(self):
        return self._set_state

    @property
    def n_steps_for_set_state(self):
        return 10
    
    def set_initial_state_from_env(self, initial_state):
        self._initial_state_from_env = initial_state
    
    def get_all_data_to_save(self, env, env_idx):
        assert self._initial_state is None or self._initial_state_from_env is None
        assert self._initial_state is not None or self._initial_state_from_env is not None

        initial_state = self._initial_state if self._initial_state is not None else self._initial_state_from_env
        dict_to_save = {
            'initial_states': initial_state.get_serialized_string(),
            'parameters': self._parameter
,
            # TODO: Should save parameter_types
            'exec_data': self._exec_data,
            #'current_shape_props': env.get_shape_props_for_objects([env_idx]),
            #'current_rb_props': env.get_rigid_body_props_for_objects([env_idx]),
        }

        return dict_to_save

    def pre_env_step(self, env, env_idx, t):
        real_robot = True
        if not self._had_first_pre_env_step:
            self._had_first_pre_env_step = True

            initial_settled_state = env.get_state(env_idx)
            controllers, info_plans = self._skill.make_controllers([initial_settled_state], [self._parameter], self._T_plan_max, 0, env.dt, env._real_robot)

            self._controller = controllers[0]
            self._exec_data['info_plan'] = info_plans[0]

        if not self.terminated:
            if t == 0 and hasattr(env, 'set_skill_params'):
                env.set_skill_params(env_idx, self._parameter)

            pillar_state = env.get_state(env_idx)
            internal_state = self._skill.pillar_state_to_internal_state(pillar_state)
            self._exec_data['end_states'] = pillar_state.get_serialized_string()
            self._exec_data['T_exec'] = t

            if self._skill.check_termination_condition(internal_state, self._parameter, t, self._controller, env_idx):
                if real_robot:
                    self._exec_data['end_states'] = env.get_end_state(should_reset_to_viewable = self._skill._should_reset_to_viewable).get_serialized_string()

                self._exec_data['terminated'] = True
                return

            action = self._controller(internal_state, t)
            self._skill.apply_action(env, env_idx, action)

            if self._skill.do_replan and t % self._skill.replan_interval == 0:
                current_pillar_state = env.get_state(env_idx)
                # TODO(jacky): do something about the unused info_plans
                controllers, _ = self._skill.make_controllers([current_pillar_state], [self._parameter], self._T_plan_max, t, env.dt, env._real_robot)
                self._controller = controllers[0]

    def post_env_step(self, step_cost):
        if self._exec_data['terminated']:
            return True, self._exec_data

        self._exec_data['costs'] += step_cost

        return False, self._exec_data
        
