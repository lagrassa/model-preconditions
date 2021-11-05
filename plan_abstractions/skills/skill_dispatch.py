"""
Author: Jacky Liang
jackyliang@cmu.edu
"""
import numpy as np
from .utils import merge_exec_data


class SkillDispatch:

    def __init__(self, env, T_exec_max=1000):
        self._env = env
        self._T_exec_max = T_exec_max
        
        self._skill_exec_cbs = [None] * env.n_envs
        self._exec_datas = [None] * env.n_envs
        self._skill_done = [False] * env.n_envs
        self._skill_ts = np.zeros(env.n_envs, dtype=np.int32)

        self._n_steps_to_settle = 10
        self._skill_ts_for_settle = np.zeros(env.n_envs)
        self._skill_has_called_set_state = [False] * env.n_envs

    def set_skill_exec_cb(self, skill_exec_cb, env_idx):
        if not self.is_env_available(env_idx):
            raise ValueError(f'Skill in env {env_idx} has not finished yet!')
        
        self._skill_exec_cbs[env_idx] = skill_exec_cb
        self._exec_datas[env_idx] = None
        self._skill_done[env_idx] = False
        self._skill_ts[env_idx] = 0
        self._skill_ts_for_settle[env_idx] = 0
        self._skill_has_called_set_state[env_idx] = False

    def set_all_skill_exec_cbs(self, skill_exec_cbs):
        assert len(skill_exec_cbs) == self._env._n_envs
        for skill_exec_cb in skill_exec_cbs:
            self.set_skill_exec_cb(skill_exec_cb)

    def _has_skill_env_settled(self, env_idx):
        skill_exec_cb = self._skill_exec_cbs[env_idx]
        if not skill_exec_cb.do_set_state:
            return True
        return self._skill_ts_for_settle[env_idx] >= skill_exec_cb.n_steps_for_set_state

    def has_skill(self, env_idx):
        return self._skill_exec_cbs[env_idx] is not None

    def remove_skill(self, env_idx):
        if not self.has_skill(env_idx):
            raise ValueError(f'Env {env_idx} does not have a skill!')
        if not self.is_skill_done(env_idx):
            raise ValueError(f'Skill in env {env_idx} has not finished yet!')

        self._skill_exec_cbs[env_idx] = None
        self._exec_datas[env_idx] = None
        self._skill_done[env_idx] = False
        self._skill_ts[env_idx] = 0
        self._skill_ts_for_settle[env_idx] = 0
        self._skill_has_called_set_state[env_idx] = False

    def step(self, real_robot=False):
        for env_idx in range(self._env.n_envs):
            if not self.has_skill(env_idx):
                continue
            skill_exec_cb = self._skill_exec_cbs[env_idx]

            # logic for skills that need to set their init state
            if not self._skill_has_called_set_state[env_idx]:
                if skill_exec_cb.do_set_state:
                    self._env.set_state(skill_exec_cb.initial_state, env_idx, n_steps=0)
                else:
                    assert skill_exec_cb.initial_state is None, "Cannot have an initial state"
                    initial_state = self._env.get_state(env_idx)
                    # TODO Maybe we should verify that all object velocities are 0.
                    skill_exec_cb.set_initial_state_from_env(initial_state)
                self._skill_has_called_set_state[env_idx] = True
                
            # skip skills that haven't had their init state settled
            if not self._has_skill_env_settled(env_idx):
                continue

            # only apply skills that are not done
            if not self.is_skill_done(env_idx):
                skill_exec_cb.pre_env_step(self._env, env_idx, self._skill_ts[env_idx])

                # A skill is done if it terminated OR it timed out to T_exec_max
                self._skill_done[env_idx] = skill_exec_cb.terminated or self._skill_ts[env_idx] >= self._T_exec_max - 1
                if real_robot:
                    if self._skill_ts[env_idx] >= self._T_exec_max - 1 and not skill_exec_cb.terminated :
                        print("Real robot skill did not terminate")
                        skill_exec_cb.exec_data["end_states"] = self._env.get_end_state(should_reset_to_viewable = skill_exec_cb._skill._should_reset_to_viewable).get_serialized_string()
                        
                self._exec_datas[env_idx] = skill_exec_cb.exec_data

                if not self.is_skill_done(env_idx):
                    self._skill_ts[env_idx] += 1
        
        step_costs = self._env.step()
        self._skill_ts_for_settle[:] += 1

        for env_idx in range(self._env.n_envs):
            if self.has_skill(env_idx) and \
                not self.is_skill_done(env_idx) and \
                    self._has_skill_env_settled(env_idx):
                self._skill_exec_cbs[env_idx].post_env_step(step_costs[env_idx])

    def is_env_available(self, env_idx):
        return not self.has_skill(env_idx) or self.is_skill_done(env_idx)

    @property
    def available_envs_idxs(self):
        env_idxs = []
        for env_idx in range(self._env.n_envs):
            if self.is_env_available(env_idx):
                env_idxs.append(env_idx)
        return env_idxs

    @property
    def has_skill_env_idxs(self):
        env_idxs = []
        for env_idx in range(self._env.n_envs):
            if self.has_skill(env_idx):
                env_idxs.append(env_idx)
        return env_idxs

    def is_skill_done(self, env_idx):
        if not self.has_skill(env_idx):
            raise ValueError(f'Env {env_idx} does not have a skill!')
        return self._skill_done[env_idx]

    @property
    def all_skills_done(self):
        return np.all(self._skill_done)

    def get_exec_data(self, env_idx):
        if self._skill_exec_cbs[env_idx] is None:
            raise ValueError(f'Env {env_idx} does not have a skill!')
        
        if not self.is_skill_done(env_idx):
            raise ValueError(f'Skill in env {env_idx} has not finished yet!')
        
        return self._exec_datas[env_idx]
    
    def get_all_skill_data(self, env_idx):
        skill_exec_cb = self._skill_exec_cbs[env_idx]
        if skill_exec_cb is None:
            raise ValueError(f'Env {env_idx} does not have a skill!')
        
        if not self.is_skill_done(env_idx):
            raise ValueError(f'Skill in env {env_idx} has not finished yet!')

        return skill_exec_cb.get_all_data_to_save(self._env, env_idx)

    def get_combined_exec_data(self):
        return merge_exec_data(self._exec_datas)
