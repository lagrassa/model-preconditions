from abc import ABC, abstractmethod

import quaternion
from pillar_state import State
from plan_abstractions.envs.utils import get_joint_position_pillar_state, get_pose_pillar_state
from plan_abstractions.models import create_deviation_wrapper_from_cfg, create_sem_wrapper_from_cfg
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
from plan_abstractions.utils import angle_axis_between_quats


class TransitionModel(ABC):
    def __init__(self, model_cfg=None):
        self._model_cfg = model_cfg
        deviation_cfg = model_cfg.get("deviation_cfg", None)
        self._deviation_cfg = deviation_cfg
        self._use_deviation_model = deviation_cfg is not None and deviation_cfg.get("use_deviation_model", False)
        if self._use_deviation_model:
            try:
                self._deviation_wrapper = create_deviation_wrapper_from_cfg(deviation_cfg)
            except RuntimeError:
                logging.error("loading error")
            self._acceptable_deviation = deviation_cfg["acceptable_deviation"]

    @abstractmethod
    def apply(self, states, params, env, T_plan_max, T_exec_max, skill, pb_env):
        pass

    def model_precondition_satisfied(self, state, parameters):
        if not self._use_deviation_model:
            return True
        if isinstance(state, np.ndarray):
            predicted_deviation = self._deviation_wrapper.predict_from_np(state, parameters)
        else:
            #assume is pillar_state
            predicted_deviation = self._deviation_wrapper.predict_from_pillar_state(state, parameters)
        res = predicted_deviation < self._acceptable_deviation
        debug_name = self._model_cfg["debug_name"]
        if not res:
            logging.debug(f"Precondition violated for {debug_name} because predicted deviation was {predicted_deviation} for parameters {parameters.round(2)}")
            print(f"Precondition violated for {debug_name} because predicted deviation was {predicted_deviation} for parameters {parameters.round(2)} and state {state.round(2)}")
        else:
            logging.debug(f"Precondition satisfied for {debug_name} because predicted deviation is {predicted_deviation} for parameters {parameters.round(2)} ")
            print(f"Precondition satisfied for {debug_name} because predicted deviation is {predicted_deviation} for parameters {parameters.round(2)} and state {state[:2]} ")
        return res


class SEMModel(TransitionModel):
    def __init__(self, model_cfg, cache_dir="/tmp"):
        # create deviation model here
        super().__init__(model_cfg)
        sem_cfg = model_cfg["sem_cfg"]
        if sem_cfg.get("pillar_state_convert", True):
            sem_state_obj_names = sem_cfg["sem_state_obj_names"]
        else:
            sem_state_obj_names = None
        self._sem_wrapper = create_sem_wrapper_from_cfg(sem_cfg, sem_state_obj_names=sem_state_obj_names, cache_dir=cache_dir)

    def apply(self, states, params, env, T_plan_max, T_exec_max, skill, pb_env):
        effects_list = [
            self._sem_wrapper(state, parameters)
            for state, parameters in zip(states, params)
        ]
        if isinstance(effects_list[0], np.ndarray):
            effects = {}
            effects_list = np.vstack(effects_list)
            effects["end_states"] = effects_list
            effects["costs"] = np.ones((len(effects_list),))

        else: #uses old effects specification
            effects = {k: [] for k in effects_list[0]}
            for i in range(len(effects_list)):
                for k in effects:
                    effects[k].append(effects_list[i][k])

            for i, param in enumerate(params):
                effects["end_states"][i] = effects["end_states"][i][0]
                effects["costs"][i] = effects["costs"][i][0]
                effects["T_exec"][i] = effects["T_exec"][i][0]
                effects["info_plan"][i]["T_plan"] = effects["info_plan"][i]["T_plan"][0]
        return effects


class SimModel(TransitionModel):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)

    def apply(self, states, params, env, T_plan_max, T_exec_max, skill, pb_env):
        robot_name = "franka"
        for state in states:
            state_copy = State.create_from_serialized_string(state.get_serialized_string())
            fk_result = pb_env.forward_kinematics(get_joint_position_pillar_state(state_copy, "franka"))
            ee_pose = get_pose_pillar_state(state, "franka:ee")
            close_pos = np.allclose(fk_result[:3], ee_pose[:3], atol=0.008)
            quat_dist = np.linalg.norm(angle_axis_between_quats(quaternion.from_float_array(ee_pose[3:]),
                                                                quaternion.from_float_array(fk_result[3:])))
            reject_collisions = False
            if not close_pos or quat_dist > 0.03:  # These numbers dont matter much, if theyre off they will be *very* off.
                # Recompute IK and adjust values
                rod0_pose = get_pose_pillar_state(state_copy, "rod0")
                rod1_pose = get_pose_pillar_state(state_copy, "rod1")
                ik_result, collision = pb_env.inverse_kinematics(ee_pose, rod0_pose, rod1_pose)
                if reject_collisions and collision:
                    logging.debug("Invalid state encountered")
                    return -1
                ik_ee_pose = pb_env.forward_kinematics(ik_result)
                ik_quat_dist = np.linalg.norm(angle_axis_between_quats(quaternion.from_float_array(ee_pose[3:]),
                                                                       quaternion.from_float_array(ik_ee_pose[3:])))
                assert (np.linalg.norm(ik_ee_pose[:3] - np.array(ee_pose[:3]) < 0.005))
                assert (ik_quat_dist < 0.04)
                state.update_property(f"frame:{robot_name}:joint/position", ik_result)
            else:
                continue
        logging.info(f"About to execute :{skill}")
        return skill.execute(env, states, params, T_plan_max, T_exec_max, plot=False)
