import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseTask(ABC):
    """
    Implements task specific methods to be used in task planning.
    """

    def __init__(self, cfg):
        """
        Skill specific parameter generator functions: a mapping from Skill name to generator.
        Used by the generic task generator.
        """
        self._cfg = cfg
        self._skill_specific_param_generators = {}
        self._setup_callbacks = []

    @property
    def setup_callbacks(self):
        return self._setup_callbacks

    def pillar_state_to_internal_state(self, pillar_state):
        """Converts to an environment/task specific representation of the state."""
        pass

    @abstractmethod
    def is_goal_state(self, pillar_state):
        pass

    def is_valid_state(self, pillar_state, skills):
        """Checks the validity of the current state. Pruning away states far
        away from the goal will help search."""
        pass

    def is_terminal_state(self, pillar_state, skills):
        return self.is_goal_state(pillar_state) \
               or not self.is_valid_state(pillar_state, skills)

    @abstractmethod
    def evaluate(self, pillar_state, q_id=0):
        """
        Returns: environment specific cost-to-go to for a state.
        """
        pass

    def evaluate_admissible(self, pillar_state):
        """
        Returns: environment specific admissible cost-to-go to for a state.
        """
        raise NotImplementedError

    def evaluate_inadmissible(self, pillar_state):
        """
        Returns: environment specific cost-to-go to for a state.
        """
        raise NotImplementedError

    @abstractmethod
    def states_similar(self, pillar_state_1, pillar_state_2):
        """
        Whether these 2 states should be considered the same for planning
        Args:
            pillar_state_1:
            pillar_state_2:

        Returns:

        """
        pass

    def pretty_print_goal_params(self):
        pass

    def set_detector(self):
        pass
    
    def pretty_print_with_reference_to_pillar_state(self, pillar_state):
        return ""

    def generate_parameters(self, skill, env, state, num_parameters=1, return_param_types=False, debug=False, 
                            valid_success_per_param_type=None, check_valid_goal_state=True):
        skill_class_name = skill.__class__.__name__
        task_oriented_sampler_gen = None
        if skill_class_name in self._skill_specific_param_generators.keys():
            task_oriented_sampler_gen = self._skill_specific_param_generators[skill_class_name](env, state)
        return skill.generate_parameters(env, state,
                                         task_oriented_sampler_gen=task_oriented_sampler_gen,
                                         num_parameters=num_parameters,
                                         return_param_types=return_param_types,
                                         debug=debug,
                                        # generic one used for data collection
                                         valid_success_per_param_type=valid_success_per_param_type,
                                         check_valid_goal_state=check_valid_goal_state)
