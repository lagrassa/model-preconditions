from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np



class BaseController(ABC):

    def __init__(self):
        self._has_planned = False

    @property
    def has_planned(self):
        return self._has_planned

    def plan(self, *args, **kwargs):
        if self.has_planned:
            raise ValueError('Controller has already been planned!')

        infos = self._plan(*args, **kwargs)
        self._has_planned = True
        return infos

    def __call__(self, internal_state, t, delta=False):
        if not self._has_planned:
            raise ValueError('Need to call plan first!')
        return self._call(internal_state, t, delta=delta)

    @abstractmethod
    def _plan(self):
        """
        Need to overwrite this method to implement planning.

        Returns:
            Info dict w/ T_plan and other planner-specific info
        """
        pass

    @abstractmethod
    def _call(self, internal_state, t):
        """
        returns action taken from a internal_state at skill time t
        """
        pass


