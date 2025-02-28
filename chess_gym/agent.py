from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class _Agent(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def act(self, observation: np.ndarray, info: Dict[str, Any]):
        # your policy goes here
        raise NotImplementedError
    

class RandomAgent(_Agent):
    def __init__(self):
        super().__init__()

    def act(self, observation, info):
        available_actions = info["available_actions"]
        action = available_actions[np.random.choice(len(available_actions))]
        return action