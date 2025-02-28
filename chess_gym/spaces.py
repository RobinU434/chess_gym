import numpy as np
from chess_gym.spaces_mod.utils import get_possible_actions
from gymnasium.spaces import Discrete


class ChessAction(Discrete):
    def __init__(self, seed=None):
        possible_uci_actions, _ = get_possible_actions(
            return_numeric=False,
        )
        self._uci_action_map = np.array(possible_uci_actions)
        n = len(possible_uci_actions)
        super().__init__(n, seed, 0)

    def from_uci(self, uci_action: str | np.ndarray):
        try:
            if isinstance(uci_action, np.ndarray):
                mask = (
                    self._uci_action_map[None].repeat(len(uci_action), axis=0)
                    == uci_action[:, None]
                )
                indices = np.argwhere(mask)[:, 1]
                return indices
            else:
                return np.argwhere(self._uci_action_map == uci_action)
        except KeyError:
            raise ValueError(
                f"Given uci action: {uci_action} is not part of action space"
            )

    def to_uci(self, action: int | np.ndarray):
        if isinstance(action, np.ndarray):
            return self._uci_action_map[action]
        else:
            return str(self._uci_action_map[action])

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        return f"{type(self).__name__}({self.n})"


class UCIChessAction(ChessAction):
    def __init__(self, seed=None):
        super().__init__(seed)

    def sample(self, mask=None):
        sample = super().sample(mask)
        return self.to_uci(sample)

    def contains(self, x):
        action = self.from_uci(x)
        return self.contains(action)
