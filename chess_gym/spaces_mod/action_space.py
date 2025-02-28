from typing import Literal
from gymnasium.spaces import Space, Discrete
import numpy as np

from chess_gym.spaces_mod.utils import get_numeric_action, get_possible_actions
from enum import Enum


class ACTION_ENCODING(Enum):
    """
    Enumeration of encoding strategies for representing chess actions in the action space.
    
    Attributes:
        action_wise (int): Encodes actions as one-hot vectors where each element corresponds 
                           to a unique action (e.g., moving a piece from one square to another).
        symbol_wise (int): Encodes actions based on individual symbols (e.g., origin and destination
                           squares and optional promotion) in a more granular, symbol-based format.
    """
    one_hot = 0
    multi_hot = 1


    class ChessAction(Discrete):
        def __init__(self, seed = None):
            possible_uci_actions, _ = get_possible_actions(
                return_numeric=False,
            )
            self._uci_action_map = np.array(possible_uci_actions)
            n = len(possible_uci_actions)
            super().__init__(n, seed, 0)

        def from_uci(self, uci_action: str | np.ndarray):
            try:
                if isinstance(uci_action, np.ndarray):
                    mask = self._uci_action_map[None].repeat(len(uci_action), axis=0) == uci_action[:, None]
                    indices = np.argwhere(mask)[:, 1]
                    return indices
                else:
                    return np.argwhere(self._uci_action_map == uci_action)
            except KeyError:
                raise ValueError(f"Given uci action: {uci_action} is not part of action space")
        
        def to_uci(self, action: int | np.ndarray):
            if isinstance(action, np.ndarray): 
                return self._uci_action_map[action]
            else:
                return str(self._uci_action_map[action])
            
        def __repr__(self) -> str:
            """Gives a string representation of this space."""
            if self.start != 0:
                return f"ChessAction({self.n}, start={self.start})"
            return f"ChessAction({self.n})"
    

class ChessAction(Space):
    """
    Custom action space for chess moves, with support for both action-wise and symbol-wise 
    encoding. This class is built to handle a finite set of valid chess actions and can 
    represent actions in a variety of formats, including strings, indices, and one-hot encoded 
    vectors.

    Args:
        action_encoding (ACTION_ENCODING): Specifies the encoding strategy, either `action_wise`
                                           or `symbol_wise`, to represent actions.
        shape (tuple, optional): Shape of the space. Defaults to None.
        dtype (data-type, optional): Desired data-type of the space. Defaults to None.
        seed (int, optional): Random seed for sampling actions.

    Attributes:
        _action_encoding (ACTION_ENCODING): The selected encoding strategy.
        _possible_actions_strings (list): List of UCI strings for possible actions in action-wise encoding.
        symbol_space_string (str): String representation of all chess symbols in action-wise encoding.
        _possible_actions_numeric (np.ndarray): Array of numeric representations of actions for symbol-wise encoding.
        symbol_space_numeric (tuple): Tuple of arrays, each representing a numeric space for different 
                                      components of symbol-wise actions.
    """
    def __init__(
        self,
        action_encoding: ACTION_ENCODING = ACTION_ENCODING.one_hot,
        shape=None,
        dtype=None,
        seed=None,
    ):
        super().__init__(shape, dtype, seed)
        self._action_encoding = action_encoding
        self._possible_actions_strings, self.symbol_space_string = get_possible_actions(
            return_numeric=False,
        )
        self._possible_actions_numeric, self.symbol_space_numeric = (
            get_possible_actions(
                return_numeric=True,
            )
        )

        if self._action_encoding == ACTION_ENCODING.one_hot:
            self._shape = (len(self._possible_actions_strings), )
        elif self._action_encoding == ACTION_ENCODING.multi_hot:
            self._shape = np.concat(self.symbol_space_numeric).shape

    def sample(self, mask=None):
        """
        Samples a valid chess action from the space, optionally using a probability mask.

        Args:
            mask (np.ndarray, optional): A probability mask over actions, where each entry is the 
                                         likelihood of sampling that action.

        Returns:
            np.ndarray: A sampled action from the action space, encoded according to `_action_encoding`.
        """
        # sample action
        if mask is None:
            mask = np.ones(len(self))
            mask /= len(self)
        index = np.random.choice(
            np.arange(len(self)),
            size=(1,),
            p=mask,
        ).item()
        return self[index]

    def contains(self, x: str | int | np.ndarray) -> bool:
        """
        Checks whether a given action is part of the defined action space.

        Args:
            x (str | int | np.ndarray): The action to check, which can be a UCI string, an index, 
                                        or a one-hot encoded vector.

        Returns:
            bool: True if the action exists within the space, otherwise False.
        """
        if isinstance(x, str):
            return x in self._possible_actions_strings
        elif isinstance(x, int):
            return x >= 0 and x < len(self)
        elif isinstance(x, np.ndarray):
            if self._action_encoding == ACTION_ENCODING.one_hot:
                assert x.sum() == 1 and x.max() == 1, "Assert one hot encoded vector"
                index = np.argmax(x)
                return index >= 0 and index < len(self)
            elif self._action_encoding == ACTION_ENCODING.multi_hot:
                action = get_numeric_action(x, self.symbol_space_numeric)
                return action in self._possible_actions_numeric
            else:
                raise NotImplementedError(
                    "Unknown action encoding method: " + self._action_encoding.name
                )

    def uci_to_action(self, uci_action: str) -> np.ndarray:
        """
        Converts a UCI-encoded action string to the corresponding action representation.

        Args:
            uci_action (str): The UCI action string (e.g., 'e2e4') to convert.

        Returns:
            np.ndarray: The one-hot encoded vector representing the action in the action space.
        """
        mask = np.equal(self._possible_actions_strings, uci_action).astype(int)
        assert mask.sum() == 1, "Given uci action is not in the set of all possible actions"
        index = np.argmax(mask)
        return self[index]
    
    def action_to_uci(self, action: np.ndarray) -> str:
        assert self.contains(action)
        if self._action_encoding == ACTION_ENCODING.one_hot:
            return self._possible_actions_strings[np.argmax(action)]
        elif self._action_encoding == ACTION_ENCODING.multi_hot:
            action = get_numeric_action(action, self.symbol_space_numeric)
            index = np.argmax((action == self._possible_actions_numeric).sum(-1))
            return self._possible_actions_strings[index]
        else: 
            raise NotImplementedError(f"No method for {self._action_encoding=}")


    def __getitem__(self, index: int) -> np.ndarray:
        """
        Retrieves the one-hot encoded vector for a specific action by index.

        Args:
            index (int): Index of the desired action.

        Returns:
            np.ndarray: One-hot encoded vector representing the action.
        """
        action = np.zeros(self._shape, dtype=self.dtype)

        if self._action_encoding == ACTION_ENCODING.one_hot:
            # one hot encoding of action
            action[index] = 1
        elif self._action_encoding == ACTION_ENCODING.multi_hot:
            action_repr = self._possible_actions_numeric[index]
            acc = 0
            for action_symbol, symbol_space in zip(
                action_repr, self.symbol_space_numeric
            ):
                action[np.argmax(action_symbol == symbol_space) + acc] = 1
                acc += len(symbol_space)
        else:
            raise NotImplementedError(
                "Unknown action encoding method: " + self._action_encoding.name
            )
        return action
    
    def __len__(self) -> int:
        """
        Returns the number of possible actions in the action space.

        Returns:
            int: Total number of actions.
        """
        return len(self._possible_actions_strings)
