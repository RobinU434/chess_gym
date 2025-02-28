import logging

import numpy as np
from gymnasium import ActionWrapper, ObservationWrapper
from gymnasium.spaces import Box, Discrete, MultiDiscrete

from chess_gym.chess_env import ChessEnv
from chess_gym.spaces import UCIChessAction
from chess_gym.utils import _build_row, one_hot2piece_map


class PieceMapWrapper(ObservationWrapper):
    """
    Observation wrapper that converts one-hot encoded board representation to a piece map.
    
    Methods:
        __init__: Initializes the wrapper with the environment.
        observation: Converts the one-hot encoded board to a piece map.
    """
    def __init__(self, env: ChessEnv):
        super().__init__(env)
        self.observation_space = MultiDiscrete(np.ones((8, 8)) * 13)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return one_hot2piece_map(observation)


class FenObsWrapper(ObservationWrapper):
    """
    Observation wrapper that represents the board state using FEN notation.
    
    Methods:
        __init__: Initializes the wrapper and sets the observation space.
        observation: Converts the board state to a FEN string representation.
    """
    def __init__(self, env: ChessEnv):
        super().__init__(env)
        n = int(7.7e45)
        self.observation_space = Discrete(n)
        logging.warning("sampling from this action space might not work")

    def observation(self, observation):
        fen = self.env.board.fen().split(" ")[0]
        fen = "/".join(map(_build_row, fen.split("/")))
        return fen


class RBGObsWrapper(ObservationWrapper):
    """
    Observation wrapper that provides an RGB image representation of the board.
    
    Methods:
        __init__: Initializes the wrapper and sets the observation space.
        observation: Converts the board state to an RGB image array.
    """
    def __init__(self, env: ChessEnv):
        super().__init__(env)
        self.observation_space = Box(0, 255, (env._render_size, env._render_size, 3))

    def observation(self, observation):
        rgb_array = np.asarray(self.env._get_image())[..., :-1]
        return np.permute_dims(rgb_array, (2, 1, 0))


class UCIActionWrapper(ActionWrapper):
    """
    Action wrapper that allows actions to be specified using UCI notation.
    
    Methods:
        __init__: Initializes the wrapper and sets the action space.
        action: Converts a UCI action to the appropriate format for the environment.
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = UCIChessAction()

    def action(self, action):
        action = self.action_space.from_uci(action).item()
        return action
