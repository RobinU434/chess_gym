import random
from typing import List, Optional
from gym import spaces

import chess
import chess.svg

import numpy as np


class ChessSpace(spaces.Space):
    def __init__(self, dtype: Optional[np.float64], board: chess.Board, observation_mode, render_size: int = 512) -> None:
        super().__init__(None, dtype)
        self.board = board
        self.observation_mode = observation_mode

        self.board_space = self._get_board_space(self.observation_mode, render_size)
        self._shape = self.board_space.shape
        self.possible_actions = self._get_possible_actions()

    def _get_board_space(self, observation_mode, render_size: int = 512) -> spaces.Space:

        observation_spaces = {  'rgb_array': spaces.Box(low = 0, high = 255,
                                                shape = (render_size, render_size, 3),
                                                dtype = np.uint8),
                                'piece_map': spaces.Box(low = 0, high = 1,
                                                shape = (8, 8, 13),
                                                dtype = np.uint)}

        return observation_spaces[observation_mode]
    
    @staticmethod
    def _get_possible_actions()-> List:
        """returns all permuations of chess board squares

        Returns:
            _type_: _description_
        """
        nums = np.linspace(1, 8, 8)
        letters = "abcdefgh"
        
        actions = []
        for pick_letter in letters:
            for pick_num in range(1, 9):
                for place_letter in letters:
                    for place_num in range(1, 9):
                        actions.append(pick_letter+str(pick_num)+place_letter+str(place_num))
        
        return actions

    def sample(self):
        """sample from board space and from possible actions 

        Returns:
            _type_: _description_
        """
        # sample form board space
        board_sample = self.board_space.sample()

        # sample from action space
        action_sample = random.sample(self.possible_actions, random.randint(1, 16))
        return board_sample, action_sample