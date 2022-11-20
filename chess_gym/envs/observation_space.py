import random
from typing import List, Optional, Tuple
from gym import spaces

import chess
import chess.svg

import numpy as np
import torch 


class ChessSpace(spaces.Space):
    def __init__(self, board: chess.Board, observation_mode, render_size: int = 512, dtype: Optional[np.float64] = np.float64) -> None:

        spaces.Dict()
        super().__init__(None, dtype)
        self.board = board
        self.observation_mode = observation_mode

        self.board_space = self._get_board_space(self.observation_mode, render_size)
        # I will pass the possible actions into the environment.
        # The possible actions are state dependent 
        #   -> different amount of actions in each state 
        #   -> No constant value possible / eventually is the encoding dimension possible
        self._shape = self.board_space.shape, None  
        self.possible_actions, self.action_symbols = self._get_possible_actions()

        self.action_encodings = self.encode_actions(self.possible_actions, method="action_wise")
    @property
    def n_action_symbols(self):
        return len(self.action_symbols)

    @property
    def n_actions(self):
        """len of all possible actions"""
        return len(self.possible_actions)
    
    def _get_board_space(self, observation_mode, render_size: int = 512) -> spaces.Space:

        observation_spaces = {  'rgb_array': spaces.Box(low = 0, high = 255,
                                                shape = (render_size, render_size, 3),
                                                dtype = np.uint8),
                                'piece_map': spaces.Box(low = 0, high = 1,
                                                shape = (8, 8, 13),
                                                dtype = np.uint)}

        return observation_spaces[observation_mode]
    
    @staticmethod
    def _get_possible_actions()-> Tuple[List, List]:
        """returns all permuations of chess board squares and all used symbols

        Returns:
            Tuple[List, List]: _description_
        """
        nums = np.linspace(1, 8, 8)
        letters = "abcdefgh"

        all_symbols = str(12345678)+ letters
        
        actions = []
        for pick_letter in letters:
            for pick_num in range(1, 9):
                for place_letter in letters:
                    for place_num in range(1, 9):
                        if pick_letter == place_letter and pick_num == place_num:
                            continue
                        actions.append(pick_letter + str(pick_num) + place_letter + str(place_num))
        
        return actions, all_symbols

    def encode_actions(self, actions, encoding: str="one_hot", method: str = "action_wise"):
        if encoding == "one_hot":
            return self.one_hot_encoding(actions, method)

    def one_hot_encoding(self, actions, method: str = "action_wise"):
        if method == "action_wise":
            return self.action_wise_encoding(actions)
        elif method == "symbol_wise":
            return self.symbol_wise_encodings(actions)
        else:
            raise NotImplemented(f"only 'action_wise' and symbol_wise' are supported methods. You chose {method}")
    
    def actionToIndex(self, action):
        # Find symbol index from all_letters, e.g. "a1a2" = 0
        return self.possible_actions.index(action)

    def actionToTensor(self, action):
        # Just for demonstration, turn an action into a <1 x n_actions> Tensor
        tensor = torch.zeros(1, self.n_actions)
        tensor[0, self.actionToIndex(action)] = 1
        return tensor

    def action_wise_encoding(self, actions: List[str]):
        # Turn a line into a <len_action_sequence x 1 x n_actions>,
        encodings = torch.zeros(len(actions), 1, self.n_actions)
        for idx, action in enumerate(actions):
            encodings[idx, 0, self.actionToIndex(action)] = 1

        return encodings

    def symbolToIndex(self, letter):
        # Find symbol index from all_letters, e.g. "1" = 0
        return self.action_symbols.index(letter)

    def symbol_wise_encoding(self, action):
        # Turn a line into a <action_lenght x n_letters>,
        # action length is in uci format per default = 4
        # or an array of one-hot letter vectors
        tensor = torch.zeros(len(action), self.n_action_symbols)
        for idx, symbol in enumerate(action):
            tensor[idx, self.symbolToIndex(symbol)] = 1
        return tensor
    
    def symbol_wise_encodings(self, actions):
        # Turn a line into a <len_action_sequence x 4 x n_action_symbols>,
        encodings = []
        for action in actions:
            encoded = self.symbol_wise_encoding(action)
            encodings.append(self.symbol_wise_encoding(action))

        return torch.stack(encodings)
        return torch.cat(encodings, out=out_format)

    def sample(self):
        """sample from board space and from possible actions 

        Returns:
            _type_: _description_
        """
        # sample form board space
        board_sample = self.board_space.sample()

        # sample from action space
        action_sample = random.sample(self.possible_actions, random.randint(1, 24))
        return board_sample, action_sample