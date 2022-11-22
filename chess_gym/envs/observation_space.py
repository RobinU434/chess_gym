import gym 
import random
from typing import List, Optional, Tuple
from gym import spaces
from progress.bar import Bar

import chess
import chess.svg

import numpy as np
import torch 


class ChessSpace(spaces.Space):
    def __init__(   self,
                    env: gym.Env, 
                    observation_mode,
                    render_size: int = 512, 
                    dtype: Optional[np.float64] = np.float64, 
                    action_encoding: str="action_wise") -> None:

        spaces.Dict()
        super().__init__(None, dtype)
        self.board = env.board
        self.observation_mode = observation_mode
        self.action_encoding_method = action_encoding
     
        self.board_space = self._get_board_space(self.observation_mode, render_size)
        self.set_shape()
        self.possible_actions, self.action_symbols = self._get_possible_actions()

        self.action_encodings = self.one_hot_encoding(self.possible_actions)
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
                                                dtype = np.float64),
                                'piece_map': spaces.Box(low = 0, high = 1,
                                                shape = (8, 8, 13),
                                                dtype = np.float64)}

        return observation_spaces[observation_mode]
    
    def set_shape(self):
        # I will pass the possible actions into the environment.
        # The possible actions are state dependent 
        #   -> different amount of actions in each state 
        #   -> No constant value possible / eventually is the encoding dimension possible 
        #   -> here one uci action. third dimension depends on action_wise or symbol_wise encoding
        if self.action_encoding_method == "action_wise":
            # (one action, batch_size, 4032 possible actions)
            action_shape = (1, 1, 4032)
        elif self.action_encoding_method == "symbol_wise":
            # (one action, uci format, 16 possible action symbols)
            action_shape = (1, 4, 16)
        
        self._shape = self.board_space.shape, action_shape

    @staticmethod
    def _get_possible_actions()-> Tuple[List, List]:
        """returns all permuations of chess board squares and all used symbols

        Returns:
            Tuple[List, List]: _description_
        """
        nums = np.linspace(1, 8, 8)
        letters = "abcdefgh"

        all_symbols = str(12345678)+ letters
        
        
        bar = Bar('Create possible actions', max=(len(nums)**2 * len(letters)**2) - len(nums) * len(letters) )
        actions = []
        for pick_letter in letters:
            for pick_num in range(1, 9):
                for place_letter in letters:
                    for place_num in range(1, 9):
                        if pick_letter == place_letter and pick_num == place_num:
                            continue
                        bar.next()
                        actions.append(pick_letter + str(pick_num) + place_letter + str(place_num))
        bar.finish()
                        
        return actions, all_symbols

    def one_hot_encoding(self, actions, method_overwrite: str = None):
        encoding_dict = {   "action_wise": self.action_wise_encoding,
                            "symbol_wise": self.symbol_wise_encodings }
        try:
            if method_overwrite is not None:
                return encoding_dict[method_overwrite](actions)
            else:
                return encoding_dict[self.action_encoding_method](actions)
        except KeyError:
            raise NotImplementedError(f"only 'action_wise' and symbol_wise' are supported methods. You chose {method_overwrite}")    
       
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
            encodings.append(self.symbol_wise_encoding(action))

        return torch.stack(encodings)

    def sample(self) -> Tuple[torch.Tensor, torch.tensor]:
        """sample from board space and from possible actions 

        Returns:
            _type_: _description_
        """
        # sample form board space
        board_sample = self.board_space.sample()

        # sample from action space
        action_sample = self.one_hot_encoding(random.choices(self.possible_actions, k=random.randint(1, 24)))
        return board_sample, action_sample