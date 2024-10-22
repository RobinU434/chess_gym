from typing import List, Literal, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
from progress.bar import Bar

import chess
import chess.svg

import numpy as np

from io import BytesIO
import cairosvg
from PIL import Image

from sklearn.preprocessing import OneHotEncoder
import torch

from chess_gym.envs.observation_space import ChessSpace
from chess_gym.envs.chess_config import piece_index, NUM_ACTIONS


class MoveSpace:
    def __init__(self, board):
        self.board = board

    def sample(self):
        return np.random.choice(list(self.board.legal_moves))


class ChessEnv(gym.Env):
    """Chess Environment"""

    metadata = {
        "render.modes": ["rgb_array", "human"],
        "observation.modes": ["rgb_array", "piece_map"],
    }

    def __init__(
        self,
        render_size=512,
        board_encoding="rgb_array",
        claim_draw=True,
        chess960: bool = False,
        action_encoding_method="action_wise",
        **kwarg,
    ):
        super(ChessEnv, self).__init__()

        self.action_encoding_method = action_encoding_method
        self.board_enccoding = board_encoding
        self.possible_actions, self.action_symbols = self._get_possible_actions()
        self.action_encodings = self.one_hot_encoding(self.possible_actions)
        self.observation_space = self._get_observation_space(render_size)
        self.onehot_encoder = OneHotEncoder(sparse_output=False)
        self.chess960 = chess960

        self.board = self._setup_board(self.chess960)

        self.render_size = render_size
        self.claim_draw = claim_draw

        self.viewer = None

        self.action_space = spaces.Discrete(
            64 * 64
        )  # each number represents the transition from a sqaure to another square
        # self.observation_space = ChessSpace(board=self, observation_mode=board_encoding)

    @property
    def n_action_symbols(self):
        return len(self.action_symbols)

    @property
    def n_actions(self):
        """len of all possible actions"""
        return len(self.possible_actions)

    @staticmethod
    def _setup_board(chess960):
        board = chess.Board(chess960=chess960)

        if chess960:
            board.set_chess960_pos(np.random.randint(0, 960))

        return board

    def _get_observation_space(self, render_size) -> spaces.Dict:

        board_obs_space = {
            "rgb_array": spaces.Box(
                low=0, high=255, shape=(render_size, render_size, 3), dtype=np.float64
            ),
            "piece_map": spaces.Box(low=0, high=1, shape=(8, 8, 13), dtype=np.float64),
        }

        action_obs_space = {
            "action_wise": spaces.Box(
                low=0, high=1, shape=(1, 1, self.n_actions), dtype=np.float64
            ),
            "symbol_wise": spaces.Box(
                low=0, high=1, shape=(1, 5, 16), dtype=np.float64
            ),
        }

        observation_space = spaces.Dict(
            {
                "board": board_obs_space[self.board_enccoding],
                "actions": action_obs_space[self.action_encoding_method],
            }
        )

        return observation_space

    def _get_image(self):
        out = BytesIO()
        bytestring = chess.svg.board(self.board, size=self.render_size).encode("utf-8")
        cairosvg.svg2png(bytestring=bytestring, write_to=out)
        image = Image.open(out)
        print(image)
        return np.asarray(image)

    def linear_board(self):
        """pass through functions

        Args:
            x (any): object to pass through

        Returns:
            any:
        """
        piece_map = np.zeros(64)

        for square, piece in zip(
            self.board.piece_map().keys(), self.board.piece_map().values()
        ):
            piece_map[square] = piece_index[str(piece)]

        piece_map = piece_map.reshape(8, 8)

        return piece_map

    def onehot_board(self):
        """returns one hot encoded matrix of game board

        Args:
            x (np.array): game board with shape 8x8

        Returns:
            np.array: one hot encoded game board with shape (8x8x13)
        """
        piece_map = np.zeros((64, 13))

        for square, piece in zip(
            self.board.piece_map().keys(), self.board.piece_map().values()
        ):
            piece_map[square, piece_index[str(piece)]] = 1

        piece_map = piece_map.reshape(8, 8, 13)  # reshape into board shape

        return piece_map

    def _get_piece_configuration(self, encoding: str = "onehot"):
        encoding_dict = {"linear": self.linear_board, "onehot": self.onehot_board}

        piece_map = encoding_dict[encoding]()

        # flip piece_map for the same orientation as for self.board
        piece_map = np.flip(piece_map, axis=0)

        return piece_map

    @staticmethod
    def _get_possible_actions() -> Tuple[List, List]:
        """returns all permuations of chess board squares and all used symbols

        Returns:
            Tuple[List, List]: _description_
        """
        nums = np.linspace(1, 8, 8)
        letters = "abcdefgh"
        promo_pieces = "prnbq"

        all_symbols = str(12345678) + letters

        bar = Bar("Create possible actions", max=NUM_ACTIONS)
        actions = []
        for pick_letter in letters:
            for pick_num in range(1, 9):
                for place_letter in letters:
                    for place_num in range(1, 9):
                        if pick_letter == place_letter and pick_num == place_num:
                            continue
                        bar.next()
                        uci_action_str = (
                            pick_letter + str(pick_num) + place_letter + str(place_num)
                        )
                        actions.append(uci_action_str)

                        # get promotions
                        if place_num == 1 or place_num == 8:
                            for piece in promo_pieces:
                                actions.append(uci_action_str + piece)
                                bar.next()

        bar.finish()

        return actions, all_symbols

    def one_hot_encoding(self, actions, method_overwrite: str = None):
        encoding_dict = {
            "action_wise": self.action_wise_encoding,
            "symbol_wise": self.symbol_wise_encodings,
        }
        try:
            if method_overwrite is not None:
                return encoding_dict[method_overwrite](actions)
            else:
                return encoding_dict[self.action_encoding_method](actions)
        except KeyError:
            raise NotImplementedError(
                f"only 'action_wise' and symbol_wise' are supported methods. You chose {method_overwrite}"
            )

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

    @staticmethod
    def movesToUCI(moves: chess.Move):
        return list(map(lambda x: str(x), moves))

    def _observe(self, encoding: str = "one_hot") -> Tuple[torch.Tensor, torch.Tensor]:
        # return board state
        observation_dict = {
            "linear": self._get_image,
            "one_hot": self._get_piece_configuration,
        }

        # return legal moves
        legal_moves = list(self.board.legal_moves)
        legal_moves = list(
            map(lambda x: str(x), legal_moves)
        )  # convert move object into uci move str
        # encode legal moves
        legal_moves = self.one_hot_encoding(legal_moves)

        obs = dict(board=observation_dict[encoding](), actions=legal_moves)

        return obs

    def _ToAction(self, action):
        if isinstance(action, str):
            # only accept uci format e.g.: e3e4
            assert action[0] in "abcdefgh"
            assert int(action[1]) in [1, 2, 3, 4, 5, 6, 7, 8]
            assert action[2] in "abcdefgh"
            assert int(action[3]) in [1, 2, 3, 4, 5, 6, 7, 8]

        if isinstance(action, [int, np.int64]):
            # get uci encoding from current legal moves
            legal_moves = self.movesToUCI(self.movesToUCI(self.board.legal_moves))
            # action is the index where an element from self.possible_action is 1
            action = legal_moves[action]

        # convert action
        return chess.Move.from_uci(action)

    def _action_to_move(self, action):
        from_square = chess.Square(action[0])
        to_square = chess.Square(action[1])
        promotion = (
            None if action[2] == 0 else chess.Piece(chess.PieceType(action[2])),
            chess.Color(action[4]),
        )
        drop = (
            None if action[3] == 0 else chess.Piece(chess.PieceType(action[3])),
            chess.Color(action[5]),
        )
        move = chess.Move(from_square, to_square, promotion, drop)
        return move

    def _move_to_action(self, move):
        from_square = move.from_square
        to_square = move.to_square
        promotion = 0 if move.promotion is None else move.promotion
        drop = 0 if move.drop is None else move.drop
        return [from_square, to_square, promotion, drop]

    def step(
        self, action
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], float, bool, dict]:
        # convert action in as str into move object
        action = self._ToAction(action)
        # print("before_action \n", self._get_piece_configuration(encoding="linear"))
        self.board.push(action)
        # print("after_action \n", self.board)

        observation = self._observe()
        result = self.board.result()
        reward = 1 if result == "1-0" else -1 if result == "0-1" else 0
        terminal = self.board.is_game_over(claim_draw=self.claim_draw)
        info = {
            "turn": self.board.turn,
            "castling_rights": self.board.castling_rights,
            "fullmove_number": self.board.fullmove_number,
            "halfmove_clock": self.board.halfmove_clock,
            "promoted": self.board.promoted,
            "chess960": self.board.chess960,
            "ep_square": self.board.ep_square,
        }

        return observation, reward, terminal, info

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.board.reset()

        if self.chess960:
            self.board.set_chess960_pos(np.random.randint(0, 960))

        return self._observe()

    def render(self, mode=Literal["rgb_array", "human"]):
        img = self._get_image()
        if mode == "rgb_array":
            return img
        elif mode == "human":
            from gymnasium.envs.classic_control import rendering

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()

            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if not self.viewer is None:
            self.viewer.close()
