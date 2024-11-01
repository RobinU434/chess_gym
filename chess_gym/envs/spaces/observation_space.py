from typing import Dict
import chess.svg
from gymnasium.spaces import Space, Box
from enum import Enum

import numpy as np

from chess_gym.envs.spaces.action_space import ChessAction
from chess_gym.envs.spaces.utils import contains_fen, contains_one_hot, contains_piece_map, contains_rgb_array, piece_map2fen, piece_map2one_hot, piece_map2rgb


class BOARD_ENCODING(Enum):
    """
    Enum for board encoding types.

    Attributes:
        rgb_array (int): Encodes the board as an RGB array.
        piece_map (int): Encodes the board as a piece map array.
        one_hot (int): Encodes the board as a one-hot array.
        fen (int): Encodes the board as a FEN string.
    """

    rgb_array = 0
    piece_map = 1
    one_hot = 2
    fen = 3


class PIECES(Enum):
    """
    Enum for chess pieces with numeric values corresponding to black and white pieces.

    Attributes:
        king_b, queen_b, rook_b, etc. (int): Values representing black pieces.
        king_w, queen_w, rook_w, etc. (int): Values representing white pieces.
    """

    # 0 is for an empty field but this is no piece
    KING_B = 1
    QUEEN_B = 2
    ROOK_B = 3
    BISHOP_B = 4
    KNIGHT_B = 5
    PAWN_B = 6
    KING_W = 7
    QUEEN_W = 8
    ROOK_W = 9
    BISHOP_W = 10
    KNIGHT_W = 11
    PAWN_W = 12


class PieceMaxOccurance(Enum):
    """
    Enum for the maximum occurrence of each piece type in a chess game.

    Attributes:
        king_b, queen_b, rook_b, etc. (int): Max occurrences for black pieces.
        king_w, queen_w, rook_w, etc. (int): Max occurrences for white pieces.
    """

    KING_B = 1
    QUEEN_B = 1
    ROOK_B = 2
    BISHOP_B = 2
    KNIGHT_B = 2
    PAWN_B = 8
    KING_W = 1
    QUEEN_W = 1
    ROOK_W = 2
    BISHOP_W = 2
    KNIGHT_W = 2
    PAWN_W = 8


PIECE_SYMBOLS = {
    1: "K",
    2: "Q",
    3: "R",
    4: "B",
    5: "N",
    6: "P",
    7: "k",
    8: "q",
    9: "r",
    10: "b",
    11: "n",
    12: "p",
}


class BoardSpace(Space):
    """
    Custom space representing a chess board in various encoding formats.

    Attributes:
        _encoding (BOARD_ENCODING): The encoding format for the board.
        encoding_funcs (dict): Functions for encoding the board.
        contains_funcs (dict): Functions for validating board contents.
    """

    def __init__(self, encoding: BOARD_ENCODING = BOARD_ENCODING.piece_map, seed=None):
        shape = None
        dtype = None
        self._encoding = encoding
        if self._encoding == BOARD_ENCODING.rgb_array:
            shape = (3, 300, 300)
            dtype = float
        elif self._encoding == BOARD_ENCODING.piece_map:
            shape = (8, 8)
            dtype = float
        elif self._encoding == BOARD_ENCODING.one_hot:
            shape = (8, 8, 12)
            dtype = float
        elif self._encoding == BOARD_ENCODING.fen:
            shape = None
            dtype = str

        super().__init__(shape, dtype, seed)
        self._all_pieces = self._get_all_pieces()
        self._all_tiles = self._get_all_tiles()

        self.encoding_funcs = {
            BOARD_ENCODING.rgb_array: piece_map2rgb,
            BOARD_ENCODING.piece_map: lambda x: x,
            BOARD_ENCODING.one_hot: piece_map2one_hot,
            BOARD_ENCODING.fen: piece_map2fen,
        }

        self.contains_funcs = {
            BOARD_ENCODING.rgb_array: contains_rgb_array,
            BOARD_ENCODING.piece_map: contains_piece_map,
            BOARD_ENCODING.one_hot: contains_one_hot,
            BOARD_ENCODING.fen: contains_fen,
        }

    def sample(self, mask: np.ndarray = None) -> np.ndarray:
        """
        Samples a board configuration based on the encoding type.

        Args:
            mask (np.ndarray, optional): Optional binary (8, 8) array mask for sampling.

        Returns:
            np.ndarray or str: Encoded sample of the board.
        """
        # sample figures on board
        n_figures = round(np.random.random_sample() * len(self._all_pieces))
        figures = np.random.choice(self._all_pieces, size=n_figures, replace=False)

        board = np.zeros((8, 8))

        # for placing bishops
        binary_tiles = self._all_tiles.sum(axis=1) % 2 == 0
        black_tiles = self._all_tiles[binary_tiles]
        white_tiles = self._all_tiles[~binary_tiles]
        black_indices = np.random.choice(len(black_tiles), size=2, replace=False)
        white_indices = np.random.choice(len(white_tiles), size=2, replace=False)
        black_tile = black_tiles[black_indices]
        white_tile = white_tiles[white_indices]

        # sample placement on board
        # if there are two bishops of the same color place them first
        bishop_mask = np.ones(len(self._all_tiles))
        if (figures == PIECES.BISHOP_B.value).sum() == 2:
            # place figures
            board[*black_tile[0]] = PIECES.BISHOP_B.value
            board[*white_tile[0]] = PIECES.BISHOP_B.value

            # update_mask
            bishop_mask[black_indices[0]] = 0
            bishop_mask[white_indices[0]] = 0

            # pop bishops
            figures = np.delete(figures, figures == PIECES.BISHOP_B.value)
        if (figures == PIECES.bishop_w.value).sum() == 2:
            # place figures
            board[*black_tile[1]] = PIECES.bishop_w.value
            board[*white_tile[1]] = PIECES.bishop_w.value

            # update_mask
            bishop_mask[black_indices[1]] = 0
            bishop_mask[white_indices[1]] = 0

            # pop bishops
            figures = np.delete(figures, figures == PIECES.bishop_w.value)

        if mask is None:
            mask = bishop_mask

        else:
            mask = mask.reshape(-1) * bishop_mask
        mask /= mask.sum()

        # sample remaining tiles to place figures
        remaining_tiles = self._all_tiles[
            np.random.choice(
                self._all_tiles.shape[0], size=(len(figures), 2), replace=False, p=mask
            )
        ]
        for tile, figure in zip(remaining_tiles, figures):
            board[*tile] = figure

        return self.encoding_funcs[self._encoding](board.astype(float))

    def contains(self, x: np.ndarray) -> bool:
        """
        Checks if a given board is valid for the defined encoding.

        Args:
            x (np.ndarray): The board to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        return self.contains_funcs[self._encoding](x)


    def _get_all_pieces(self) -> np.ndarray:
        """
        Retrieves an array of all pieces with maximum occurrences.

        Returns:
            np.ndarray: Array of all pieces by maximum occurrence.
        """
        all_pieces = []
        for piece in PIECES:
            all_pieces.extend(
                [
                    piece.value
                    for _ in range(getattr(PieceMaxOccurance, piece.name).value)
                ]
            )
        return np.array(all_pieces)

    def _get_all_tiles(self) -> np.ndarray:
        """
        Retrieves coordinates for all 64 tiles on an 8x8 board.

        Returns:
            np.ndarray: Array of all board tile coordinates.
        """
        x, y = np.meshgrid(np.arange(8), np.arange(8))
        coords = np.stack([x, y]).reshape(2, -1).T
        return coords


class ChessObservation(Space):
    """
    Custom observation space for representing chess observations.

    Attributes:
        _action_space (ChessAction): The action space for chess moves.
        _board_encoding (BOARD_ENCODING): Encoding type for the board.
        _board_space (BoardSpace): Space for board configurations.
    """

    def __init__(
        self,
        action_space: ChessAction,
        board_encoding: BOARD_ENCODING = BOARD_ENCODING.piece_map,
        shape=None,
        dtype=None,
        seed=None,
    ):
        super().__init__(shape, dtype, seed)
        self._action_space = action_space
        self._board_encoding = board_encoding

        self._board_space = BoardSpace(encoding=BOARD_ENCODING.piece_map)

    def sample(self, mask: np.ndarray = None) -> Dict[str, str | np.ndarray]:
        """
        Samples an observation containing a board state and legal actions.

        Args:
            mask (np.ndarray, optional): Optional binary (8, 8) mask.

        Returns:
            dict: Sample containing 'board' encoding and 'actions' array.
        """
        board_sample = self._board_space.sample()
        board_encoding = self._board_space.encoding_funcs[self._board_encoding](
            board_sample
        )
        board = chess.Board(piece_map2fen(board_sample), chess960=False)
        actions = np.empty((board.legal_moves.count(), *self._action_space.shape))
        for idx, move in enumerate(board.legal_moves):
            actions[idx] = self._action_space.uci_to_action(move.uci())

        return {"board": board_encoding, "actions": actions}

    def contains(self, x: Dict[str, np.ndarray | str]) -> bool:
        """
        Checks if the observation is valid, including board and actions.

        Args:
            x (dict): Observation containing 'board' and 'actions'.

        Returns:
            bool: True if valid, False otherwise.
        """
        board_ok = self._board_space.contains_funcs[self._board_encoding](x["board"])
        action_ok = bool(
            np.prod([self._action_space.contains(action) for action in x["actions"]])
        )
        return board_ok and action_ok
