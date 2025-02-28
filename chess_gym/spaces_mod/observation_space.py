from io import BytesIO
from typing import Dict, Tuple
import cairosvg
import chess.svg
from gymnasium.spaces import Space, Box
from enum import Enum

import numpy as np

from chess_gym.chess_config import BoardEncoding, PieceMaxOccurance, Pieces
from chess_gym.spaces_mod.action_space import ChessAction
from chess_gym.spaces_mod.utils import contains_fen, contains_one_hot, contains_piece_map, contains_rgb_array, fen2piece_map, piece_map2fen, piece_map2one_hot, piece_map2rgb
from PIL import Image

class BoardSpace(Space):
    """
    Custom space representing a chess board in various encoding formats.

    Attributes:
        _encoding (BOARD_ENCODING): The encoding format for the board.
        encoding_funcs (dict): Functions for encoding the board.
        contains_funcs (dict): Functions for validating board contents.
    """

    def __init__(self, encoding: BoardEncoding = BoardEncoding.PIECE_MAP, seed=None):
        shape = None
        dtype = None
        self._encoding = encoding
        if self._encoding == BoardEncoding.RGB_ARRAY:
            shape = (3, 300, 300)
            dtype = float
        elif self._encoding == BoardEncoding.PIECE_MAP:
            shape = (8, 8)
            dtype = float
        elif self._encoding == BoardEncoding.ONE_HOT:
            shape = (8, 8, 12)
            dtype = float
        elif self._encoding == BoardEncoding.FEN:
            shape = None
            dtype = str

        super().__init__(shape, dtype, seed)
        self._all_pieces = self._get_all_pieces()
        self._all_tiles = self._get_all_tiles()

        self.encoding_funcs = {
            BoardEncoding.RGB_ARRAY: piece_map2rgb,
            BoardEncoding.PIECE_MAP: lambda x: x,
            BoardEncoding.ONE_HOT: piece_map2one_hot,
            BoardEncoding.FEN: piece_map2fen,
        }

        self.contains_funcs = {
            BoardEncoding.RGB_ARRAY: contains_rgb_array,
            BoardEncoding.PIECE_MAP: contains_piece_map,
            BoardEncoding.ONE_HOT: contains_one_hot,
            BoardEncoding.FEN: contains_fen,
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
        if (figures == Pieces.BISHOP_B.value).sum() == 2:
            # place figures
            board[*black_tile[0]] = Pieces.BISHOP_B.value
            board[*white_tile[0]] = Pieces.BISHOP_B.value

            # update_mask
            bishop_mask[black_indices[0]] = 0
            bishop_mask[white_indices[0]] = 0

            # pop bishops
            figures = np.delete(figures, figures == Pieces.BISHOP_B.value)
        if (figures == Pieces.BISHOP_W.value).sum() == 2:
            # place figures
            board[*black_tile[1]] = Pieces.BISHOP_W.value
            board[*white_tile[1]] = Pieces.BISHOP_W.value

            # update_mask
            bishop_mask[black_indices[1]] = 0
            bishop_mask[white_indices[1]] = 0

            # pop bishops
            figures = np.delete(figures, figures == Pieces.BISHOP_W.value)

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

    def encode(self, board: chess.Board) -> str | np.ndarray:
        fen = board.fen()
        fen = fen.split(" ")[0]
        if self._encoding == BoardEncoding.FEN:
            return fen
        peice_map = fen2piece_map(fen)
        return self.encoding_funcs[self._encoding](peice_map)


    def _get_all_pieces(self) -> np.ndarray:
        """
        Retrieves an array of all pieces with maximum occurrences.

        Returns:
            np.ndarray: Array of all pieces by maximum occurrence.
        """
        all_pieces = []
        for piece in Pieces:
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
        board_encoding: BoardEncoding = BoardEncoding.PIECE_MAP,
        shape=None,
        dtype=None,
        seed=None,
    ):
        super().__init__(shape, dtype, seed)
        self._action_space = action_space
        self._board_encoding = board_encoding

        self._board_space = BoardSpace(encoding=BoardEncoding.PIECE_MAP)

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
    
    def encode(self, board: chess.Board) -> np.ndarray | str:
        return self._board_space.encode(board)
