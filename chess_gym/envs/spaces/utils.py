from itertools import product
from typing import List
import numpy as np
from tqdm import tqdm
import chess

from chess_gym.envs.chess_config import NUM_ACTIONS, PIECE_SYMBOLS, Pieces


def get_possible_actions(return_numeric: bool = False):
    """
    Generates all possible chess actions by creating permutations of moves between squares
    on the chessboard, including potential promotion moves for pawns. Actions can be
    returned in either a numeric format (arrays of integers) or as human-readable UCI
    (Universal Chess Interface) strings.

    Args:
        return_numeric (bool): If True, returns actions as numeric arrays where each element
                               is represented numerically. If False, returns actions as UCI
                               strings (e.g., 'e2e4').

    Returns:
        tuple:
            - If `return_numeric` is True, returns a tuple of:
                - actions (np.ndarray): Array of possible moves as numeric arrays representing
                                        board positions and optional promotion piece.
                - components (tuple): A tuple containing (letters, nums, letters, nums, promo_pieces),
                                      representing all components used for generating moves.
            - If `return_numeric` is False, returns a tuple of:
                - actions (list): List of possible UCI strings representing moves and promotions.
                - components (tuple): A tuple of components (letters, nums, letters, nums, promo_pieces)
                                      used in move generation.
    """
    nums = np.linspace(1, 8, 8, dtype=int)
    if return_numeric:
        letters = np.linspace(1, 8, 8, dtype=int)
        promo_pieces = np.linspace(0, 5, 6, dtype=int)
    else:
        letters = "abcdefgh"
        promo_pieces = "prnbq"

    bar = tqdm(desc="Create possible actions", total=NUM_ACTIONS)
    actions = []
    for pick_letter, pick_num, place_letter, place_num in product(
        letters, nums, letters, nums
    ):
        if pick_letter == place_letter and pick_num == place_num:
            continue

        bar.update(1)
        if return_numeric:
            uci_action = np.array([pick_letter, pick_num, place_letter, place_num, 0])
        else:
            uci_action = pick_letter + str(pick_num) + place_letter + str(place_num)

        actions.append(uci_action)

        # get promotions
        if place_num == 1 or place_num == 8:
            for piece in promo_pieces:
                if return_numeric:
                    uci_action[-1] = piece
                    actions.append(uci_action.copy())
                else:
                    actions.append(uci_action + piece)
                bar.update(1)
    bar.close()

    if return_numeric:
        return np.stack(actions).squeeze(), (letters, nums, letters, nums, promo_pieces)
    else:
        return actions, (letters, nums, letters, nums, promo_pieces)


def piece_map2fen(board: np.ndarray) -> str:
    """
    Converts a piece map to FEN notation.

    Args:
        board (np.ndarray): An (8, 8) array representing pieces on a chessboard.

    Returns:
        str: The FEN string representation of the board.
    """
    fen = ""
    for row in range(board.shape[0]):
        count = 0
        for col in range(board.shape[0]):
            if board[row, col] == 0:
                count += 1
                continue
            if count > 0:
                fen += str(count)
            fen += PIECE_SYMBOLS[board[row, col]]
            count = 0
        if count > 0:
            fen += str(count)
        fen += "/"
    return fen[:-1]


def piece_map2rgb(board: np.ndarray) -> np.ndarray:
    """
    Converts a piece map to an RGB representation.

    Args:
        board (np.ndarray): An (8, 8) piece map array.

    Returns:
        np.ndarray: RGB representation of the board.
    """
    raise NotImplementedError
    fen = piece_map2fen(board)
    board = chess.Board(fen)


def piece_map2one_hot(board: np.ndarray) -> np.ndarray:
    """
    Converts a piece map to a one-hot encoding representation.

    Args:
        board (np.ndarray): An (8, 8) array representing pieces on a chessboard.

    Returns:
        np.ndarray: An (8, 8, 12) array with one-hot encoding of pieces.
    """
    encoding = np.zeros((8, 8, 12))
    for idx, piece in enumerate(Pieces):
        encoding[(board == piece.value), idx] = 1
    return encoding.astype(float)


def contains_rgb_array(board: np.ndarray) -> bool:
    """
    Validates if a given board is a valid RGB array representation.

    Args:
        board (np.ndarray): Array to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    raise NotImplementedError


def contains_piece_map(board: np.ndarray) -> bool:
    """
    Validates if a given board is a valid piece map.

    Args:
        board (np.ndarray): Array to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    return board.shape == (8, 8) and board.max() <= 12 and board.min() >= 0


def contains_one_hot(board: np.ndarray) -> bool:
    """
    Validates if a given board is a valid one-hot encoded representation.

    Args:
        board (np.ndarray): Array to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    return board.shape == (8, 8, 12) and set(np.unique(board).astype(int)) == {0, 1}


def contains_fen(board: str) -> bool:
    """
    Validates if a given string is a valid FEN representation.

    Args:
        board (str): FEN string to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    for row in board.split("/"):
        count = 0
        for col in row:
            col: str
            if col.isnumeric():
                count += int(col)
            else:
                count += 1
        if count != 8:
            return False
    return True

def fen2piece_map(fen: str) -> np.ndarray:
    piece_symbols_reversed = {v: k for k, v in PIECE_SYMBOLS.items()}
    board = np.zeros((8, 8))
    for row_idx, row in enumerate(fen.split("/")):
        col_idx = 0
        for col in row:
            if col.isnumeric():
                col_idx += int(col)
                continue
            board[row_idx, col_idx] = piece_symbols_reversed[col]
    return board

def get_numeric_action(action: np.ndarray, numeric_symbol_spaces: List[np.ndarray]) -> np.ndarray:
    offset = 0
    out = np.empty(len(numeric_symbol_spaces))
    for idx, numeric_space in enumerate(numeric_symbol_spaces):
        number = numeric_space[action[offset: offset + len(numeric_space)].astype(bool)]
        out[idx] = number.item()
        offset += len(numeric_space)
    return out