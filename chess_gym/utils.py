import numpy as np
from chess_gym.chess_config import PIECE_SYMBOLS


def _build_row(row: str):
    res = []
    for ele in row:
        if ele.isnumeric():
            res.extend(["."] * int(ele))
            continue
        res.append(ele)
    return res


def fen2one_hot(fen: str) -> np.ndarray:
    """_summary_

    Args:
        fen (str): _description_

    Returns:
        np.ndarray: (12, 8, 8)
    """
    char_array = fen.split("/")
    char_array = list(map(_build_row, char_array))
    char_array = np.array(char_array)
    m = np.array(list(PIECE_SYMBOLS.values()))
    char_array = char_array[None].repeat(len(m), axis=0)
    res = (char_array == m[:, None, None]).astype(int)
    return res


def one_hot2piece_map(one_hot: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        one_hot (np.ndarray): (12, 8, 8)

    Returns:
        np.ndarray: (8, 8)
    """
    indices = np.arange(one_hot.shape[0])[:, None, None] + 1
    index_map = one_hot * indices
    piece_map = index_map.sum(axis=0)
    return piece_map


def fen2piece_map(fen: str) -> np.ndarray:
    return one_hot2piece_map(fen2one_hot(fen))
