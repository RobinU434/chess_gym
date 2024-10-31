from itertools import product
import numpy as np
from tqdm import tqdm

from chess_gym.envs.chess_config import NUM_ACTIONS


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
