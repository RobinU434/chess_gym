from enum import Enum


NUM_ACTIONS = 9072

class BoardEncoding(Enum):
    """
    Enum for board encoding types.

    Attributes:
        rgb_array (int): Encodes the board as an RGB array.
        piece_map (int): Encodes the board as a piece map array.
        one_hot (int): Encodes the board as a one-hot array.
        fen (int): Encodes the board as a FEN string.
    """

    RGB_ARRAY = 0
    PIECE_MAP = 1
    ONE_HOT = 2
    FEN = 3


class Pieces(Enum):
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