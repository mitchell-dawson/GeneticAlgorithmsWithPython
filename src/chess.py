from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class PieceType(Enum):
    """Represents the type of a chess piece."""

    KING = "K"
    QUEEN = "Q"
    ROOK = "R"
    BISHOP = "B"
    KNIGHT = "N"
    PAWN = "P"
    EMPTY = "."


class PieceColour(Enum):
    """Represents the colour of a chess piece."""

    WHITE = "w"
    BLACK = "b"


@dataclass
class PiecePosition:
    """Represents the position of a chess piece."""

    x: int
    y: int

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def xy(self) -> Tuple[int, int]:
        return self.x, self.y

    def yx(self) -> Tuple[int, int]:
        return self.y, self.x

    def __str__(self):
        return f"({self.x},{self.y})"


@dataclass
class Piece:
    """Represents a chess piece."""

    piece_type: PieceType
    colour: PieceColour = PieceColour.WHITE
    captured: bool = False

    def __str__(self):
        if self.colour == PieceColour.WHITE:
            return self.piece_type.value
        else:
            return self.piece_type.value.lower()

    def get_attacks(self, board) -> Tuple[PiecePosition, ...]:
        """Get all the positions that this piece can attack."""
        raise NotImplementedError


@dataclass
class EmptyPiece(Piece):
    """Represents an empty chess piece."""

    piece_type: PieceType = PieceType.EMPTY

    def get_attacks(self, board) -> Tuple[PiecePosition, ...]:
        return tuple()


class Board(np.ndarray):
    """Represents a chess board."""

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    @property
    def height(self) -> int:
        """Get the height of the board"""
        return self.shape[0]

    @property
    def width(self) -> int:
        """Get the width of the board"""
        return self.shape[1]

    @property
    def total_squares(self) -> int:
        """Get the total number of squares on the board"""
        return self.height * self.width

    def positions(
        self, piece_type: Optional[PieceType] = None, include_edges: bool = True
    ):
        """Get all the positions on the board"""

        if include_edges:
            y_range = range(0, self.height)
            x_range = range(0, self.width)
        else:
            y_range = range(1, self.height - 1)
            x_range = range(1, self.width - 1)

        for y in y_range:
            for x in x_range:

                if piece_type is None:
                    yield PiecePosition(x, y)
                else:
                    if self[y, x].piece_type == piece_type:
                        yield PiecePosition(x, y)

    def __str__(self) -> str:
        """return a visualisation of the chessboard"""
        return "\n" + "\n".join(
            ["".join([str(piece).rjust(3) for piece in row]) for row in self]
        )

    def __repr__(self) -> str:
        """return a visualisation of the chessboard"""
        return "\n" + "\n".join(
            ["".join([str(piece).rjust(3) for piece in row]) for row in self]
        )

    def place(self, piece: Piece, position: PiecePosition):
        """Place a piece on the board at the given position."""
        self[position.y, position.x] = piece

    def move(self, old_position: PiecePosition, new_position: PiecePosition):
        """Move a piece to a new position."""
        piece = self[old_position.y, old_position.x]
        self.place(piece, new_position)
        self.place(EmptyPiece(), old_position)

    @staticmethod
    def empty_board(height: int = 8, width: int = 8):
        """Create an empty chess board."""
        return Board(np.full((height, width), Piece(PieceType.EMPTY)))


@dataclass
class Knight(Piece):
    """Represents a knight chess piece."""

    piece_type: PieceType = PieceType.KNIGHT

    def get_attacks(
        self, position: PiecePosition, board: Board
    ) -> Tuple[PiecePosition, ...]:
        """Get all the positions that this piece can attack. First look for all
        possible attacks, then filter out the ones that are off the board.
        """

        assert position is not None
        x, y = position.xy()

        all_possible_attacks = [
            PiecePosition(x + 1, y + 2),
            PiecePosition(x + 2, y + 1),
            PiecePosition(x + 2, y - 1),
            PiecePosition(x + 1, y - 2),
            PiecePosition(x - 1, y - 2),
            PiecePosition(x - 2, y - 1),
            PiecePosition(x - 2, y + 1),
            PiecePosition(x - 1, y + 2),
        ]

        in_bounds_positions_that_can_be_attacked = [
            position
            for position in all_possible_attacks
            if 0 <= position.x < board.width and 0 <= position.y < board.height
        ]

        return tuple(in_bounds_positions_that_can_be_attacked)
