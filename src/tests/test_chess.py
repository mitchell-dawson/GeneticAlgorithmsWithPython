import numpy as np

from src.chess import Board, Knight, Piece, PieceColour, PiecePosition, PieceType


def test_piece_type():
    """GIVEN a piece type
    WHEN the value is requested
    THEN the correct value is returned
    """

    piece_type = PieceType.KING
    assert piece_type.value == "K"

    piece_type = PieceType.ROOK
    assert piece_type.value == "R"


def test_piece_colour():
    """GIVEN a piece colour
    WHEN the value is requested
    THEN the correct value is returned
    """
    piece_colour = PieceColour.WHITE
    assert piece_colour.value == "w"

    piece_colour = PieceColour.BLACK
    assert piece_colour.value == "b"


def test_piece():
    """GIVEN a piece type and colour
    WHEN the piece is created
    THEN the correct piece type and colour are returned
    """

    piece_type = PieceType.KING
    piece_colour = PieceColour.WHITE
    piece = Piece(piece_type, piece_colour)
    assert str(piece) == "K"

    piece_type = PieceType.ROOK
    piece_colour = PieceColour.BLACK
    piece = Piece(piece_type, piece_colour)
    assert str(piece) == "r"


################################################################################
### Board tests ################################################################
################################################################################


def test_empty_Board():
    """GIVEN a board
    WHEN the board is created
    THEN the board is empty
    """

    board = Board.empty_board()
    assert board.shape == (8, 8)
    assert board[0, 0] == Piece(PieceType.EMPTY)


def test_str_Board():
    """GIVEN a board
    WHEN the board is created
    THEN the board is empty
    """

    board = Board.empty_board()
    print(board)


def test_str_with_pieces_Board():
    """GIVEN a board with some pieces on it
    WHEN the board is printed
    THEN the board is printed with the pieces on it
    """

    board = Board.empty_board()

    piece_1 = Piece(PieceType.KING, PieceColour.WHITE)
    board.place(piece_1, PiecePosition(x=0, y=0))

    piece_2 = Piece(PieceType.ROOK, PieceColour.BLACK)
    board.place(piece_2, PiecePosition(x=3, y=1))

    print(board)


def test_place_piece_Board():
    """GIVEN a board
    WHEN a piece is placed on the board
    THEN the piece is placed on the board
    """

    board = Board.empty_board()
    piece = Piece(PieceType.KING, PieceColour.WHITE)
    board.place(piece, PiecePosition(x=0, y=0))
    assert board[0, 0] == piece


def test_place_piece_two_pieces_Board():
    """GIVEN a board
    WHEN two pieces are placed on the board
    THEN the pieces found d on the board
    """

    board = Board.empty_board()

    piece_1 = Piece(PieceType.KING, PieceColour.WHITE)
    board.place(piece_1, PiecePosition(x=0, y=0))

    piece_2 = Piece(PieceType.ROOK, PieceColour.BLACK)
    board.place(piece_2, PiecePosition(x=1, y=1))

    assert board[0, 0] == piece_1
    assert board[1, 1] == piece_2


def test_place_piece_replace_a_piece_Board():
    """GIVEN a board
    WHEN a piece is placed on the board where another piece is already placed
    THEN the new piece is placed on the board
    """

    board = Board.empty_board()

    piece_1 = Piece(PieceType.KING, PieceColour.WHITE)
    board.place(piece_1, PiecePosition(x=0, y=0))

    assert board[0, 0] == piece_1

    piece_2 = Piece(PieceType.ROOK, PieceColour.BLACK)
    board.place(piece_2, PiecePosition(x=0, y=0))

    assert board[0, 0] == piece_2


def test_move_Board():
    """GIVEN a board
    WHEN a piece is moved on the board
    THEN the piece is moved on the board
    """

    board = Board.empty_board()
    position = PiecePosition(x=0, y=0)
    piece = Knight()
    board.place(piece, position)
    assert board[0, 0] == piece

    board.move(position, PiecePosition(x=1, y=1))
    assert board[0, 0].piece_type == PieceType.EMPTY
    assert board[1, 1] == piece


################################################################################
### Knight tests ###############################################################
################################################################################


def test_init_Knight():
    """GIVEN a knight
    WHEN the knight is created
    THEN the knight is created with the correct piece type and colour
    """

    knight = Knight(colour=PieceColour.WHITE)
    assert str(knight) == "N"

    knight = Knight(colour=PieceColour.BLACK)
    assert str(knight) == "n"


def test_get_attacks_centre_of_board_Knight():
    """GIVEN a knight
    WHEN the knight is placed in the centre of the board
    THEN the knight has 8 possible attacks
    """

    board = Board.empty_board(8, 8)
    position = PiecePosition(x=3, y=3)
    knight = Knight()
    attacks = knight.get_attacks(position, board)

    assert len(attacks) == 8


def test_get_attacks_edge_of_board_Knight():
    """GIVEN a knight
    WHEN the knight is placed on the edge of the board
    THEN the knight has 4    possible attacks
    """

    board = Board.empty_board(8, 8)
    position = PiecePosition(x=0, y=3)
    knight = Knight()
    attacks = knight.get_attacks(position, board)

    assert len(attacks) == 4


def test_get_attacks_corner_of_board_Knight():
    """GIVEN a knight
    WHEN the knight is placed on the corner of the board
    THEN the knight has 2 possible attacks
    """

    board = Board.empty_board(8, 8)
    position = PiecePosition(x=0, y=0)
    knight = Knight()
    attacks = knight.get_attacks(position, board)

    assert len(attacks) == 2
