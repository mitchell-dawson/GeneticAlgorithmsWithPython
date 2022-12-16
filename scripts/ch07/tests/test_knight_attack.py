import numpy as np

from scripts.ch07.knight_attack import (
    KnightAttackChromosome,
    KnightAttackFitness,
    KnightAttackMutation,
    knight_attack,
)
from src.chess import Board, Knight, PiecePosition


def test_knight_attack():
    board_height, board_width = 4, 4
    max_knights = 6

    board = Board.empty_board(board_height, board_width)

    knight_attack(board, max_knights)


def test_call_one_knight_in_centre_of_board_KnightAttackFitness():

    board = Board.empty_board(8, 8)
    position = PiecePosition(x=3, y=3)
    knight = Knight()
    board.place(knight, position)

    chromosome = KnightAttackChromosome(board)

    fitness = KnightAttackFitness()

    assert fitness(chromosome) == 8


def test_call_one_knight_on_edge_of_board_KnightAttackFitness():

    board = Board.empty_board(8, 8)
    position = PiecePosition(x=0, y=3)
    knight = Knight()
    board.place(knight, position)

    chromosome = KnightAttackChromosome(board)

    fitness = KnightAttackFitness()

    assert fitness(chromosome) == 4


def test_call_one_knight_in_corner_of_board_KnightAttackFitness():

    board = Board.empty_board(8, 8)
    position = PiecePosition(x=0, y=0)
    knight = Knight()
    board.place(knight, position)
    chromosome = KnightAttackChromosome(board)

    fitness = KnightAttackFitness()
    assert fitness(chromosome) == 2


def test_call_two_knights_with_some_overlap_KnightAttackFitness():

    board = Board.empty_board(8, 8)

    position = PiecePosition(x=3, y=3)
    knight_1 = Knight()
    board.place(knight_1, position)

    position = PiecePosition(x=4, y=4)
    knight_2 = Knight()
    board.place(knight_2, position)

    print(board)

    chromosome = KnightAttackChromosome(board)

    fitness = KnightAttackFitness()

    # 8 + 8 - 2 = 14
    assert fitness(chromosome) == 14


def test_call_edge_knights_with_some_overlap_KnightAttackFitness():

    board = Board.empty_board(8, 8)

    y = 7

    for x in (2, 4, 6):
        position = PiecePosition(x=x, y=y)
        knight = Knight()
        print(knight.get_attacks(position, board))
        board.place(knight, position)

    print(board)
    chromosome = KnightAttackChromosome(board)

    fitness = KnightAttackFitness()

    # 8 + 8 - 2 = 14
    assert fitness(chromosome) == 8


def test_call_KnightAttackMutation():

    fitness = KnightAttackFitness()

    board = Board.empty_board(1000, 1000)
    gene_set = list(board.positions())

    mutation = KnightAttackMutation(fitness, gene_set, max_knights=1)

    position = PiecePosition(x=3, y=3)
    knight = Knight()
    board.place(knight, position)

    chromosome = KnightAttackChromosome(board)

    mutated_chromosome = mutation(chromosome)

    assert not np.array_equal(mutated_chromosome.genes, chromosome.genes)
