from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

from src.chess import Board, EmptyPiece, Knight, PiecePosition, PieceType
from src.genetic import (
    AbsoluteFitness,
    ChromosomeGenerator,
    Mutation,
    Runner,
    StoppingCriteria,
)


class GeneSet(Tuple[PiecePosition, ...]):
    """Gene set for the knight attack problem."""


class Chromosome(Board):
    """Chromosome for the knight attack problem."""


class KnightAttackFitness(AbsoluteFitness):
    """Fitness function of a chromosome for the knight attack problem. The fitness is
    the number of squares on the board attacked by the knights. So the  maximal fitness
    is the number of squares on the board.
    """

    def __call__(self, chromosome: Chromosome) -> float:

        attacked_positions = set()

        for position in chromosome.positions():

            if chromosome[position.y, position.x].piece_type == PieceType.KNIGHT:
                attacked_positions.update(Knight().get_attacks(position, chromosome))

        return len(attacked_positions)


class KnightAttackMutation(Mutation):
    """Mutation function for the graph colouring problem."""

    def __init__(
        self, fitness: KnightAttackFitness, gene_set: GeneSet, max_knights: int
    ):
        super().__init__(fitness, gene_set)
        self.max_knights = max_knights

    def __call__(self, parent: Chromosome) -> Chromosome:

        child = parent.copy()

        knight_positions = list(child.positions(piece_type=PieceType.KNIGHT))

        # choose a random new position for the knight
        new_position = random.choice(self.gene_set)

        if len(knight_positions) < self.max_knights:
            # add a knight
            new_position = random.choice(self.gene_set)
            child.place(Knight(), new_position)
        else:

            for ii in range(random.choice(range(4))):
                # choose a random knight position
                knight_position = random.choice(list(knight_positions))

                # move the knight to the new position
                child.move(knight_position, new_position)

        return Chromosome(child)


class KnightAttackStoppingCriteria(StoppingCriteria):
    def __init__(self, target: int, fitness: KnightAttackFitness):
        self.target = target
        self.fitness = fitness

    def __call__(self, chromosome: Chromosome) -> bool:
        """Return true if can stop the genetic algorithm."""

        return self.fitness(chromosome) == self.target


class KnightAttackChromosomeGenerator(ChromosomeGenerator):
    def __init__(self, board) -> None:
        self.board = board

    def __call__(self) -> Chromosome:
        return Chromosome(self.board)


class KnightAttackRunner(Runner):
    def __init__(
        self,
        chromosome_generator: ChromosomeGenerator,
        fitness: KnightAttackFitness,
        stopping_criteria: StoppingCriteria,
        mutate: Mutation,
    ):
        self.chromosome_generator = chromosome_generator
        self.fitness = fitness
        self.stopping_criteria = stopping_criteria
        self.mutate = mutate

    def display(self, candidate):
        time_diff = time.time() - self.start_time

        logging.debug("candidate=%s", str(candidate))
        logging.info("fitness=%f", self.fitness(candidate))
        logging.debug("time=%f", time_diff)


def knight_attack(
    board: Board,
    max_knights: int,
    include_edges: bool = True,
    fitness_stagnation_limit: int = 10000,
) -> Chromosome:

    target = board.total_squares  # all squares on the board attacked by a knight

    # create a gene set of all the positions on the board, excluding the edges
    # this is a trick to help improve performance
    gene_set = GeneSet(tuple(board.positions(include_edges=include_edges)))

    fitness = KnightAttackFitness()
    chromosome_generator = KnightAttackChromosomeGenerator(board)
    stopping_criteria = KnightAttackStoppingCriteria(target, fitness)
    mutate = KnightAttackMutation(fitness, gene_set, max_knights=max_knights)

    runner = KnightAttackRunner(
        chromosome_generator, fitness, stopping_criteria, mutate
    )

    best = runner.run(fitness_stagnation_limit)
    print(best)
    return best


def main():
    """Run the knight attack problem.
    4x4 board requires 6 knights to attack all
    8x8 board requires 14 knights to attack all
    10x10 board requires 22 knights to attack all
    """

    logging_format = (
        "[%(levelname)8s :%(filename)20s:%(lineno)4s - %(funcName)10s] %(message)s"
    )
    logging.basicConfig(format=logging_format, level=logging.INFO)

    board_height, board_width = 8, 8
    max_knights = 14
    include_edges = False  # use edges fo small boards, but not for large boards
    fitness_stagnation_limit = (
        10000  # stop after this many generations of no improvement
    )

    board = Board.empty_board(board_height, board_width)

    knight_attack(board, max_knights, include_edges, fitness_stagnation_limit)


if __name__ == "__main__":
    main()
