from __future__ import annotations

import sys

# File: genetic.py
#    from chapter 1 of _Genetic Algorithms with Python_
#
# Author: Clinton Sheppard <fluentcoder@gmail.com>
# Copyright (c) 2016 Clinton Sheppard
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.  See the License for the specific language governing
# permissions and limitations under the License.


print(sys.path)


import logging
import random
import time
from abc import ABC, abstractmethod
from bisect import bisect_left
from copy import deepcopy
from math import exp
from typing import Any, List, Optional, Protocol


class Gene(Protocol):
    pass


class Chromosome(Protocol):
    """Chromosome for the genetic algorithm."""

    genes: Any
    age: int = 0


class GeneSet(Protocol):
    """Gene set for the genetic algorithm."""


class Generation:
    pass


class Fitness(ABC):
    """Fitness is the abstract base class for fitness functions."""


class AbsoluteFitness(Fitness):
    """AbsoluteFitness is a fitness function that returns the absolute value
    of the candidate fitness
    """

    @abstractmethod
    def __call__(self, chromosome: Chromosome) -> float:
        """Return a fitness score for the guess"""


class RelativeFitness(Fitness):
    """A fitness function that can be computed from the chromosome's genes."""

    def __init__(self, chromosome: Chromosome):
        self.chromosome = chromosome

    @abstractmethod
    def __gt__(self, other: RelativeFitness) -> bool:
        """Return True if this fitness is better than the other."""


class Mutation(ABC):
    """Abstract base class for mutation functions."""

    def __init__(
        self, fitness: Optional[Fitness] = None, gene_set: Optional[GeneSet] = None
    ):
        self.fitness = fitness
        self.gene_set = gene_set

    @abstractmethod
    def __call__(self, parent: Chromosome) -> Chromosome:
        """Mutate the parent to create a child"""


class Selection:
    pass


class Crossover:
    pass


class Population:
    pass


class StoppingCriteria(ABC):
    """Abstract base class for stopping criteria functions."""

    @abstractmethod
    def __call__(self, chromosome: Chromosome) -> bool:
        """Return true if can stop the genetic algorithm."""


class ChromosomeGenerator(ABC):
    """Abstract base class for generating chromosomes."""

    gene_set: Any

    @abstractmethod
    def __call__(self) -> Chromosome:
        """Generate a chromosome from the gene set"""


class Runner(ABC):
    """Class for running the genetic algorithm."""

    start_time: float
    chromosome_generator: ChromosomeGenerator
    fitness: Fitness
    stopping_criteria: StoppingCriteria
    mutate: Mutation
    fitness_stagnation_detection: FitnessStagnationDetection

    @abstractmethod
    def display(self, candidate):
        pass

    def run(self, fitness_stagnation_limit: int = 10000) -> Chromosome:
        """Run the genetic algorithm.

        Returns
        -------
        Chromosome
            The best chromosome found
        """
        logging.info("Starting genetic algorithm run")
        random.seed()

        fitness_stagnation_detector = FitnessStagnationDetection(
            self.fitness, fitness_stagnation_limit
        )

        self.start_time = time.time()
        iteration_num = 0

        # generate an initial chromosome
        best_parent = self.chromosome_generator()
        logging.info("Initial chromosome: %s", best_parent)

        # if the initial chromosome is good enough, we're done
        if self.stopping_criteria(best_parent):
            logging.info("Stopping criteria met on initial chromosome")
            return best_parent

        while True:  # repeat until child is as good or better than the parent
            iteration_num += 1
            logging.debug("Starting iteration %s", iteration_num)

            logging.debug("Creating child by mutating parent...")
            child = self.mutate(best_parent)
            logging.debug("Child created: %s", str(child))

            logging.debug("Comparing child and parent fitness...")
            if self.fitness(best_parent) > self.fitness(child):
                logging.debug("Child is not as fit as parent")

                if fitness_stagnation_detector(best_parent):
                    logging.info("Fitness stagnation detected")
                    logging.debug("Iteration complete")
                    logging.debug("= " * 30)
                    return best_parent

                logging.debug("Fitness stagnation not detected")
                logging.debug("Iteration complete")
                logging.debug("= " * 30)
                continue

            logging.debug("Child is as fit or more fit than parent")
            logging.info(
                "Iteration %s - New best chromosome found: %s",
                iteration_num,
                child,
            )
            self.display(child)

            # stop if the child is good enough, otherwise repeat
            logging.debug("Checking stopping criteria...")
            if self.stopping_criteria(child):
                logging.info("Stopping criteria met by child")
                logging.debug("Iteration complete")
                logging.debug("= " * 30)
                return child

            logging.debug("Stopping criteria not met by child")
            best_parent = child

            logging.debug("Iteration complete")
            logging.debug("= " * 30)


class AgeAnnealing:
    def __init__(self, age_limit: int):
        self.age_limit = age_limit
        self.historical_fitnesses: List[float] = []


class AgeAnnealingRunner(ABC):
    """Class for running the genetic algorithm."""

    start_time: float
    chromosome_generator: ChromosomeGenerator
    fitness: Fitness
    stopping_criteria: StoppingCriteria
    mutate: Mutation
    fitness_stagnation_detection: FitnessStagnationDetection
    age_annealing: AgeAnnealing

    @abstractmethod
    def display(self, candidate):
        pass

    def run(self, fitness_stagnation_limit: int = 10000) -> Chromosome:
        """Run the genetic algorithm.

        Returns
        -------
        Chromosome
            The best chromosome found
        """
        logging.info("Starting genetic algorithm run")
        random.seed()

        fitness_stagnation_detector = FitnessStagnationDetection(
            self.fitness, fitness_stagnation_limit
        )

        self.start_time = time.time()
        iteration_num = 0

        # generate an initial chromosome
        best_parent = self.chromosome_generator()
        logging.info("Initial chromosome: %s", best_parent)

        # if the initial chromosome is good enough, we're done
        if self.stopping_criteria(best_parent):
            logging.info("Stopping criteria met on initial chromosome")
            return best_parent

        logging.debug("Adding initial chromosome fitness to historical fitnesses")
        self.age_annealing.historical_fitnesses.append(self.fitness(best_parent))

        logging.debug("Creating parent copy from initial chromosome")
        parent = deepcopy(best_parent)

        while True:  # repeat until child is as good or better than the parent
            iteration_num += 1
            logging.debug("Starting iteration %s", iteration_num)

            logging.debug("Creating child by mutating parent...")
            child = self.mutate(parent)
            logging.debug("Child created: %s", str(child))

            logging.debug("Comparing child and parent fitness...")
            if self.fitness(parent) > self.fitness(child):
                logging.debug("Child is not as fit as parent")

                logging.debug("Checking fitness stagnation...")
                if fitness_stagnation_detector(best_parent):
                    logging.info("Fitness stagnation detected")
                    logging.debug("Iteration complete")
                    logging.debug("= " * 30)
                    return best_parent

                logging.debug("Fitness stagnation not detected")

                logging.debug("Checking age annealing...")
                if not self.age_annealing.age_limit:
                    logging.debug("No age limit set, skipping age annealing")
                    logging.debug("Iteration complete")
                    logging.debug("= " * 30)
                    continue

                logging.debug("Incrementing parent age")
                parent.age += 1
                logging.debug("Parent age: %s", parent.age)

                logging.debug(
                    "Comparing parent age to age annealing limit (%s)...",
                    self.age_annealing.age_limit,
                )
                if parent.age < self.age_annealing.age_limit:
                    logging.debug("Parent age below limit, skipping age annealing")
                    logging.debug("Iteration complete")
                    logging.debug("= " * 30)
                    continue

                logging.debug("Parent age above limit, considering age annealing...")
                index = bisect_left(
                    self.age_annealing.historical_fitnesses,
                    self.fitness(child),
                    0,
                    len(self.age_annealing.historical_fitnesses),
                )

                proportion_similar = index / len(
                    self.age_annealing.historical_fitnesses
                )
                logging.debug(
                    "Proportion of historical fitnesses similar to child: %s",
                    proportion_similar,
                )

                if random.random() < exp(-proportion_similar):
                    logging.debug("Annealing not chosen, skipping age annealing")
                    parent = child
                    logging.debug("Iteration complete")
                    logging.debug("= " * 30)
                    continue

                logging.debug("Annealing chosen, resetting parent to best parent")
                best_parent.age = 0
                parent = best_parent

            if self.fitness(parent) == self.fitness(child):
                logging.debug("Child is as fit as parent")
                child.age = parent.age + 1
                parent = deepcopy(child)
                logging.debug("Iteration complete")
                logging.debug("= " * 30)
                continue

            logging.debug("Child more fit than parent")
            child.age = 0
            parent = child

            logging.debug("Comparing child to best parent...")
            if self.fitness(best_parent) < self.fitness(child):
                logging.debug("Child is more fit than best parent")
                best_parent = child
                self.age_annealing.historical_fitnesses.append(
                    self.fitness(best_parent)
                )

            logging.debug("Best parent is as fit or more fit than child")
            logging.info(
                "Iteration %s - New best chromosome found: %s",
                iteration_num,
                best_parent,
            )
            self.display(child)

            # stop if the child is good enough, otherwise repeat
            logging.debug("Checking stopping criteria...")
            if self.stopping_criteria(child):
                logging.info("Stopping criteria met by child")
                logging.debug("Iteration complete")
                logging.debug("= " * 30)
                return child

            logging.debug("Stopping criteria not met by child")
            best_parent = child

            logging.debug("Iteration complete")
            logging.debug("= " * 30)


class FitnessStagnationDetection(StoppingCriteria):
    """Stop the genetic algorithm if the fitness has not improved in the
    last 10 generations.
    """

    def __init__(self, fitness: Fitness, generations: int = 10):
        self.fitness = fitness
        self.generations = generations
        self.last_fitness = None
        self.last_generation = 0

    def __call__(self, chromosome: Chromosome) -> bool:
        """Return true if can stop the genetic algorithm."""

        if self.last_fitness is None:
            self.last_fitness = self.fitness(chromosome)
            self.last_generation = 0
            logging.debug("FitnessStagnationDetection: setting first fitness")
            return False

        if self.fitness(chromosome) > self.last_fitness:
            self.last_fitness = self.fitness(chromosome)
            self.last_generation = 0
            logging.debug("FitnessStagnationDetection: fitness improved")
            return False

        self.last_generation += 1

        if self.last_generation >= self.generations:
            logging.info(
                "FitnessStagnationDetection: %s generations without improvement",
                self.last_generation,
            )

        return self.last_generation >= self.generations
