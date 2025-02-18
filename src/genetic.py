from __future__ import annotations

import copy
import sys

from tqdm import tqdm

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
from typing import Any, List, Optional, Protocol, Tuple, Type, Union


class Gene(Protocol):
    pass


class Chromosome(Protocol):
    """Chromosome for the genetic algorithm."""

    genes: Any
    age: int

    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation of the chromosome."""

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the chromosome."""


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


class AgeAnnealing:
    """Class for annealing the genetic algorithm process based on a chromosomes age.
    Annealing is done by reducing the mutation rate as the chromosome ages.
    """

    def __init__(self, age_limit: float = float("inf")):
        self.age_limit = age_limit
        self.historical_fitnesses: List[float] = []


class Runner(ABC):
    """Class for running the genetic algorithm."""

    start_time: float
    iteration_num: int
    best_parent: Chromosome
    parent: Chromosome
    child: Chromosome

    def __init__(
        self,
        chromosome_generator: ChromosomeGenerator,
        fitness: Fitness,
        stopping_criteria: StoppingCriteria,
        mutate: Mutation,
        age_annealing: AgeAnnealing,
        fitness_stagnation_detector: FitnessStagnationDetector,
    ) -> None:

        self.iteration_num = 0
        self.chromosome_generator = chromosome_generator
        self.fitness = fitness
        self.stopping_criteria = stopping_criteria
        self.mutate = mutate
        self.age_annealing = age_annealing
        self.fitness_stagnation_detector = fitness_stagnation_detector

    @abstractmethod
    def display(self, candidate):
        """Display the candidate in a runner specific way"""

    def run(self) -> Chromosome:
        """Run the genetic algorithm.

        Returns
        -------
        Chromosome
            The best chromosome found
        """
        logging.info("Starting genetic algorithm run")
        self.start_time = time.time()

        self.best_parent = self.chromosome_generator()
        logging.info("Initial chromosome: %s", self.best_parent)

        if self.stopping_criteria(self.best_parent):
            raise StoppingCriteriaMet(self.best_parent, self.iteration_num)

        logging.debug("Adding initial chromosome fitness to historical fitnesses")
        self.age_annealing.historical_fitnesses.append(self.fitness(self.best_parent))

        logging.debug("Creating parent copy from initial chromosome")
        self.parent = deepcopy(self.best_parent)

        def generator():
            while True:  # repeat until child is as good or better than the parent
                yield

        for _ in tqdm(generator()):
            self.iteration_num += 1
            logging.debug("Starting iteration %s", self.iteration_num)

            try:
                self.main_loop()
            except FitnessStagnationDetected as exception:
                return exception.chromosome
            except StoppingCriteriaMet as exception:
                return exception.chromosome
            except ContinueLoop:
                continue

    def main_loop(self):
        """Main loop for the genetic algorithm."""
        self.create_child()

        (
            parent_fitness,
            child_fitness,
        ) = self.calculate_parent_child_fitnesses()
        self.compare_parent_and_child(parent_fitness, child_fitness)

        best_parent_fitness = self.fitness(self.best_parent)
        self.compare_best_parent_and_child(best_parent_fitness, child_fitness)
        self.check_stopping_criteria()

        self.best_parent = self.child
        logging.debug("Iteration complete")
        logging.debug("= " * 30)

    def calculate_parent_child_fitnesses(self) -> Tuple[float, float, float]:
        return map(self.fitness, (self.parent, self.child))

    def compare_best_parent_and_child(self, best_parent_fitness, child_fitness):
        """Compare the best parent and child fitnesses."""

        logging.debug("Comparing child to best parent...")
        if best_parent_fitness < child_fitness:
            logging.debug("Child is more fit than best parent")
            self.best_parent = self.child
            self.age_annealing.historical_fitnesses.append(best_parent_fitness)

            logging.info(
                "Iteration %s - New best chromosome found: %s",
                self.iteration_num,
                self.best_parent,
            )

            self.display(self.child)
        else:
            logging.debug("Best parent is as fit or more fit than child")

    def compare_parent_and_child(self, parent_fitness, child_fitness):
        """Compare the parent and child fitnesses."""

        logging.debug("Comparing child and parent fitness...")
        if parent_fitness > child_fitness:
            logging.debug("Child is not as fit as parent")
            self.detect_fitness_stagnation()
            self.run_age_annealing(child_fitness)

        if parent_fitness == child_fitness:
            self.child.age = self.parent.age + 1
            self.parent = deepcopy(self.child)
            raise ContinueLoop("Child is as fit as parent")

        logging.debug("Child is more fit than parent")
        self.child.age = 0
        self.parent = deepcopy(self.child)

    def check_stopping_criteria(self):
        """Check if the stopping criteria has been met."""

        logging.debug("Checking stopping criteria...")
        if self.stopping_criteria(self.child):
            raise StoppingCriteriaMet(self.child, self.iteration_num)
        logging.debug("Stopping criteria not met by child")

    def create_child(self):
        """Create a child by mutating the parent."""

        logging.debug("Creating child by mutating parent...")
        self.child = self.mutate(self.parent)
        logging.debug("Child created: %s", str(self.child))

    def detect_fitness_stagnation(self):
        """Detect fitness stagnation."""

        logging.debug("Checking fitness stagnation...")
        if self.fitness_stagnation_detector(self.best_parent):
            raise FitnessStagnationDetected(self.best_parent, self.iteration_num)
        logging.debug("Fitness stagnation not detected")

    def run_age_annealing(self, child_fitness):
        """Run age annealing."""

        logging.debug("Checking age annealing...")
        if not self.age_annealing.age_limit:
            raise ContinueLoop("No age limit set, skipping age annealing")

        logging.debug("Incrementing parent age")
        self.parent.age += 1
        logging.debug("Parent age: %s", self.parent.age)

        logging.debug(
            "Comparing parent age to age annealing limit (%s)...",
            self.age_annealing.age_limit,
        )
        if self.parent.age < self.age_annealing.age_limit:
            raise ContinueLoop("Parent age below limit, skipping age annealing")

        logging.debug("Parent age above limit, considering age annealing...")
        index = bisect_left(
            self.age_annealing.historical_fitnesses,
            child_fitness,
            0,
            len(self.age_annealing.historical_fitnesses),
        )

        proportion_similar = index / len(self.age_annealing.historical_fitnesses)
        logging.debug(
            "Proportion of historical fitnesses similar to child: %s",
            proportion_similar,
        )

        if random.random() < exp(-proportion_similar):
            self.parent = self.child
            raise ContinueLoop("Annealing not chosen, skipping age annealing")

        logging.debug("Annealing chosen, resetting parent to best parent")
        self.best_parent.age = 0
        self.parent = self.best_parent


class StoppingCriteriaMet(Exception):
    """Exception raised when stopping criteria is met."""

    def __init__(self, chromosome: Chromosome, iteration_num: int):
        self.chromosome = chromosome
        self.iteration_num = iteration_num
        logging.info(
            "Stopping criteria met by chromosome %s on iteration %d",
            chromosome,
            iteration_num,
        )
        logging.debug("Iteration complete")
        logging.debug("= " * 30)


class FitnessStagnationDetected(Exception):
    """Exception raised when fitness stagnation is detected."""

    def __init__(self, chromosome: Chromosome, iteration_num: int):
        self.chromosome = chromosome
        self.iteration_num = iteration_num
        logging.info("Fitness stagnation detected")
        logging.info("Best chromosome found: %s", chromosome)
        logging.debug("Iteration complete")
        logging.debug("= " * 30)


class ContinueLoop(Exception):
    """Exception raised to continue the loop."""

    def __init__(self, message):
        logging.debug(message)
        logging.debug("Iteration complete")
        logging.debug("= " * 30)


class FitnessStagnationDetector(StoppingCriteria):
    """Stop the genetic algorithm if the fitness has not improved in the
    last 10 generations.
    """

    def __init__(
        self, fitness: Fitness, generations_limit: Union[float, int] = float("inf")
    ):
        self.fitness = fitness
        self.generations_limit = generations_limit
        self.last_fitness = None
        self.last_generation = 0

    def __call__(self, chromosome) -> bool:
        """Return true if can stop the genetic algorithm."""

        chromosome_fitness = self.fitness(chromosome)

        if self.last_fitness is None:
            self.last_fitness = chromosome_fitness
            self.last_generation = 0
            logging.debug("FitnessStagnationDetection: setting first fitness")
            return False

        if chromosome_fitness > self.last_fitness:
            self.last_fitness = chromosome_fitness
            self.last_generation = 0
            logging.debug("FitnessStagnationDetection: fitness improved")
            return False

        self.last_generation += 1

        if self.last_generation >= self.generations_limit:
            logging.info(
                "FitnessStagnationDetection: %s generations without improvement",
                self.last_generation,
            )

        return self.last_generation >= self.generations_limit
