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


import random
import time
from abc import ABC, abstractmethod
from typing import Any, List


class Gene:
    pass

class GeneSet(List[Gene]):
    pass


class Generation:
    pass

class Chromosome:
    def __init__(self, genes):
        self.genes = genes

    def __repr__(self) -> str:
        return f"Chromosome({self.genes})"

class Fitness(ABC):
    """Fitness is the abstract base class for fitness functions.
    """

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

    def __init__(self, fitness: Fitness, gene_set: GeneSet):
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
    
    @abstractmethod
    def display(self, candidate):
        pass

    def run(self) -> Chromosome:
        """Run the genetic algorithm.

        Returns
        -------
        Chromosome
            The best chromosome found
        """

        random.seed()

        self.start_time = time.time()

        # generate an initial chromosome
        best_parent = self.chromosome_generator()
        self.display(best_parent)

        # if the initial chromosome is good enough, we're done
        if self.stopping_criteria(best_parent):
            return best_parent

        while True:

            # create child by mutating the parent
            child = self.mutate(best_parent)

            # repeat until child is better than the parent
            if self.fitness(best_parent) > self.fitness(child):
                continue
            else:
                self.display(child)

            # stop if the child is good enough, otherwise repeat
            if self.stopping_criteria(child):
                return child
            else:
                best_parent = child


