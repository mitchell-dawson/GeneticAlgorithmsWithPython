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

import sys

print(sys.path)


import random
import time
from abc import ABC, abstractmethod
from typing import Any, List


class Gene:
    pass

class GeneSet(List[Gene]):
    pass

class Fitness(ABC):
    """Abstract base class for fitness functions."""

    def __init__(self, target: Any):
        self.target = target

    @abstractmethod
    def __call__(self, guess: Any) -> float:
        """Return a fitness score for the guess

        Parameters
        ----------
        guess : Any
            The guess to score

        Returns
        -------
        float
            The fitness score
        """


    
class Generation:
    pass

class Chromosome:
    def __init__(self, genes):
        self.genes = genes

    def __repr__(self) -> str:
        return f"Chromosome({self.genes})"

    def fitness(self, fitness: Fitness):
        return fitness(self)

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

    @abstractmethod
    def __call__(self, chromosome: Chromosome) -> bool:
        """Return true if can stop the genetic algorithm."""

def _generate_parent(length, gene_set):
    genes = []
    while len(genes) < length:
        sampleSize = min(length - len(genes), len(gene_set))
        genes.extend(random.sample(gene_set, sampleSize))
    genes = "".join(genes)
    return Chromosome(genes)

class ChromosomeGenerator(ABC):
    """Abstract base class for chromosome generators."""

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

    def run(self) -> Chromosome:
        """Run the genetic algorithm.

        Returns
        -------
        Chromosome
            The best chromosome found
        """

        random.seed()

        self.start_time = time.time()

        best_parent = self.chromosome_generator()
        self.display(best_parent, self.fitness)

        if self.stopping_criteria(best_parent):
            return best_parent

        while True:

            child = self.mutate(best_parent)

            if self.fitness(best_parent) >= self.fitness(child):
                continue

            self.display(child, self.fitness)

            if self.stopping_criteria(child):
                return child

            best_parent = child

    @abstractmethod
    def display(self, candidate, fitness):
        pass


