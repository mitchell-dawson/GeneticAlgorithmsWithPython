import os
import random
import sys

print(sys.path)
print(os.getcwd())

from src.genetic import (Chromosome, ChromosomeGenerator, Fitness, GeneSet,
                         Mutation, Population, Runner, StoppingCriteria,
                         get_best)


class GuessPasswordFitness(Fitness):

    def __call__(self, chromosome: Chromosome) -> float:
        """Return a fitness score for the guess. The higher the score, the better the 
        guess. The score is the sum of the number of characters in the guess that
        match the corresponding character in the target.

        Parameters
        ----------
        chromosome : Chromosome
            The guess to score

        Returns
        -------
        float
            The fitness score
        """        

        return sum(
            1 for expected, actual in zip(self.target, chromosome.genes) if expected == actual
        )


class GuessPasswordMutation(Mutation):

    def __init__(self, fitness: Fitness, gene_set: GeneSet):
        super().__init__(fitness, gene_set)

    def __call__(self, parent):
        index = random.randrange(0, len(parent.genes))
        child_genes = list(parent.genes)
        new_gene, alternate = random.sample(self.gene_set, 2)
        child_genes[index] = alternate if new_gene == child_genes[index] else new_gene
        genes = "".join(child_genes)
        return Chromosome(genes)


class GuessPasswordStoppingCriteria(StoppingCriteria):
    def __init__(self, target):
        self.target = target

    @property
    def optimal_fitness(self) -> float:
        return len(self.target)

    def __call__(self, population: Population) -> bool:
        return population.best.fitness >= len(self.target)

class GuessPasswordChromosomeGenerator(ChromosomeGenerator):

    def __init__(self, gene_set, length) -> None:
        self.gene_set = gene_set
        self.length = length

    def __call__(self) -> Chromosome:
        genes = []
        while len(genes) < self.length:
            sampleSize = min(self.length - len(genes), len(self.gene_set))
            genes.extend(random.sample(self.gene_set, sampleSize))
        genes = "".join(genes)
        return Chromosome(genes)

def guess_password(target, gene_set):

    runner = Runner()
    runner.run()

    fitness = GuessPasswordFitness(target)

    chromosome_generator = GuessPasswordChromosomeGenerator(gene_set, len(target))

    best = get_best(
        chromosome_generator,
        fitness,
        GuessPasswordStoppingCriteria(target),
        runner.display,
        GuessPasswordMutation(fitness, gene_set),
    )
    assert best.genes == target


def main():

    target = "Hello World!"
    gene_set = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!."

    guess_password(target, gene_set)


if __name__ == "__main__":
    main()
