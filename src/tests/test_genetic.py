import random

from src.genetic import AbsoluteFitness, Chromosome, Mutation, RelativeFitness


class TestChromosome(Chromosome):
    """A simple test chromosome."""

    def __init__(self, genes: str, age: int):
        self.genes = genes
        self.age = age

    def __str__(self) -> str:
        return f"Chromosome({self.genes}, age={self.age})"

    def __repr__(self) -> str:
        return str(self)


class TestAbsoluteFitness(AbsoluteFitness):
    def __init__(self, value: float):
        self.value = value

    def __call__(self, chromosome: TestChromosome) -> float:
        return len(chromosome.genes) + (chromosome.age * self.value)


class TestRelativeFitness(RelativeFitness):
    """A simple test RelativeFitness."""

    def __gt__(self, other: RelativeFitness) -> bool:
        """A simple relative fitness. In this case a shorter chromosome is better, and
        a if chromosomes are of equal length, the older one is better

        Parameters
        ----------
        other : RelativeFitness
            TestRelativeFitness

        Returns
        -------
        bool
            True if self is better than other, False otherwise
        """

        if len(self.chromosome.genes) == len(other.chromosome.genes):
            return self.chromosome.age > other.chromosome.age
        else:
            return len(self.chromosome.genes) > len(other.chromosome.genes)


class TestMutation(Mutation):
    """A simple test Mutation."""

    def __init__(self, fitness: AbsoluteFitness, gene_set: str):
        self.fitness = fitness
        self.gene_set = gene_set

    def __call__(self, parent: TestChromosome) -> TestChromosome:
        index = 0
        new_genes = parent.genes.replace(parent.genes[index], self.gene_set[index])
        return TestChromosome(genes=new_genes, age=parent.age)


def test_simple_Chromosome():
    """GIVEN a simple Chromosome
    WHEN a new instance is created
    THEN the attributes are set correctly
    """

    chromosome = TestChromosome(genes="abc", age=2)

    assert chromosome.genes == "abc"
    assert chromosome.age == 2

    assert str(chromosome) == "Chromosome(abc, age=2)"


def test_simple_AbsoluteFitness():
    """GIVEN a simple AbsoluteFitness
    WHEN a chromosome is evaluated
    THEN the correct fitness value is returned
    """

    value = 30.0

    # len(genes) = 3, age = 2
    chromosome = TestChromosome(genes="abc", age=2)

    fitness = TestAbsoluteFitness(value=value)

    # 3 + (2*30) = 63
    assert fitness(chromosome) == 63.0


def test_simple_RelativeFitness():
    """GIVEN a simple RelativeFitness
    WHEN a chromosomes are compared
    THEN the order of chromosome fitness is correct
    """

    chromosome_1 = TestChromosome(genes="abc", age=2)
    chromosome_2 = TestChromosome(genes="abc", age=3)
    chromosome_3 = TestChromosome(genes="abcd", age=2)
    chromosome_4 = TestChromosome(genes="abcd", age=4)

    # shorter chromosome is better here
    assert TestRelativeFitness(chromosome_1) < TestRelativeFitness(chromosome_3)
    assert TestRelativeFitness(chromosome_2) < TestRelativeFitness(chromosome_3)
    assert TestRelativeFitness(chromosome_1) < TestRelativeFitness(chromosome_4)
    assert TestRelativeFitness(chromosome_2) < TestRelativeFitness(chromosome_4)

    # If chromosomes are of equal length, the older one is better
    assert TestRelativeFitness(chromosome_1) < TestRelativeFitness(chromosome_2)
    assert TestRelativeFitness(chromosome_3) < TestRelativeFitness(chromosome_4)


def test_call_Mutation():
    """GIVEN a simple Mutation
    WHEN a chromosome is mutated
    THEN the chromosome is mutated correctly
    """

    value = 30.0
    gene_set = "dcba"
    chromosome = TestChromosome(genes="abc", age=2)
    fitness = TestAbsoluteFitness(value=value)
    mutation = TestMutation(fitness, gene_set)

    child = mutation(chromosome)

    assert child.genes == "dbc"
    assert child.age == 2
