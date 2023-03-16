from src.cricket_fixtures.chromosome import FixtureListChromosome
from src.cricket_fixtures.fitness import FixtureListFitness
from src.genetic import StoppingCriteria


class FixtureListStoppingCriteria(StoppingCriteria):
    """Stopping criteria for the magic squares problem."""

    def __init__(self, target: int, fitness: FixtureListFitness):
        self.target = target
        self.fitness = fitness

    def __call__(self, chromosome: FixtureListChromosome) -> bool:
        """Return true if can stop the genetic algorithm."""
        return self.fitness(chromosome) >= self.target
