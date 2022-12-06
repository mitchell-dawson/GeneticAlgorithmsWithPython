import numpy as np

from scripts.ch06.card_problem import card_problem


def test_card_problem():

    gene_set = list(range(1, 11))

    best = card_problem(gene_set)

    assert np.sum(best.left_group) == 36
    assert np.prod(best.right_group) == 360
