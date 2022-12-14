import logging
import random

from scripts.ch01.guess_password import guess_password


def test_guess_password():

    logging_format = (
        "[%(levelname)8s :%(filename)20s:%(lineno)4s - %(funcName)10s] %(message)s"
    )
    logging.basicConfig(format=logging_format, level=logging.INFO)

    random.seed(3)
    target = "Hello World!"
    gene_set = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!."
    best = guess_password(target, gene_set)

    assert best.genes == target


def test_guess_password_with_stagnation_limit():

    logging_format = (
        "[%(levelname)8s :%(filename)20s:%(lineno)4s - %(funcName)10s] %(message)s"
    )
    logging.basicConfig(format=logging_format, level=logging.INFO)

    random.seed(3)
    target = "Hello World!"
    gene_set = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!."
    best = guess_password(target, gene_set, fitness_stagnation_limit=10000)

    assert best.genes == target
