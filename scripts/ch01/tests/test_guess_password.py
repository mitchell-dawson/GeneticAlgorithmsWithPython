from scripts.ch01.guess_password import guess_password


def test_guess_password():
    target = "Hello World!"
    gene_set = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!."
    best = guess_password(target, gene_set)

    assert best.genes == target
