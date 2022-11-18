from scripts.ch02.one_max import one_max


def test_one_max():
    target = 100
    gene_set = [0, 1]
    best = one_max(target, gene_set)

    assert len(best.genes) == target
    assert best.genes.count("1") == target
    assert best.genes.count("0") == 0