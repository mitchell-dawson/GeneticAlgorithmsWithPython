from scripts.ch03.sort_numbers import sort_numbers


def test_sort_numbers():
    length = 10
    best = sort_numbers(length)
    assert best.genes == list(range(length))