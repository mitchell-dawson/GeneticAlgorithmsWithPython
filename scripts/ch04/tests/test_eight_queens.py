from scripts.ch04.eight_queens import eight_queens


def test_eight_queens():
    size = 8
    best = eight_queens(size)
    
    # list of coordinates of the queens of form [row1, col1, row2, col2, ..., rown, coln]
    assert len(best.genes) == 2 * size

    # check each row and column appears only once as a coordinate
    for ii in range(size):
        assert best.genes.count(ii) == 2
