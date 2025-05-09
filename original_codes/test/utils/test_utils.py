from utils.manipulate_data import _gen_cellids_by_colrow
from utils.milano_grid import map_back


def test_gen_cellids_by_colrow():
    col1 = 40
    col2 = 70
    row1 = 40
    row2 = 70
    cellids = _gen_cellids_by_colrow(col1, col2, row1, row2)
    assert len(cellids) == (col2 - col1 + 1) * (row2 - row1 + 1)
    assert cellids[0] == map_back(row1-1, col1-1)
    assert cellids[-1] == map_back(row2-1, col2-1)

    cellids = _gen_cellids_by_colrow(1, 100, 1 ,100)
    assert len(cellids) == 100 * 100
    assert len(set(cellids)) == 100 * 100

def test_map_back():
    row, col = 40, 40
    assert map_back(row-1, col-1) == 6040
    row, col = 100, 1
    assert map_back(row-1, col-1) == 1
    row, col = 1, 100
    assert map_back(row-1, col-1) == 10000
