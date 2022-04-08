size = (100, 100)

def _map_grid(cell_id):
    # returns row_nr, column_nr in a 2D grid
    num_rows, num_columns = size
    cell_indx = cell_id - 1
    row_nr = int(num_rows - 1 - cell_indx // num_rows)
    column_nr = int(cell_indx % num_rows)
    
    return row_nr, column_nr

def _map_back(row_nr, column_nr):
    # returns cell_id in a 2D grid, nr starts at 0
    num_rows, num_columns = size
    cell_id = int(num_columns * (num_rows - 1 - row_nr) + column_nr + 1)
    
    return cell_id


def gen_cellids_by_colrow(grid_range) -> list:
    """
    Generate cellid list by col and row
    :param grid_range: the range of grids, (row_min, row_max, col_min, col_max)
    """
    row1, row2, col1, col2 = grid_range
    cellids = []
    for col in range(col1, col2 + 1):
        for row in range(row1, row2 + 1):
            cellids.append(_map_back(row-1, col-1))
    return cellids