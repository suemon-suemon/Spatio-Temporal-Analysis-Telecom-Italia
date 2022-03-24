size = (100, 100)

def map_grid(cell_id):
    # returns row_nr, column_nr in a 2D grid
    num_rows, num_columns = size
    cell_indx = cell_id - 1
    row_nr = int(num_rows - 1 - cell_indx // num_rows)
    column_nr = int(cell_indx % num_rows)
    
    return row_nr, column_nr

def map_back(row_nr, column_nr):
    # returns cell_id in a 2D grid, nr starts at 0
    num_rows, num_columns = size
    cell_id = int(num_columns * (num_rows - 1 - row_nr) + column_nr + 1)
    
    return cell_id