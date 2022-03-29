from utils.load_data import load_part_grid_data

def test_load_data_hour_period():
    data, df = load_part_grid_data('data/sms-call-internet-mi', aggr_time='hour', 
                                col1=41, col2=70, row1=41, row2=70)
    assert data.shape == (720, 900) # 30 * 24, 30 * 30
    assert df.values.shape == (720 * 900, 3)


# if __name__ == '__main__':
#     test_load_data_hour_period()