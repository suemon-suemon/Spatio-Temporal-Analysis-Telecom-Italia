import pandas as pd
from cleaning_txt import clean_txt_telecomm

# time_range = clean_txt_telecomm('2013-11-01','2014-01-01')


time_range = pd.period_range('2013-11-01', '2013-12-24', freq='D').astype('string')


def telecomm_mi(cdrtype, cell=None):
    telco = pd.DataFrame({})
    for i in time_range:
        df = pd.read_csv(f'data/{cdrtype}/{cdrtype}-mi-{i}.csv', parse_dates=['datetime'])
        telco = telco.append(df)
    telco = telco.set_index(['CellID', 'datetime'])
    # telco = telco[telco['CellID'] == cell].reset_index(drop=False).drop(['CellID'], axis=1)
    telco.to_csv(f'data/{cdrtype}/{cdrtype}.csv')
    return telco


def provinces_mi(cell=None, start='2013-11-1', end='2013-12-1'):
    df_prov = pd.DataFrame({})
    for i in time_range:
        df = pd.read_csv(f'data/calls_province/mi-to-provinces-{i}.csv', parse_dates=['datetime'])
        df_prov = df_prov.append(df)
    df_prov.set_index(['CellID', 'datetime'], inplace=True, drop=True)
    # df_prov = df_prov[df_prov['CellID'] == cell].drop('CellID', axis=1)
    return df_prov


def precipitation_mi(start='2013-11-01', end='2013-12-24'):
    df_prec = pd.read_csv('MeteoMilano_01-11-13_01-01-14.csv',
                          names=['datetime', 'CellID', 'intensity', 'coverage', 'type'],
                          parse_dates=['datetime'])
    df_prec = df_prec.set_index('datetime', drop=True).sort_values('CellID').sort_index()
    df_prec = df_prec[start:end]

    # you can find the exact quadrants in precip_cell_locate
    df_prec_2 = df_prec[df_prec['CellID'] == 2]
    df_prec_2['CellID'] = 5060
    df_prec_3 = df_prec[df_prec['CellID'] == 3]
    df_prec_3_1 = df_prec_3.copy()
    df_prec_3['CellID'] = 4456
    df_prec_3_1['CellID'] = 4259
    df_prec_33 = pd.concat([df_prec_3, df_prec_3_1])
    df_prec = pd.concat([df_prec_2, df_prec_33]).reset_index(drop=False).set_index(['CellID', 'datetime'],
                                                                                   drop=True).sort_index()

    return df_prec


def tweets():
    df = pd.read_csv('data/tweets/tweets_unclassified.csv', parse_dates=['datetime'])
    seq = r'[2][0][1](21|22|36|43)'
    filterer = df['addresses'].str.contains(seq)
    df = df[filterer]
    df = df.assign(CellID=df['addresses'].str.extract(seq).dropna().replace({'21': 5060,
                                                                             '22': 5060, '36': 4259,
                                                                             '43': 4456})).reset_index(drop=True)
    df['datetime'] = df.datetime.dt.floor('10min')
    df['datetime'] = df.datetime.dt.tz_localize(None)
    d_filter = (df['datetime'] >= '2013-11-01') & (df['datetime'] < '2013-12-24 23:00:00')
    tweet = df[d_filter].reset_index(drop=True)
    tweet = tweet.drop(['coordinates', 'addresses'], axis=1)
    tweet = tweet.groupby(['CellID', 'datetime']).size()
    tweet.name = 'tweets'
    return tweet


def bi_dir_strength_processing():
    strength = pd.DataFrame({})
    for date in time_range:
        df = pd.read_csv(f'data/bidir_strgth/MItoMI-{date}.csv',
                         parse_dates=['datetime'])
        df = df.sort_values(['CellID_1', 'datetime'], axis=0)
        strength = strength.append(df)
    strength = strength.sort_values(['datetime', 'CellID_1']).reset_index(drop=True)
    strength = strength[strength['datetime'] >= '2013-11-01 00:00:00']
    strength = strength.rename({'CellID_1': 'CellID'}, axis=1).drop('CellID_2', axis=1)
    strength = strength.groupby(['CellID', 'datetime']).sum()
    strength.name = 'interaction_strength'
    return strength


if __name__ == '__main__':
    strength = bi_dir_strength_processing()
    twitter = tweets()
    df_prec = precipitation_mi()
    df_prov = provinces_mi()
    df_int = telecomm_mi('internet')

    df_fint = df_int.merge(twitter, how='outer', left_index=True, right_index=True).fillna(0)
    df_final = df_fint.merge(df_prec, how='outer', left_index=True, right_index=True).fillna(method='ffill')
    df_final = df_final.merge(df_prov, how='outer', left_index=True, right_index=True).fillna(0)
    df_final = df_final.merge(strength, how='outer', left_index=True, right_index=True).fillna(0)

    # if you write the csv files with index=False, this next line will not be required (It is used to remove
    # Unnamed columns
    df_final = df_final[['internet', 'tweets', 'intensity', 'coverage', 'type', 'To Milan', 'From Milan',
                         'strength']]
    df_final = df_final.reset_index(drop=False)
    # df_final = df_final[df_final['CellID'] == 5060]
    # df_final = df_final.reset_index(drop=True).drop('CellID', axis=1)

    # df_final.to_csv('internet_5060.csv')

    print(-1)


"""
# cleaning_txt.py:

import pandas as pd


def clean_txt_telecomm(start, end, feature):
    time_range = pd.period_range(start, end, freq='D').astype('string')
    for i in time_range:
        file_df = pd.read_csv(f'raw_txtdata/dataverse_files/sms-call-internet-mi-{i}.txt', header=None, sep='\t',
                              names=['CellID', 'datetime', 'countrycode', 'smsin', 'smsout', 'callin',
                                     'callout', 'internet'])
        # file_df = file_df[(file_df['CellID'] == 4259) | (file_df['CellID'] == 4456) | (file_df['CellID'] == 5060)]
        # file_df = file_df[file_df['countrycode'] == 39]
        if feature == 'all':
            pass
        if feature == 'Calls':
            file_df = file_df.interpolate('akima', limit_area='inside')
            file_df['Calls'] = file_df['callin'] + file_df['callout']
            file_df = file_df.drop(['smsin', 'smsout', 'callin', 'callout', 'countrycode',
                                    'internet'], axis=1)
        elif feature == 'SMS':
            file_df = file_df.interpolate('akima', limit_area='inside')
            file_df['SMS'] = file_df['smsin'] + file_df['smsout']
            file_df = file_df.drop(['smsin', 'smsout', 'callin', 'callout', 'countrycode',
                                    'internet'], axis=1)
        elif feature == 'internet':
            file_df = file_df.drop(['smsin', 'smsout', 'callin', 'callout', 'countrycode'], axis=1)
        file_df['datetime'] = pd.to_datetime(file_df['datetime'], unit='ms')
        file_df.to_csv(f'data/{feature}/{feature}-mi-{i}.csv', index=False)
    return time_range


def clean_txt_provinces(start, end):
    time_range = pd.period_range(start, end, freq='D').astype('string')
    for i in time_range:
        file_df = pd.read_csv(f'dataverse_files_mi_to_prov/mi-to-provinces-{i}.txt', header=None,
                              sep='\t',
                              names=['CellID', 'Name', 'datetime', 'From Milan', 'To Milan'])
        file_df = file_df[(file_df['CellID'] == 4259) | (file_df['CellID'] == 4456) | (file_df['CellID'] == 5060)]
        file_df['datetime'] = pd.to_datetime(file_df['datetime'], unit='ms')
        file_df = file_df.drop('Name', axis=1)
        file_df = file_df.sort_values(['CellID', 'datetime']).reset_index(drop=True)
        file_df = file_df.fillna(0)
        file_df = file_df.groupby(['CellID', 'datetime']).sum()
        file_df.reset_index(drop=False, inplace=True)
        file_df.to_csv(f'New data/inter_prov_calls/interp/mi-to-provinces-{i}.csv', index=False)
    return time_range


def bi_dir_strength(start, end):
    time_range = pd.period_range(start, end, freq='D').astype('string')
    for date in time_range:
        df = pd.read_csv(f'MItoMI-{date}.txt', sep='\t', skiprows=1, names=['datetime', 'CellID_1',
                                                                            'CellID_2', 'strength'])
        df = df[(df['CellID_1'] == 4259) | (df['CellID_1'] == 4456) | (df['CellID_1'] == 5060)]
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        df.to_csv(f'raw_csv_data/MItoMI-{date}.csv', index=False)


if __name__ == '__main__':
    _ = clean_txt_telecomm('2013-11-01', '2014-01-01', 'internet')


"""