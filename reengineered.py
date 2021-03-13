import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

df = pd.read_csv('./data/hackathon_kpis_anonymised.csv')

print(df.head())


# Voice attempts
# Data attempts



def extract_cell_name_data(data, onehot_site=False):
    d = data.copy()

    tech = {
        'Z': '4G',
        'X': '4G',
        'Y': '2G',
        'W': '4G',
        'V': '3G',
        'R': '4G',
        'Q': '3G',
        'P': '2G',
    }

    freq = {
        'Z': 2100,
        'X': 800,
        'Y': 900,
        'W': 2600,
        'V': 900,
        'R': 1800,
        'Q': 2100,
        'P': 1800,
    }
    
    d['site'] = d['cell_name'].apply(lambda cell_name: cell_name[0:2])
    # Split cell names into different categories
    if onehot_site:
        d['site'] = d['cell_name'].apply(lambda cell_name: cell_name[0:2])
        y = pd.get_dummies(d.site, prefix='site_')
        del d['site'] # delete the site column 
        d = pd.concat([d, y], axis=1)

    d['sector'] = d['cell_name'].apply(lambda cell_name: cell_name[3])
    # d['carrier'] = d['cell_name'].apply(lambda cell_name: int(cell_name[4]))
    d['tech_freq'] = d['cell_name'].apply(lambda cell_name: cell_name[5])
    d['tech'] = d['tech_freq'].apply(lambda t: tech[t] )
    d['freq'] = d['tech_freq'].apply(lambda t: freq[t] )

    return d

def fix_failure_rates(d: pd.DataFrame, median=True):
    data = d.copy()

    # Add median or null
    nan_columns = [
        'voice_setup_failure_rate',
        'voice_tot_failure_rate',
        'data_setup_failure_rate',
        'data_tot_failure_rate',
    ]

    cell_names = data['cell_name'].unique()

    for col in tqdm(nan_columns):
        data[f'{col}_is_nan'] = data[col].isna() 
        if median:
            for cell in cell_names:
                # Replace nan with median for given cell
                cell_index = data.index[data['cell_name'] == cell].tolist()
                # print(cell_index)
                data_cell = data[data['cell_name'] == cell]

                if data_cell[col].empty:
                    data.loc[cell_index] = 0
                else:
                    numpy_value_col = data_cell[col].to_numpy()
                    mean_value = np.nanmean(numpy_value_col)
                    data.loc[cell_index] = data.loc[cell_index].fillna(mean_value)
                    #data.loc[cell_index] = data.loc[cell_index].fillna(data_cell[col].median())

        else:
            data[col] = data[col].fillna(0)

    # Fill the rest of nans with zeros
    data = data.fillna(0)

    return data




df = fix_failure_rates(df, median=True)

df = extract_cell_name_data(df, True)

df.to_csv('kpis_cleaned_onehot_median.csv')

 