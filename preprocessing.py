import pandas as pd
import numpy as np
from tqdm import tqdm

class DataLoader:

    def __init__(self, path):
        self.path = path

    def load(self) -> pd.DataFrame:
        path = self.path
        print('Loading from path:', path)
        
        # Load data into dataframe
        kpis = pd.read_csv(path)

        # Format timestamp
        kpis['timestamp'] = pd.to_datetime(kpis['timestamp'])
        kpis.sort_values('timestamp', inplace=True)
        
        # Split data into training and testing sets
        n = len(kpis)
        #idx_train = int(n*.7)
        idx_train = int(n)
        idx_val = idx_train
        #idx_val += int(n*.15)
        train = kpis[:idx_train]
        val = kpis[idx_train:idx_val]
        test = kpis[idx_val:]
        
        print(train.shape, val.shape, test.shape)
        return train, val, test


# builder pattern of datapreprocessing
class DataPreProcessor:

    def __init__(self):
        self.path = None
        self.df = None

    #reads kpi path and sets self.df as the file. 
    def read_file(self, read_kpis_path='./data/hackathon_kpis_anonymised.csv'):
        self.df = pd.read_csv(read_kpis_path)
        return self

    def fetch_data(self):
        return self.df

    # preprocess preprocesses the dataset into a format most of the anomaly-models wants it to be 
    def process_from_path(self, cluster_data_path='data/cell_clusters.csv'):
        if self.df is None:
            print("read the csv file")
            return self

        df = self.df
        drop_columns = ['tech', 'freq', 'site', 'sector']
        df.drop(columns=drop_columns, inplace=True)

        # Onehot-encode
        df = pd.get_dummies(df, columns=['tech_freq'])*1

        # Fix nans
        df_temp = df.fillna(df.median(), axis='index')
        
        # Add cluster-info onehot-encoded
        if cluster_data_path:
            df_clusters = pd.read_csv(cluster_data_path)
            dict_clusters = df_clusters.set_index('cell_name').to_dict('index')
            df_temp['zone'] = df_temp['cell_name'].apply(lambda cell_name: dict_clusters[cell_name]['cluster'] if cell_name in dict_clusters else -1)
            df_temp = pd.get_dummies(df_temp, columns=['zone'])*1
        
        # Sort by timestamp
        df_temp = df_temp.sort_values('timestamp')
        self.df = df_temp.set_index('timestamp')
        return self

    # splitting up the cell_name column to multiple columns.
    def extract_cell_name_data(self, onehot_site=False):
        if self.df is None:
            print("read the csv file")
            return self

        d = self.df

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

        if onehot_site: # onehot encoding of site. 
            d['site'] = d['cell_name'].apply(lambda cell_name: cell_name[0:2])
            y = pd.get_dummies(d.site, prefix='site_')
            del d['site'] # delete the site column 
            d = pd.concat([d, y], axis=1)

        d['sector'] = d['cell_name'].apply(lambda cell_name: cell_name[3])
        # d['carrier'] = d['cell_name'].apply(lambda cell_name: int(cell_name[4]))
        d['tech_freq'] = d['cell_name'].apply(lambda cell_name: cell_name[5])
        d['tech'] = d['tech_freq'].apply(lambda t: tech[t] )
        d['freq'] = d['tech_freq'].apply(lambda t: freq[t] )
        self.df = d

        return self

    # Fixing NaN values that is created by for example dividing by 0.
    def fix_failure_rates(self, median=True):
        if self.df is None:
            print("read the csv file")
            return self
        
        # set local data
        data = self.df

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
            else:
                data[col] = data[col].fillna(0)

        # Fill the rest of nans with zeros
        self.df = data.fillna(0)
        return self
