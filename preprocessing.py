import pandas as pd

class DataLoader:

    def __init__(path):
        self.path = path

    def load() -> pd.DataFrame:
        path = self.path
        print('Loading from path:', path)
        
        # Load data into dataframe
        kpis = pd.read_csv(path)

        # Format timestamp
        kpis['timestamp'] = pd.to_datetime(kpis['timestamp'])
        kpis.sort_values('timestamp', inplace=True)
        
        # Split data into training and testing sets
        n = len(kpis)
        idx_train = int(n*.7)
        idx_val = idx_train + int(n*.15)
        train = kpis[:idx_train]
        val = kpis[idx_train:idx_val]
        test = kpis[idx_val:]
        
        print(train.shape, val.shape, test.shape)
        return train, val, test


class DataPreProcessor:

    # preprocess preprocesses the dataset into a format most of the anomaly-models wants it to be 
    def process_from_path(path='data/kpis_cleaned_notonehot_median.csv', cluster_data_path='data/cell_clusters.csv'):
        df = pd.read_csv(path)
        drop_columns = ['tech', 'freq', 'site', 'sector', 'Unnamed: 0']
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
        return df_temp.set_index('timestamp')



if __name__ == '__main__':
    df = DataPreProcessor.process_from_path()
    print(df.head())
    df.to_csv('./data/ml_models_dataset_with_zone.csv')
