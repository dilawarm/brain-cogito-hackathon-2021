import pandas as pd


def load_data(path='hackathon_kpis_anonymised.csv'):
    print('Load', path)
    kpis = pd.read_csv(path)
    kpis['timestamp'] = pd.to_datetime(kpis['timestamp'])
    kpis.sort_values('timestamp', inplace=True)
    n = len(kpis)
    idx_train = int(n*.7)
    idx_val = idx_train + int(n*.15)
    train = kpis[:idx_train]
    val = kpis[idx_train:idx_val]
    test = kpis[idx_val:]
    print(train.shape, val.shape, test.shape)
    return train, val, test


if __name__ == '__main__':
    load_data()