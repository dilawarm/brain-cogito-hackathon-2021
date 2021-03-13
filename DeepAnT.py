import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from utils import normalize
from tqdm.auto import trange, tqdm

from DeepAnT_model import DeepAnT2d

EPOCHS = 2
SAMPLE_STEPS = 10
DEVICE = torch.device('cuda')


def make_samples(df):
    n_steps, n_features = df.shape
    nda = df.to_numpy()
    X = np.zeros((n_steps - SAMPLE_STEPS, SAMPLE_STEPS, n_features))
    Y = np.zeros((n_steps - SAMPLE_STEPS, n_features))
    for i in range(SAMPLE_STEPS, n_steps):
        X[i - SAMPLE_STEPS, :, :] = nda[i - SAMPLE_STEPS:i]
        Y[i - SAMPLE_STEPS, :] = nda[i]
    return X, Y


df = pd.read_csv("hackathon_kpis_anonymised/ml_models_dataset_with_zone_v2.csv")


# df['timestamp'] = pd.to_datetime(df['timestamp'])
# df.drop(columns='timestamp', inplace=True)


def get_anomaly_score(model, data, true_next_value):
    model.eval()
    pred_next_value = model(data)
    return (pred_next_value - true_next_value.to(model.device)).abs().cpu().detach().numpy()


def model_for_dataframe(name, df):
    df_model = df.drop(columns=['timestamp', 'cell_name'])
    X, Y = make_samples(df_model)
    X, Y = map(lambda x: torch.tensor(x, dtype=torch.float32), (X, Y))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42, shuffle=False)

    n_features = X_train.shape[-1]
    model = DeepAnT2d(SAMPLE_STEPS, n_features).to(DEVICE)
    try:
        paths = [
            os.path.join('models', path)
            for path in os.listdir('models')
            if name in path and 'deepant' in path
        ]
        model.load_state_dict(torch.load(max(paths)))
    except Exception as e:
        print(e)
        print('Cannot load model, training it instead')
        for epoch in trange(EPOCHS):
            train_loss = model.train_epoch(X_train, Y_train, verbose=name == 'all')
            if name == 'all':
                model = model.cpu()
            eval_loss = model.evaluate(X_test, Y_test)
            if name == 'all':
                model = model.to(DEVICE)
                print('Epoch {0} done\t train_loss={1} \t eval_loss={2}'.format(epoch, train_loss, eval_loss))
            torch.save(model.state_dict(),
                       'models/deepant_{0}_epoch_{1}__eval_loss_{2}.pth'.format(name, epoch, eval_loss))

    if name == 'all':
        model = model.cpu()
    anom_score = get_anomaly_score(model, X, Y)
    plt.plot(df.index[10:], np.linalg.norm(anom_score, axis=1))
    plt.title('Anomaly scores from ' + name)
    plt.savefig('figs/deepant_' + name + '.png')
    plt.close()

    anom_score = normalize(anom_score)

    anom_score = np.concatenate((np.zeros((SAMPLE_STEPS, n_features)), anom_score))

    anom_df = pd.DataFrame(anom_score, index=df.index, columns=df_model.columns)
    anom_df['timestamp'] = df['timestamp']
    anom_df['cell_name'] = df['cell_name']
    anom_df.to_pickle('preds/deepant_' + name + '.pkl')

    return model


for name, idx in tqdm(df.groupby('cell_name').groups.items()):
    model_for_dataframe(name, df.loc[idx])
model_for_dataframe('all', df)
