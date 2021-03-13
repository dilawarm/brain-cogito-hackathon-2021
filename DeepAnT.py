import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm.auto import trange, tqdm

from DeepAnT_model import DeepAnT, DeepAnT2d

EPOCHS = 10
SAMPLE_STEPS = 10


def make_samples(df):
    n_steps, n_features = df.shape
    nda = df.to_numpy()
    X = np.zeros((n_steps - SAMPLE_STEPS, SAMPLE_STEPS, n_features))
    Y = np.zeros((n_steps - SAMPLE_STEPS, n_features))
    for i in range(SAMPLE_STEPS, n_steps):
        X[i - SAMPLE_STEPS, :, :] = nda[i - SAMPLE_STEPS:i]
        Y[i - SAMPLE_STEPS, :] = nda[i]
    return X, Y


df = pd.read_csv("hackathon_kpis_anonymised/ml_models_dataset_with_zone.csv")


# df['timestamp'] = pd.to_datetime(df['timestamp'])
# df.drop(columns='timestamp', inplace=True)


def get_anomaly_score(model, data, true_next_value):
    model.eval()
    pred_next_value = model(data)
    return torch.norm(pred_next_value - true_next_value.to(model.device), dim=1).cpu().detach().numpy()


def model_for_dataframe(name, df):
    X, Y = make_samples(df.drop(columns=['timestamp', 'cell_name']))
    X, Y = map(lambda x: torch.tensor(x, dtype=torch.float32), (X, Y))

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.25, random_state=42, shuffle=False)

    model = DeepAnT2d(SAMPLE_STEPS, X_train.shape[-1]).to('cuda')
    for epoch in trange(EPOCHS):
        train_loss = model.train_epoch(X_train, Y_train, verbose=name == 'all')
        eval_loss = model.evaluate(X_test, Y_test)
        if name == 'all':
            print('Epoch {0} done\t train_loss={1} \t eval_loss={2}'.format(epoch, train_loss, eval_loss))
        torch.save(model.state_dict(), 'models/deepant_{0}_epoch_{1}__eval_loss_{2}.pth'.format(name, epoch, eval_loss))

    anom_score = get_anomaly_score(model, X, Y)
    plt.plot(df.index[10:], anom_score)
    plt.title('Anomaly scores from ' + name)
    plt.savefig('figs/deepant_' + name + '.png')
    plt.close()
    return model


for name, idx in tqdm(df.groupby('cell_name').groups.items()):
    model_for_dataframe(name, df.loc[idx])
model_for_dataframe('all', df)
