import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm.auto import trange

from DeepAnT_model import DeepAnT

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


df = pd.read_csv("hackathon_kpis_anonymised/ml_models_dataset.csv")
# df['timestamp'] = pd.to_datetime(df['timestamp'])
df.drop(columns='timestamp', inplace=True)

X, Y = make_samples(df)

X, Y = map(lambda x: torch.tensor(x, dtype=torch.float32), (X, Y))

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42, shuffle=False)

model = DeepAnT(SAMPLE_STEPS, len(df.columns)).to('cuda')

try:
    model.load_state_dict(torch.load('models/epoch_8__eval_loss_0.0008937545935623348.pth'))
except Exception as e:
    print(e)
    print('Can\'t load state dict, training model')
    for epoch in trange(EPOCHS):
        train_loss = model.train_epoch(X_train, Y_train)
        eval_loss = model.evaluate(X_test, Y_test)
        print('Epoch {0} done\t train_loss={1} \t eval_loss={2}'.format(epoch, train_loss, eval_loss))
        torch.save(model.state_dict(), 'models/epoch_{0}__eval_loss_{1}.pth'.format(epoch, eval_loss))


def get_anomaly_score(model, data, true_next_value):
    model.eval()
    pred_next_value = model(data)
    return torch.norm(pred_next_value - true_next_value.to(model.device), dim=1).cpu().detach().numpy()


anom_score = get_anomaly_score(
    model,
    torch.from_numpy(X.astype(np.float32)),
    torch.from_numpy(Y.astype(np.float32))
)
plt.plot(df.index[10:], anom_score)
plt.title('Anomaly scores on FULL dataset')
plt.show()
