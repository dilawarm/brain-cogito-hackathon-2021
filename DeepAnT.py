import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange
from DeepAnT_model import DeepAnT

EPOCHS = 10
SAMPLE_STEPS = 10

def make_samples(df):
    n_steps, n_features = df.shape
    nda = df.to_numpy()
    X = np.zeros((n_steps - SAMPLE_STEPS, SAMPLE_STEPS, n_features))
    Y = np.zeros((n_steps - SAMPLE_STEPS, n_features))
    for i in range(SAMPLE_STEPS, n_steps):
        X[i-SAMPLE_STEPS, :, :] = nda[i-SAMPLE_STEPS:i]
        Y[i-SAMPLE_STEPS, :] = nda[i]
    return X, Y


df = pd.read_csv("hackathon_kpis_anonymised.csv")
df = df[df['cell_name'] == '00_11Z']
df = df.fillna(df.median(), axis='index')
df = df.drop(columns='cell_name')
df = df.sort_values('timestamp')
df = df.set_index('timestamp')

X, Y = make_samples(df)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42, shuffle=False)


DEVICE = torch.device('cuda')
model = DeepAnT(SAMPLE_STEPS, len(df.columns)).to(DEVICE)
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-5)

train_data = torch.utils.data.TensorDataset(
    torch.from_numpy(X_train.astype(np.float32)),
    torch.from_numpy(Y_train.astype(np.float32))
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=32, shuffle=True)

try:
    model.load_state_dict(torch.load('models/epoch_8__eval_loss_0.0008937545935623348.pth'))
except Exception as e:
    print(e)
    print('Can\'t load state dict, training model')
    for epoch in trange(EPOCHS):
        model.train()
        loss_sum = 0.0
        steps = 0
        for x, y in tqdm(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            yhat = model(x)
            if torch.any(yhat != yhat):
                print(x)
                print(y)
                print(yhat)
                input()
            loss = criterion(y, yhat)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_sum += loss.item()
            steps += 1
            if steps % 10_000 == 0:
                print('Train loss: {0} @ epoch {1}, step {2}'.format(float(loss_sum/steps), epoch, steps))
        train_loss = loss_sum / steps
        model.eval()
        yhat = model.cpu()(torch.from_numpy(X_test.astype(np.float32)))
        eval_loss = criterion(torch.from_numpy(Y_test.astype(np.float32)), yhat).item()
        print('Epoch {0} done\t train_loss={1} \t eval_loss={2}'.format(epoch, train_loss, eval_loss))
        torch.save(model.state_dict(), 'models/epoch_{0}__eval_loss_{1}.pth'.format(epoch, eval_loss))

def get_anomaly_score(model, data, true_next_value):
    model.eval()
    pred_next_value = model(data.to(DEVICE))
    return torch.norm(pred_next_value - true_next_value.to(DEVICE), dim=1).cpu().detach().numpy()


anom_score = get_anomaly_score(
    model,
    torch.from_numpy(X.astype(np.float32)),
    torch.from_numpy(Y.astype(np.float32))
)
plt.plot(df.index[10:], anom_score)
plt.title('Anomaly scores on FULL dataset')
plt.show()
