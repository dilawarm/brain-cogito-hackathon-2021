import torch
from tqdm.auto import tqdm


class DeepAnT(torch.nn.Module):
    """
        Model : Class for DeepAnT model
    """

    def __init__(self, time_steps, n_features):
        super(DeepAnT, self).__init__()
        self.conv1d_1_layer = torch.nn.Conv1d(
            in_channels=time_steps, out_channels=16, kernel_size=3)
        self.relu_1_layer = torch.nn.ReLU()
        self.maxpooling_1_layer = torch.nn.MaxPool1d(kernel_size=2)
        self.conv1d_2_layer = torch.nn.Conv1d(
            in_channels=16, out_channels=16, kernel_size=3)
        self.relu_2_layer = torch.nn.ReLU()
        self.maxpooling_2_layer = torch.nn.MaxPool1d(kernel_size=2)
        self.flatten_layer = torch.nn.Flatten()
        self.dense_1_layer = torch.nn.Linear(80, n_features * 2)
        self.relu_3_layer = torch.nn.ReLU()
        self.dropout_layer = torch.nn.Dropout(p=0.25)
        self.dense_2_layer = torch.nn.Linear(n_features * 2, n_features)

        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-5)

    @property
    def device(self):
        return self.conv1d_1_layer.bias.get_device()

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv1d_1_layer(x)
        x = self.relu_1_layer(x)
        x = self.maxpooling_1_layer(x)
        x = self.conv1d_2_layer(x)
        x = self.relu_2_layer(x)
        x = self.maxpooling_2_layer(x)
        x = self.flatten_layer(x)
        try:
            x = self.dense_1_layer(x)
        except RuntimeError as e:
            print(x.shape)
            raise e
        x = self.relu_3_layer(x)
        x = self.dropout_layer(x)
        return self.dense_2_layer(x)

    def train_epoch(self, X_train: torch.Tensor, Y_train: torch.Tensor):
        train_data = torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_data, batch_size=32, shuffle=True)
        self.train()
        loss_sum = 0.0
        steps = 0
        for x, y in tqdm(train_loader):
            x, y = x.to(self.device), y.to(self.device)
            yhat = self(x)
            loss = self.criterion(y, yhat)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_sum += loss.item()
            steps += 1
            if steps % 10_000 == 0:
                print('Train loss: {0} step {1}'.format(float(loss_sum / steps), steps))
        return loss_sum / steps

    def evaluate(self, X, Y):
        self.eval()
        yhat = self(X)
        return self.criterion(yhat, Y.to(self.device)).item()
