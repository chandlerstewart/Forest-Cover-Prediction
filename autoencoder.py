import torch
import torch.nn as nn
from tqdm import trange



class AutoEncoder(nn.Module):

    def __init__(self, in_features, latent, lr=0.001, batch_size=512, epochs=500):
        super(AutoEncoder, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        self.Encoder = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Linear(in_features, int(in_features*3/4)),
            nn.ReLU(),
            nn.Linear(int(in_features*3/4), int(in_features*1/2)),
            nn.ReLU(),
            nn.Linear(int(in_features*1/2), int(in_features*1/4)),
            nn.ReLU(),
            nn.Linear(int(in_features*1/4), latent),
        )

        self.Decoder = nn.Sequential(
            nn.Linear(latent, int(in_features*1/4)),
            nn.ReLU(),
            nn.Linear(int(in_features*1/4),int(in_features*1/2)),
            nn.ReLU(),
            nn.Linear(int(in_features*1/2), int(in_features*3/4)),
            nn.ReLU(),
            nn.Linear(int(in_features*3/4), in_features),
            nn.Linear(in_features, in_features),
        )

        self.model = nn.Sequential(
            self.Encoder,
            self.Decoder
        )


        self = self.to(self.device)

    def forward(self,X):
        return self.model(X)


    def fit(self,X):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        X = torch.tensor(X.to_numpy()).type(torch.float32).to(self.device)

        self.epoch_losses = []
        with trange(self.epochs, desc="Epoch") as tepoch:
            for epoch in tepoch:
                batch_losses = []
                permutation = torch.randperm(X.shape[0])
                for i in range(0,X.shape[0], self.batch_size):
                    indices = permutation[i:i+self.batch_size]
                    batch_x= X[indices]

                    optimizer.zero_grad()
                    outputs = self.forward(batch_x)
                    loss = criterion(outputs,batch_x)

                    loss.backward()
                    optimizer.step()

                    batch_losses.append(loss.item())

                batch_losses = torch.mean(torch.tensor(batch_losses))
                self.epoch_losses.append(batch_losses)
                tepoch.set_postfix(loss=batch_losses)

        return self.epoch_losses