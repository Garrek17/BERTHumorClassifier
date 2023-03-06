from torch.utils.data import Dataset
from dataloader import train_dataset, test_dataset
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    print(device)

#Parameters
num_epochs = 800
batch_size = 512
learning_rate = .001

#Load the dataset
print("Starting loading...")
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           shuffle=True,
                                           batch_size=batch_size,
                                           drop_last=True)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                           shuffle=True,
                                           batch_size=batch_size,
                                           drop_last=True)
print("Finished loading...")

#Define feedforward network
class Feedforward(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Feedforward, self).__init__()
        self.generator_stack = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 4),
            nn.LeakyReLU(),
            nn.Linear(4, output_dim),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.generator_stack(X)

#Training
def train(model):
    FF = Feedforward(64, 1).to(device)
    lossFF = nn.BCELoss()
    optimizer = torch.optim.Adam(list(FF.parameters()), lr = learning_rate)

    metrics = []
    for ep in range(num_epochs):
        train_loss = []
        test_loss = []
        FF.train()
        for index, batch_inputs in enumerate(train_loader):
            batch_X, batch_y = batch_inputs
            batch_y = torch.reshape(batch_y.to(device), (batch_size, 1))
            optimizer.zero_grad()
            prediction = FF(batch_X.to(device))
            loss = lossFF(prediction.to(torch.float32).to(device), batch_y.to(torch.float32).to(device))
            loss.backward()
            optimizer.step()
            train_loss.append(loss)

        FF.eval()
        for batches in test_loader:
            X, y = batches
            y = torch.reshape(y.to(device), (batch_size, 1))
            pred = FF(X)
            loss_test = lossFF(pred.to(torch.float32).to(device), y.to(torch.float32).to(device))
            test_loss.append(loss_test)

        train_epoch_loss = sum(train_loss)/len(train_loss)
        test_epoch_loss = sum(test_loss) / len(test_loss)
        metrics.append((train_epoch_loss, test_epoch_loss))
        print("Train Loss: {}\t Test Loss: {}\t at epoch: {}".format(
            train_epoch_loss, test_epoch_loss, ep))

    return metrics, FF

if __name__ == '__main__':
    print("Beginning Training...")
    FF = torch.load("Model/feedforward.pth", map_location=torch.device(device))
    metrics, model = train(FF)
    torch.save(model, "Model/feedforward.pth")








