import numpy as np
import pandas as pd
import torch

class Network(torch.nn.Module):

    def __init__(self, dimensions):
        super(Network, self).__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(dimensions[i], dimensions[i+1]) for i in range(len(dimensions) - 1)])

    def forward(self, input):
        out = torch.nn.functional.relu(self.layers[0](input))
        for layer in self.layers[1:-1]:
            out = torch.nn.functional.relu(layer(out))
        return torch.sigmoid(self.layers[-1](out))


model = Network([4, 30, 30, 3])
optimiser = torch.optim.Adam(model.parameters(), lr=0.1)
loss_func = torch.nn.BCELoss()

data = np.loadtxt("iris.dat")
np.random.shuffle(data)

x = data[:, :4].astype(np.float32)
y = data[:, 4:].astype(np.float32)

split_idx = int(0.8 * len(x))

x_train = torch.tensor(x[:split_idx])
y_train = torch.tensor(y[:split_idx])
x_val = torch.tensor(x[split_idx:])
y_val = torch.tensor(y[split_idx:])



n_epochs = 1000  # or whatever
batch_size = 50  # or whatever

for epoch in range(n_epochs):

    print('Training ... Epoch {}'.format(epoch))

    # X is a torch Variable
    permutation = torch.randperm(x_train.size()[0])

    for i in range(0, x_train.size()[0], batch_size):
        optimiser.zero_grad()

        indices = permutation[i:i + batch_size]
        batch_x, batch_y = x_train[indices], y_train[indices]

        # in case you wanted a semi-full example
        outputs = model.forward(batch_x)
        loss = loss_func(outputs, batch_y)

        loss.backward()
        optimiser.step()

        print('Current Loss = {}'.format(loss.item()))

preds = model.forward(x_val).argmax(axis=1).squeeze()

print(preds)
targets = y_val.argmax(axis=1).squeeze()
print(targets)
accuracy = (preds == targets).mean()
print("Validation accuracy: {}".format(accuracy))