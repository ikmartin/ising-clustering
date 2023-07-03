import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
import matplotlib.pyplot as plt

class PlaneDataset(Dataset):
    def __init__(self, filename):
        self.data = pd.read_csv(filename)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        row = self.data.iloc[index]
        vals = torch.tensor(literal_eval(row['planes']))
        x,y = vals, 100*row['objective']
        return x, y 


def main():
    model = nn.Sequential(
        nn.LazyLinear(4096),
        nn.LeakyReLU(0.1),
        nn.LazyLinear(4096),
        nn.LeakyReLU(0.1),
        nn.LazyLinear(4096),
        nn.LeakyReLU(0.1),
        nn.LazyLinear(1)
    )

    loss_fn = nn.MSELoss(reduction = 'mean')

    dataset = PlaneDataset('3x3.csv')
    num_train = int(len(dataset) * 0.9)
    num_test = len(dataset) - num_train
    train_dataset, test_dataset = random_split(dataset, [num_train, num_test])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size = 64,
        shuffle = True,
        drop_last = True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size = 64,
        shuffle = True,
        drop_last = True
    )

    optimizer = torch.optim.SGD(
        params = model.parameters(),
        lr = 1e-4,
        weight_decay = 1e-5,
        momentum = 0.9,
        nesterov = True
    )

    optimizer = torch.optim.Adam(
        params = model.parameters()
    )

    for epoch in range(20):
        run_epoch(model, optimizer, loss_fn, train_dataloader, train=True)
        run_epoch(model, optimizer, loss_fn, test_dataloader, train=False)


def run_epoch(model, optimizer, loss_fn, dataloader, train):
    if train:
        model.train()
    else:
        model.eval()

    loop = tqdm(iter(dataloader), leave=True)
    losses = []
    preds = []
    truth = []
    for x, y in loop:
        x = x.float()
        y = y.float()
        out = model(x)
        preds += out.flatten().tolist()
        truth += y.flatten().tolist()
        loss = loss_fn(out, y)
        losses.append(loss.item())

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loop.set_postfix(loss = sum(losses)/len(losses))
        
    if not train:
        plt.scatter(truth, preds)
        plt.savefig('ml_3x3_1.png')


if __name__ == '__main__':
    main()
