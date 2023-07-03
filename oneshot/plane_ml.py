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
        x,y = vals, row['objective']
        return x, y 

    def get_sampler(self):
        buckets = {}

        for i in range(len(self)):
            row = self.data.iloc[i]
            bucket = int(row['objective']*10-1e-4)/10
            if bucket not in buckets:
                buckets[bucket] = 0
            buckets[bucket] += 1

        weights = []
        for i in range(len(self)):
            row = self.data.iloc[i]
            bucket = int(row['objective']*10-1e-4)/10
            weights.append(1/buckets[bucket])

        weights = torch.tensor(weights)
        return torch.utils.data.sampler.WeightedRandomSampler(weights, len(self), replacement=True)



def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = nn.Sequential(
        nn.LazyLinear(8096),
        nn.LeakyReLU(0.1),
        nn.LazyLinear(4024),
        nn.LeakyReLU(0.1),
        nn.LazyLinear(4024),
        nn.LeakyReLU(0.1),
        nn.LazyLinear(512),
        nn.LeakyReLU(0.1),
        nn.LazyLinear(1)
    ).to(device)

    loss_fn = nn.MSELoss(reduction = 'mean').to(device)

    train_dataset = PlaneDataset('3x3-1k-test.csv')
    test_dataset = PlaneDataset('3x3-10k.csv')
    #num_train = int(len(dataset) * 0.9)
    #num_test = len(dataset) - num_train
    #train_dataset, test_dataset = random_split(dataset, [num_train, num_test])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size = 1,
        pin_memory = True,
        num_workers = 2,
        drop_last = True,
        sampler = train_dataset.get_sampler()
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size = 128,
        pin_memory = True,
        num_workers = 2,
        drop_last = True,
        sampler = test_dataset.get_sampler()
    )

    optimizer = torch.optim.SGD(
        params = model.parameters(),
        lr = 1e-2,
        weight_decay = 1e-5,
        momentum = 0.9,
        nesterov = True
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, verbose = True)

    for epoch in range(199):
        run_epoch(device, model, optimizer, loss_fn, train_dataloader, train=True)
        run_epoch(device, model, optimizer, loss_fn, test_dataloader, train=False)
        scheduler.step()


def run_epoch(device, model, optimizer, loss_fn, dataloader, train):
    if train:
        model.train()
    else:
        model.eval()

    loop = tqdm(iter(dataloader), leave=True)
    losses = []
    preds = []
    truth = []
    for x, y in loop:
        x = x.float().to(device)
        y = y.float().to(device)
        out = model(x).flatten()
        preds += out.flatten().cpu().tolist()
        truth += y.flatten().cpu().tolist()
        loss = loss_fn(out, y)
        losses.append(loss.cpu().item())

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loop.set_postfix(loss = sum(losses)/len(losses))
        
    if train:
        plt.clf()

    plt.scatter(truth, preds)
    plt.savefig('ml_3x3_1.png')


if __name__ == '__main__':
    main()
