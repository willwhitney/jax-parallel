from __future__ import print_function
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from dataset_wrappers import DatasetCache


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=14, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument('--fullbatch', action='store_true', default=False)
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)


def stack_data(loader, dataset_len):
    i = 0
    for x, y in loader:
        if i == 0:
            xs = torch.empty((dataset_len, *x.shape[1:]), dtype=x.dtype)
            ys = torch.empty((dataset_len, *y.shape[1:]), dtype=y.dtype)
        xs[i: i + x.shape[0]] = torch.as_tensor(x)
        ys[i: i + y.shape[0]] = torch.as_tensor(y)
        i += x.shape[0]
    return xs, ys

device = torch.device("cuda" if True else "cpu")

train_kwargs = {'batch_size': args.batch_size}
test_kwargs = {'batch_size': args.test_batch_size}
if True:
    cuda_kwargs = {'num_workers': 0,
                   'pin_memory': True,
                   'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)


transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
dataset1 = datasets.MNIST('../data', train=True, download=True,
                   transform=transform)
dataset2 = datasets.MNIST('../data', train=False,
                   transform=transform)
                   
train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

train_x, train_y = stack_data(train_loader, len(dataset1))
train_x = train_x.to(device)
train_y = train_y.to(device)

test_x, test_y = stack_data(test_loader, len(dataset2))
test_x = test_x.to(device)
test_y = test_y.to(device)


def train_fullbatch(args, model, device, optimizer, epoch):
    model.train()
    batch_idx = 0
    data = train_x
    target = train_y
    # data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % args.log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test_fullbatch(model, device):
    model.eval()
    test_loss = 0
    correct = 0

    output = model(test_x)
    test_loss += F.nll_loss(output, test_y, reduction='sum').item()  # sum up batch loss
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(test_y.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


def main():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.Tanh(),
        nn.Linear(128, 128),
        nn.Tanh(),
        nn.Linear(128, 10),
        nn.LogSoftmax()
    ).to(device)
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    results = {
        'epoch': [],
        'accuracy': [],
        'mode': []
    }
    
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        if args.fullbatch:
            train_fullbatch(args, model, device, optimizer, epoch)
            accuracy = test_fullbatch(model, device)
        else:
            train(args, model, device, train_loader, optimizer, epoch)
            accuracy = test(model, device, test_loader)        
        results['epoch'].append(epoch)
        results['accuracy'].append(accuracy)
        results['mode'].append('full' if args.fullbatch else 'stochastic')
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"mnist_full{args.fullbatch}.csv")

        scheduler.step()
    stop = time.time()
    print(f"Training time: {stop - start :.2f}s")
    # test(model, device, test_loader)
    

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()