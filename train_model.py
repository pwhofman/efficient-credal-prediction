"""Basic script for training a model."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import models
import data
import utils
from tqdm import tqdm
import os

MODEL_FOLDER = './checkpoints/'
os.environ["TORCH_HOME"] = MODEL_FOLDER


def _str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true', 't', '1'):
        return True
    elif s.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train_model(model, train_loader, device, Optimizer, lr, wd, epochs, val_loader=None):
    criterion = nn.CrossEntropyLoss()
    if isinstance(Optimizer, optim.SGD):
        optimizer = Optimizer(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        optimizer = Optimizer(model.parameters(), lr=lr, weight_decay=wd)
    if isinstance(optimizer, optim.SGD):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'After epoch {epoch}, loss {running_loss / len(train_loader)}')

        if isinstance(optimizer, optim.SGD):
            scheduler.step()

        if val_loader and (epoch + 1) % 10 == 0:
            model.eval()
            correct, total, val_loss = 0, 0, 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    correct += (outputs.argmax(dim=1) == targets).sum().item()
                    total += targets.size(0)
                    val_loss += criterion(outputs, targets).item()
            val_loss /= len(val_loader)
            print(f'After epoch {epoch}, validation loss: {val_loss:.3f}, accuracy: {correct / total * 100:.3f}%')
    return model


def main(args):
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Set device
    if torch.cuda.is_available():
        device = utils.get_best_gpu()
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = 'cpu'

    print(f'Using device: {device} and the following args:')
    utils.printargs(args)

    train_loader, val_loader, test_loader = data.get_data_train(args.dataset,
                                                                batch_size=128,
                                                                validation=args.validate,
                                                                seed=args.seed)
    model = models.get_model(args.model, args.classes)
    model = model.to(device)

    res_model = train_model(model, train_loader, device, optim.Adam if args.optimizer == "adam" else optim.SGD, args.lr,
                            args.wd,
                            args.epochs, val_loader=val_loader if args.validate else None)

    torch.save(res_model.state_dict(), f'{MODEL_FOLDER}{args.dataset}_{args.model}_{args.seed}.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset')
    parser.add_argument('--validate', type=_str2bool, help='Use validation set (20%) or not?')
    parser.add_argument('--classes', type=int, help='Number of classes')
    parser.add_argument('--model', type=str, help='Base model to use')
    parser.add_argument('--optimizer', type=str, help='The optimizer to use')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--wd', type=float, help='Weight decay')
    parser.add_argument('--seed', type=int, help='Seed')
    args = parser.parse_args()
    main(args)

# python train_model.py --validate True --dataset cifar10 --classes 10 --model resnet --optimizer adam --epochs 20 --lr 1e-05 --wd 0.001 --seed 1
