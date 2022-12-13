import argparse
import wandb
import importlib
from easydict import EasyDict
import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm.auto import tqdm


from utils import get_data_loaders, get_model



parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-name', default='mynet_pro')
args = parser.parse_args()

config_module = importlib.import_module(f'config.{args.config_name}')
config = config_module.config
config.update(vars(args))
config = EasyDict(config)

exp_name = f'{config.model}'
exp_name += f'_{config.data}'
exp_name += f'_{config.epochs}'
exp_name += f'_{config.lr}'
exp_name += f'_{config.batch_size}'
exp_name += f'_optim_{config.optimizer}'
exp_name += f'_BN_{config.BatchNorm}'
exp_name += f'_TF_{config.transform}'

wandb.init(project="MarineAnimals", config=config, name=exp_name)


def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # forward pass
        outputs = model(image)
        # calculate the loss
        loss = criterion(outputs, labels)
        train_loss = loss.item()
        train_running_loss += train_loss
        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_acc = (preds == labels).sum().item()
        train_running_correct += train_acc
        # backpropagation
        loss.backward()
        # update the optimizer parameters
        optimizer.step()
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc


def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)
            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc










def save_model(config, epoch, model, optimizer):
    """
    Function to save the trained model to disk.
    """
    save_name = f'{config.model}'
    save_name += f'_{config.data}'
    save_name += f'_{config.epochs}'
    save_name += f'_{config.lr}'
    save_name += f'_{config.batch_size}'
    save_name += f'_optim_{config.optimizer}'
    save_name += f'_BN_{config.BatchNorm}'
    save_name += f'_TF_{config.transform}'

    save_folder = f'saved_model/{save_name}'
    # check if the folder exists
    if not pathlib.Path(save_folder).exists():
        pathlib.Path(save_folder).mkdir(exist_ok=True)

    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'{save_folder}/{epoch}.pth')


if __name__ == '__main__':
    lr = config.lr
    epochs = config.epochs
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}\n")

    model = get_model(config)
    model.to(device)

    # total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in model.parameters()
                                 if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # optimizer
    if config.optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=lr)
    if config.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    # loss function
    criterion = nn.CrossEntropyLoss()

    # get data loaders
    train_loader, valid_loader = get_data_loaders(config)

    # lists to keep track of losses and accuracies
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    # start the training loop
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader,
                                                  optimizer, criterion)
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,
                                                     criterion)
        wandb.log({
            "train_loss": train_epoch_loss,
            "train_acc": train_epoch_acc,
            "valid_loss": valid_epoch_loss,
            "valid_acc": valid_epoch_acc
        })
        # save the trained model weights
        if (epoch + 1) % 1 == 0:
            save_model(config, epoch, model, optimizer)
    print('TRAINING COMPLETE')