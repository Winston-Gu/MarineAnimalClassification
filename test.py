import argparse
import importlib
from easydict import EasyDict
import tqdm

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from utils import get_model, get_test_data_loader

from matplotlib import pyplot as plt

import torch

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-name', default='mynet_inbalanced')
parser.add_argument('-n', '--checkpoint_num', default='59')
args = parser.parse_args()

config_module = importlib.import_module(f'config.{args.config_name}')
config = config_module.config
config.update(vars(args))
config = EasyDict(config)

load_name = f'{config.model}'
load_name += f'_{config.data}'
load_name += f'_{config.epochs}'
load_name += f'_{config.lr}'
load_name += f'_{config.batch_size}'
load_name += f'_optim_{config.optimizer}'
load_name += f'_BN_{config.BatchNorm}'
load_name += f'_TF_{config.transform}'

device = ('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    model = get_model(config).to(device)
    checkpoint = torch.load(
        f'saved_model/{load_name}/{config.checkpoint_num}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    test_loader = get_test_data_loader(config)
    y_true = []
    y_pred = []
    marine_classes = [
        'Corals', 'Crabs', 'Dolphin', 'Eel', 'Jelly Fish', 'Lobster',
        'Nudibranchs', 'Octopus', 'Penguin', 'Puffers', 'Sea Rays',
        'Sea Urchins', 'Seahorse', 'Seal', 'Sharks', 'Squid', 'Starfish',
        'Turtle_Tortoise', 'Whale'
    ]
    for i, data in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        outputs = model(image)
        _, preds = torch.max(outputs.data, 1)
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())
    print(classification_report(y_true, y_pred, target_names=marine_classes))
    confusion = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion,
                                  display_labels=marine_classes)
    fig, ax = plt.subplots(figsize=(12, 11))
    disp.plot(
        include_values=True,
        cmap="Blues",
        ax=ax,
        xticks_rotation="vertical",
        values_format="d",
    )

    plt.savefig(f'confusion_matrix/{load_name}.png')