import argparse
import importlib
from easydict import EasyDict
import tqdm

from sklearn.metrics import classification_report

from utils import get_model, get_test_data_loader


import torch

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-name', default='mynet_inbalanced')
parser.add_argument('-n', '--checkpoint_num', default='99')
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
    checkpoint = torch.load(f'saved_model/{load_name}/{config.checkpoint_num}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    test_loader = get_test_data_loader(config)
    y_true = []
    y_pred = []
    for i, data in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        outputs = model(image)
        _, preds = torch.max(outputs.data, 1)
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())
    print(classification_report(y_true, y_pred))