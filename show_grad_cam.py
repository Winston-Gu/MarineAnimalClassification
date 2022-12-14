import argparse
import importlib
from easydict import EasyDict
import tqdm
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from utils import get_model, get_test_data_loader

from matplotlib import pyplot as plt

import torch

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-name', default='mynet_inbalanced')
parser.add_argument('-n', '--checkpoint_num', default='59')
parser.add_argument('-i', '--img_path', default='data/test/Corals/0.jpg')
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

    img = Image.open(config.img_path).resize((224, 224))
    rgb_img = np.float32(img) / 255
    img_tensor = torch.from_numpy(rgb_img).permute(2, 0,
                                                   1).unsqueeze(0).to(device)

    target_layers = [model.features[-1]]
    # 选取合适的类激活图，但是ScoreCAM和AblationCAM需要batch_size
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(10)]
    # 上方preds需要设定，比如ImageNet有1000类，这里可以设为200
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    imgplot = plt.imshow(cam_img)
    imgplot.figure.savefig(f'gradcam_result/{load_name}.png')