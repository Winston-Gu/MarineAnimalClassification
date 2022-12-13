import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class MyNet(BaseModel):

    def __init__(
        self,
        out_classes,
        BatchNorm=False,
    ):
        super(MyNet, self).__init__()

        if BatchNorm:
            self.features = nn.Sequential(
                nn.Conv2d(in_channels=3,
                          out_channels=5,
                          kernel_size=3,
                          padding=1), nn.BatchNorm2d(5), nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=5,
                          out_channels=64,
                          kernel_size=3,
                          padding=1), nn.MaxPool2d(2, 2), nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64,
                          out_channels=128,
                          kernel_size=3,
                          padding=1), nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=128,
                          out_channels=256,
                          kernel_size=3,
                          padding=1), nn.MaxPool2d(2, 2), nn.BatchNorm2d(256),
                nn.ReLU(inplace=True))
        else:
            self.features = nn.Sequential(
                nn.Conv2d(in_channels=3,
                          out_channels=5,
                          kernel_size=3,
                          padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=5,
                          out_channels=64,
                          kernel_size=3,
                          padding=1), nn.MaxPool2d(2,
                                                   2), nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64,
                          out_channels=128,
                          kernel_size=3,
                          padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=128,
                          out_channels=256,
                          kernel_size=3,
                          padding=1), nn.MaxPool2d(2, 2),
                nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(nn.Linear(256, 1024),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(1024, out_classes))

    def forward(self, x):
        x = self.features(x)
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.classifier(x)
        return x


def get_pretrained_model(model_name, out_classes):
    """
    Get a pretrained model
    :param model_name: Name of the model
    :param out_classes: Number of output classes
    :return: Model
    """
    if model_name == 'resnet50':
        resnet50_model = timm.create_model('resnet50d',
                                           pretrained=True,
                                           num_classes=out_classes)
        return resnet50_model
    elif model_name == 'vit':
        vit_model = timm.create_model('vit_large_patch16_224',
                                      pretrained=True,
                                      num_classes=out_classes)
        return vit_model
    else:
        raise ValueError('Model not found: {}'.format(model_name))


if __name__ == '__main__':
    model = get_pretrained_model('resnet50', 19)
    print(model)
    model = get_pretrained_model('vit', 19)
    print(model)