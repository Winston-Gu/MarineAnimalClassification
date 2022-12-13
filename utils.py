from data_loader import MarineDataLoader
from model import MyNet, get_pretrained_model
import torch.nn as nn


def get_data_loaders(config):
    if config.data == 'inbalanced':
        train_loader = MarineDataLoader('data_inbalanced/train',
                                        config.batch_size,
                                        num_workers=4)
        train_data_loader = train_loader.get_loader(train=True,
                                                    transform=config.transform)
        valid_loader = MarineDataLoader('data_inbalanced/valid',
                                        config.batch_size,
                                        num_workers=4)
        valid_data_loader = valid_loader.get_loader(train=False)

    elif config.data == 'balanced':
        train_loader = MarineDataLoader('data/train',
                                        config.batch_size,
                                        num_workers=4)
        train_data_loader = train_loader.get_loader(train=True,
                                                    transform=config.transform)
        valid_loader = MarineDataLoader('data/valid',
                                        config.batch_size,
                                        num_workers=4)
        valid_data_loader = valid_loader.get_loader(train=False)

    elif config.data == 'pro':
        train_loader = MarineDataLoader(
            'data/train_gauss_sp_blur_rotate_flip_random_erasing',
            config.batch_size,
            num_workers=4)
        train_data_loader = train_loader.get_loader(train=True,
                                                    transform=config.transform)
        valid_loader = MarineDataLoader('data/valid',
                                        config.batch_size,
                                        num_workers=4)
        valid_data_loader = valid_loader.get_loader(train=False)

    return train_data_loader, valid_data_loader


def get_test_data_loader(config):
    data_loader = MarineDataLoader('data/test',
                                   config.batch_size,
                                   num_workers=4)
    test_data_loader = data_loader.get_loader(train=False)
    return test_data_loader


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def get_model(config):
    if config.model == 'mynet':
        if config.BatchNorm:
            model = MyNet(out_classes=19, BatchNorm=True)
        else:
            model = MyNet(out_classes=19)
        model.apply(weight_init)
    elif config.model == 'resnet50':
        model = get_pretrained_model('resnet50', 19)
    elif config.model == 'vit':
        model = get_pretrained_model('vit', 19)
    return model