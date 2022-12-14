import os

config_field = [
    'mynet_inbalanced', 'mynet_balanced', 'mynet_pro', 'mynet_pro_batchnorm',
    'mynet_pro_batchnorm_adam', 'resnet_pro', 'vit_pro',
    'mynet_pro_batchnorm_adam_transform', 'mynet_balanced_transform'
]

if __name__ == '__main__':
    for config in config_field:
        os.system('python train.py -c {}'.format(config))