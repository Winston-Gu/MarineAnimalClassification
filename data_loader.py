import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


class MarineDataLoader():

    def __init__(self, data_path, batch_size, num_workers):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def load_data(self, train=True):
        if train:
            train_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                transforms.RandomRotation(degrees=(30, 70)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            train_dataset = datasets.ImageFolder(root=self.data_path,
                                                 transform=train_transform)

            train_loader = DataLoader(train_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=self.num_workers,
                                      shuffle=True)

            return train_loader

        else:
            valid_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            valid_dataset = datasets.ImageFolder(root=self.data_path,
                                                 transform=valid_transform)

            valid_loader = DataLoader(valid_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=self.num_workers,
                                      shuffle=False)

            return valid_loader