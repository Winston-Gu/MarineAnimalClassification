import pathlib
import numpy as np
import shutil


class RawData(object):
    """_summary_

    Args:
        object (_type_): _description_
    """

    def __init__(self, raw_folder_path):
        self.path = pathlib.Path(__file__).resolve()
        self.project_folder = self.path.parent
        self.raw_path = self.project_folder / raw_folder_path
        # check if is folder
        if not self.raw_path.is_dir():
            raise Exception("Not a folder")

        self.sub_folder_list = [
            sub_folder for sub_folder in self.raw_path.iterdir()
            if sub_folder.is_dir()
        ]
        self.label_list = self.get_label_list()
        self.label_num_dict = self.get_each_label_num()

    def get_label_list(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return [sub_folder.name for sub_folder in self.sub_folder_list]

    def get_each_label_num(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        label_num_dict = {}
        for sub_folder in self.sub_folder_list:
            data_num = len([
                file for file in sub_folder.iterdir() if file.is_file() and (
                    file.suffix == '.jpg' or file.suffix == '.png')
            ])
            label_num_dict[sub_folder.name] = data_num
        return label_num_dict


class SplitData(object):
    """_summary_

    Args:
        object (_type_): _description_
    """

    def __init__(self,
                 raw_data,
                 train_ratio=0.8,
                 valid_ratio=0.1,
                 test_ratio=0.1):
        self.raw_data = raw_data
        self.project_folder = raw_data.project_folder
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio

    def split_data(self, balanced=False, random_seed=0):
        np.random.seed(random_seed)
        if not balanced:
            data_folder_path = self.project_folder / 'data_inbalanced'
            data_folder_path.mkdir(parents=True)
            for label in self.raw_data.label_list:
                data_num = self.raw_data.label_num_dict[label]
                img_list = [
                    img for img in (self.raw_data.raw_path / label).iterdir()
                    if img.is_file() and (
                        img.suffix == '.jpg' or img.suffix == '.png')
                ]
                num_list = np.arange(data_num)
                np.random.shuffle(num_list)

                train_folder = data_folder_path / 'train' / label
                train_folder.mkdir(parents=True)

                valid_folder = data_folder_path / 'valid' / label
                valid_folder.mkdir(parents=True)
                test_folder = data_folder_path / 'test' / label
                test_folder.mkdir(parents=True)

                train_num = int(data_num * self.train_ratio)
                valid_num = int(data_num * self.valid_ratio)
                test_num = int(data_num * self.test_ratio)

                for i in range(data_num):
                    img = img_list[num_list[i]]
                    if i < train_num:
                        new_img = train_folder / (f"{i}" + img.suffix)
                        shutil.copyfile(img, new_img)
                    elif i < train_num + valid_num:
                        new_img = valid_folder / (f"{i - train_num}" +
                                                  img.suffix)
                        shutil.copyfile(img, new_img)
                    else:
                        new_img = test_folder / (
                            f"{i - train_num - valid_num}" + img.suffix)
                        shutil.copyfile(img, new_img)
        else:
            data_folder_path = self.project_folder / 'data'
            data_folder_path.mkdir(parents=True)
            min_num = min(self.raw_data.label_num_dict.values())
            for label in self.raw_data.label_list:
                data_num = self.raw_data.label_num_dict[label]
                img_list = [
                    img for img in (self.raw_data.raw_path / label).iterdir()
                    if img.is_file() and (
                        img.suffix == '.jpg' or img.suffix == '.png')
                ]
                num_list = np.arange(data_num)
                num_list = num_list[:min_num]
                np.random.shuffle(num_list)

                train_folder = data_folder_path / 'train' / label
                train_folder.mkdir(parents=True)
                valid_folder = data_folder_path / 'valid' / label
                valid_folder.mkdir(parents=True)
                test_folder = data_folder_path / 'test' / label
                test_folder.mkdir(parents=True)

                train_num = int(min_num * self.train_ratio)
                valid_num = int(min_num * self.valid_ratio)
                test_num = int(min_num * self.test_ratio)

                for i in range(min_num):
                    img = img_list[num_list[i]]
                    if i < train_num:
                        new_img = train_folder / (f"{i}" + img.suffix)
                        shutil.copyfile(img, new_img)
                    elif i < train_num + valid_num:
                        new_img = valid_folder / (f"{i - train_num}" +
                                                  img.suffix)
                        shutil.copyfile(img, new_img)
                    else:
                        new_img = test_folder / (
                            f"{i - train_num - valid_num}" + img.suffix)
                        shutil.copyfile(img, new_img)


class TrainingData(object):

    def __init__(self, training_folder_path):
        self.path = pathlib.Path(__file__).resolve()
        self.project_folder = self.path.parent
        self.training_path = self.project_folder / training_folder_path
        self.train_folder = self.training_path / 'train'
        self.valid_folder = self.training_path / 'valid'
        self.test_folder = self.training_path / 'test'

        self.label_list = self.get_label_list()
        self.label_num_dict = self.get_each_label_num()

    def get_label_list(self):
        train_label = [label.name for label in self.train_folder.iterdir()]
        valid_folder = [label.name for label in self.valid_folder.iterdir()]
        test_folder = [label.name for label in self.test_folder.iterdir()]

        if train_label == valid_folder == test_folder:
            return train_label
        else:
            raise ValueError('train_label != valid_folder != test_folder')

    def get_each_label_num(self):
        label_num_dict = {}
        for label in self.label_list:
            train_num = len(list((self.train_folder / label).iterdir()))
            valid_num = len(list((self.valid_folder / label).iterdir()))
            test_num = len(list((self.test_folder / label).iterdir()))
            label_num_dict[label] = {
                'train': train_num,
                'valid': valid_num,
                'test': test_num
            }
        return label_num_dict


if __name__ == "__main__":
    raw_data = RawData('raw_data')
    split_data = SplitData(raw_data)
    split_data.split_data(balanced=False)
    split_data.split_data(balanced=True)
    inbanlanced_data = TrainingData('data_inbalanced')
    balanced_data = TrainingData('data')
    if inbanlanced_data.label_list == raw_data.label_list:
        print('inbalanced label_list is same')
    if balanced_data.label_list == raw_data.label_list:
        print('balanced label_list is same')

    print(inbanlanced_data.label_num_dict)
    for label in inbanlanced_data.label_list:
        if inbanlanced_data.label_num_dict[label][
                'train'] + inbanlanced_data.label_num_dict[label][
                    'valid'] + inbanlanced_data.label_num_dict[label][
                        'test'] == raw_data.label_num_dict[label]:
            print('inbalanced data num is same')
    print(balanced_data.label_num_dict)
