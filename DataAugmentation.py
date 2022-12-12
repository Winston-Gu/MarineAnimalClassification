import cv2
import numpy as np
import os.path
import copy
import pathlib


class DataAugmentation(object):
    """_summary_

    Args:
        object (_type_): _description_
    """

    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)

    def add_salt_and_pepper(self, ratio=0.42):
        """_summary_

        Args:
            ratio (float, optional): _description_. Defaults to 0.42.

        Returns:
            _type_: _description_
        """
        sp_noise_img = self.image.copy()
        img_w = self.image.shape[1]
        img_h = self.image.shape[0]
        sp_noise_num = int(ratio * self.image.shape[0] * self.image.shape[1])
        for _ in range(sp_noise_num):
            temp_x = np.random.randint(0, img_h)
            temp_y = np.random.randint(0, img_w)
            if np.random.randint(0, 1) == 0:
                sp_noise_img[temp_x][temp_y][np.random.randint(3)] = 0
            else:
                sp_noise_img[temp_x][temp_y][np.random.randint(3)] = 255
        return sp_noise_img

    def add_gaussian_noise(self, ratio=0.42):
        """_summary_

        Args:
            ratio (float, optional): _description_. Defaults to 0.42.

        Returns:
            _type_: _description_
        """
        gaussian_noise_img = self.image.copy()
        img_w = self.image.shape[1]
        img_h = self.image.shape[0]
        gaussian_noise_num = int(ratio * self.image.shape[0] *
                                 self.image.shape[1])
        for i in range(gaussian_noise_num):
            temp_x = np.random.randint(0, img_h)
            temp_y = np.random.randint(0, img_w)
            gaussian_noise_img[temp_x][temp_y][np.random.randint(
                3)] = np.random.randn(1)[0]
        return gaussian_noise_img

    def darker(self, ratio=0.9):
        """_summary_

        Args:
            ratio (float, optional): _description_. Defaults to 0.9.

        Returns:
            _type_: _description_
        """
        dark_img = self.image.copy()
        img_w = self.image.shape[1]
        img_h = self.image.shape[0]
        for i in range(img_w):
            for j in range(img_h):
                dark_img[j, i, 0] = int(dark_img[j, i, 0] * ratio)
                dark_img[j, i, 1] = int(dark_img[j, i, 1] * ratio)
                dark_img[j, i, 2] = int(dark_img[j, i, 2] * ratio)
        return dark_img

    def brighter(self, ratio=1.5):
        """_summary_

        Args:
            ratio (float, optional): _description_. Defaults to 1.5.

        Returns:
            _type_: _description_
        """
        brt_img = self.image.copy()
        img_w = self.image.shape[1]
        img_h = self.image.shape[0]

        for i in range(img_w):
            for j in range(img_h):
                brt_img[j, i, 0] = np.clip(int(brt_img[j, i, 0] * ratio),
                                           a_max=255,
                                           a_min=0)
                brt_img[j, i, 1] = np.clip(int(brt_img[j, i, 1] * ratio),
                                           a_max=255,
                                           a_min=0)
                brt_img[j, i, 2] = np.clip(int(brt_img[j, i, 2] * ratio),
                                           a_max=255,
                                           a_min=0)
        return brt_img

    def rotate(self, angle, center=None, scale=1.0):
        """_summary_

        Args:
            angle (_type_): Angle of Rotation. Angle is positive for anti-clockwise and negative for clockwise.
            center (_type_, optional): _description_. Defaults to None.
            scale (float, optional): _description_. Defaults to 1.0.

        Returns:
            _type_: _description_
        """
        (img_h, img_w) = self.image.shape[:2]
        # If no rotation center is specified, the center of the image is set as the rotation center
        if center is None:
            center = (img_w / 2, img_h / 2)
        rotate_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(self.image, rotate_matrix, (img_w, img_h))
        return rotated

    def flip(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        fliped_img = cv2.flip(self.image, 1)
        return fliped_img

    def blur(self, kernel_size=3):
        """_summary_

        Args:
            kernel_size (int, optional): _description_. Defaults to 3.

        Returns:
            _type_: _description_
        """
        blur_img = cv2.blur(self.image, (kernel_size, kernel_size))
        return blur_img

    def random_erasing(self,
                       probability=0.5,
                       sl=0.02,
                       sh=0.4,
                       r1=0.3,
                       mean=[0.4914, 0.4822, 0.4465]):
        """_summary_

        Args:
            probability (_type_): _description_
            sl (_type_): _description_
            sh (_type_): _description_
            r1 (_type_): _description_
            mean (_type_, optional): _description_. Defaults to [0.4914, 0.4822, 0.4465].

        Returns:
            _type_: _description_
        """
        if np.random.uniform(0, 1) > probability:
            return self.image
        else:
            random_erased_img = self.image.copy()
            for _ in range(100):
                area = random_erased_img.shape[1] * random_erased_img.shape[0]
                target_area = np.random.uniform(sl, sh) * area
                aspect_ratio = np.random.uniform(r1, 1 / r1)
                h = int(round(np.sqrt(target_area * aspect_ratio)))
                w = int(round(np.sqrt(target_area / aspect_ratio)))
                if np.random.randint(2) == 0:
                    x1 = np.random.randint(random_erased_img.shape[1])
                    y1 = np.random.randint(random_erased_img.shape[0] - h)
                else:
                    x1 = np.random.randint(random_erased_img.shape[1] - w)
                    y1 = np.random.randint(random_erased_img.shape[0])
                if x1 + w <= random_erased_img.shape[
                        1] and y1 + h <= random_erased_img.shape[0]:
                    random_erased_img[y1:y1 + h, x1:x1 + w, 0] = mean[0]
                    random_erased_img[y1:y1 + h, x1:x1 + w, 1] = mean[1]
                    random_erased_img[y1:y1 + h, x1:x1 + w, 2] = mean[2]
                    return random_erased_img
            return random_erased_img


class FolderDataAugmentation(object):
    """_summary_

    Args:
        object (_type_): _description_
    """

    def __init__(self, folder_path):
        """_summary_

        Args:
            folder_path (_type_): relative path of the folder
        """

        self.path = pathlib.Path(__file__).resolve()
        self.project_folder = self.path.parent
        self.folder_path = self.project_folder / folder_path
        self.folder_name = folder_path.name

        # check if is folder
        if not self.folder_path.is_dir():
            raise Exception("Not a folder")

        self.sub_folder_list = [
            sub_folder for sub_folder in self.folder_path.iterdir()
            if sub_folder.is_dir()
        ]

    def make_folder(self, folder_name):
        """_summary_

        Args:
            folder_name (_type_): _description_
        """
        new_folder = self.folder_path.parent / folder_name
        if not new_folder.exists():
            new_folder.mkdir()
        return new_folder

    def folder_pro(self,
                   gauss_noise=False,
                   sp_noise=False,
                   darker= False,
                   brighter=False,
                   blur=False,
                   rotate=False,
                   random_erasing=False,
                   noise_ratio=0, 
                   dark_ratio = 0,
                   bright_ratio = 0,
                   blur_size = 0,
                   rotate_angle = 0,
                   rotate_num = 0,
                   random_erasing_probability = 0):
        """_summary_

        Args:
            gauss_noise (bool, optional): _description_. Defaults to False.
            sp_noise (bool, optional): _description_. Defaults to False.
            darker (bool, optional): _description_. Defaults to False.
            brighter (bool, optional): _description_. Defaults to False.
            blur (bool, optional): _description_. Defaults to False.
            rotate (bool, optional): _description_. Defaults to False.
            random_erasing (bool, optional): _description_. Defaults to False.
            noise_ratio (int, optional): _description_. Defaults to 0.
            dark_ratio (int, optional): _description_. Defaults to 0.
            bright_ratio (int, optional): _description_. Defaults to 0.
            blur_size (int, optional): _description_. Defaults to 0.
            rotate_angle (int, optional): _description_. Defaults to 0.
            rotate_num (int, optional): _description_. Defaults to 0.
            random_erasing_probability (int, optional): _description_. Defaults to 0.
        """
        pro_type_name = ""
        if gauss_noise:
            pro_type_name += "_gauss"
        if sp_noise:
            pro_type_name += "_sp"
        if darker:
            pro_type_name += "_dark"
        if brighter:
            pro_type_name += "_bright"
        if blur:
            pro_type_name += "_blur"
        if rotate:
            pro_type_name += "_rotate"
        if random_erasing:
            pro_type_name += "_random_erasing"

        new_folder = self.make_folder(self.folder_name + pro_type_name)
        for sub_folder in self.sub_folder_list:
            new_sub_folder = new_folder / sub_folder.name
            if not new_sub_folder.exists():
                new_sub_folder.mkdir()
            
        for sub_folder in self.sub_folder_list:
            img_list = [
                img for img in sub_folder.iterdir() if img.is_file() and (
                    img.suffix == ".jpg" or img.suffix == ".png")
            ]
            for img in img_list:
                # copy raw images to new folder
                raw_img = cv2.imread(str(img))
                new_img_path = new_folder / sub_folder.name / img.stem + "_raw" + img.suffix
                cv2.imwrite(str(new_img_path), raw_img)

                # add pro
                img_aug = DataAugmentation(raw_img)
                if gauss_noise:
                    gauss_img = img_aug.add_gaussian_noise(noise_ratio)
                    new_img_path = new_folder / sub_folder.name / img.stem + "_gauss" + img.suffix
                    cv2.imwrite(str(new_img_path), gauss_img)
                if sp_noise:
                    sp_img = img_aug.add_salt_and_pepper(noise_ratio)
                    new_img_path = new_folder / sub_folder.name / img.stem + "_sp" + img.suffix
                    cv2.imwrite(str(new_img_path), sp_img)
                if darker:
                    dark_img = img_aug.darker(dark_ratio)
                    new_img_path = new_folder / sub_folder.name / img.stem + "_dark" + img.suffix
                    cv2.imwrite(str(new_img_path), dark_img)
                if brighter:
                    bright_img = img_aug.brighter(bright_ratio)
                    new_img_path = new_folder / sub_folder.name / img.stem + "_bright" + img.suffix
                    cv2.imwrite(str(new_img_path), bright_img)
                if blur:
                    blur_img = img_aug.blur(blur_size)
                    new_img = new_folder / sub_folder.name / img.stem + "_blur" + img.suffix
                    cv2.imwrite(str(new_img), blur_img)
                if rotate:
                    if rotate_angle == 0:
                        for _ in range(rotate_num):
                            each_angle = np.random.randint(1, 360)
                            rotate_img = img_aug.rotate(each_angle)
                            new_img_path = new_folder / sub_folder.name / img.stem + "_rotate" + each_angle + img.suffix
                            cv2.imwrite(str(new_img_path), rotate_img)
                    else:
                        rotate_img = img_aug.rotate(rotate_angle)
                        new_img_path = new_folder / sub_folder.name / img.stem + "_rotate" + rotate_angle + img.suffix
                        cv2.imwrite(str(new_img_path), rotate_img)
                if random_erasing:
                    random_erasing_img = img_aug.random_erasing(random_erasing_probability)
                    new_img_path = new_folder / sub_folder.name / img.stem + "_random_erasing" + img.suffix
                    cv2.imwrite(str(new_img_path), random_erasing_img)


if __name__ == "__main__":
    test_folder = FolderDataAugmentation("test_folder/train")
    
    test_folder.folder_pro(gauss_noise=True, noise_ratio=0.3)
    test_folder.folder_pro(sp_noise=True, noise_ratio=0.3)
    test_folder.folder_pro(darker=True, dark_ratio=0.3)
    test_folder.folder_pro(brighter=True, bright_ratio=3.0)
    test_folder.folder_pro(blur=True, blur_size=3)
    test_folder.folder_pro(rotate=True, rotate_angle=30)
    test_folder.folder_pro(random_erasing=True, random_erasing_probability=0.8)
    