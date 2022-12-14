# 环境配置

项目目录下的`requirements.txt`, 包含了实验运行需要的所有环境.

```bash
pip install -r requirements.txt
```

当然, 为了使用wandb来可视化结果, 也可以安装一下唉:

```bash
pip install wandb
```







# 数据集

## 数据集的获取

获取已划分好的数据集与初步训练好的结果请到以下地址下载

https://cloud.tsinghua.edu.cn/d/59e5b4c121e54e81a1e1/



## 获取原始数据集使用脚本生成

将原始数据集(raw_data.zip)解压缩, 重命名为raw_data, 放置于项目文件夹下.

此时项目文件架构为:

```
├── config
│   ├── baseline.py
│   ├── mynet_balanced.py
│   ├── mynet_balanced_transform.py
│   ├── mynet_inbalanced.py
│   ├── mynet_pro.py
│   ├── mynet_pro_batchnorm.py
│   ├── mynet_pro_batchnorm_adam.py
│   ├── mynet_pro_batchnorm_adam_transform.py
│   ├── resnet_pro.py
│   └── vit_pro.py
├── data_augmentation.py
├── data_loader.py
├── data_split.py
├── model.pys

├── pipeline.py
├── raw_data
│   ├── Corals
│   ├── Crabs
│   ├── ...
├── readme.md
├── report.pdf
├── requirements.txt
├── show_grad_cam.py
├── test.py
├── train.py
└── utils.py
```

运行`python data_split.py`生成训练集和测试集. 此时数据集位于项目文件夹的data文件夹下, 其中train文件夹下为训练集, test文件夹下为测试集, valid文件夹下为验证集.

此时文件架构为

```python
├── config
│   ├── baseline.py
│   ├── mynet_balanced.py
│   ├── mynet_balanced_transform.py
│   ├── mynet_inbalanced.py
│   ├── mynet_pro.py
│   ├── mynet_pro_batchnorm.py
│   ├── mynet_pro_batchnorm_adam.py
│   ├── mynet_pro_batchnorm_adam_transform.py
│   ├── resnet_pro.py
│   └── vit_pro.py
├── data
│   ├── test
│   ├── train
│   ├── valid
├── data_inbalanced
│   ├── test
│   ├── train
│   ├── valid
├── data_augmentation.py
├── data_loader.py
├── data_split.py
├── model.pys
├── pipeline.py
├── raw_data
│   ├── Corals
│   ├── Crabs
│   ├── ...
├── readme.md
├── report.pdf
├── requirements.txt
├── show_grad_cam.py
├── test.py
├── train.py
└── utils.py
```

运行`python data_augmentation.py`, 在data文件夹下生成`train_gauss_sp_blur_rotate_flip_random_erasing`文件夹, 包含增强后的数据集, 此时项目架构为

```
├── config
│   ├── baseline.py
│   ├── mynet_balanced.py
│   ├── mynet_balanced_transform.py
│   ├── mynet_inbalanced.py
│   ├── mynet_pro.py
│   ├── mynet_pro_batchnorm.py
│   ├── mynet_pro_batchnorm_adam.py
│   ├── mynet_pro_batchnorm_adam_transform.py
│   ├── resnet_pro.py
│   └── vit_pro.py
├── data
│   ├── test
│   ├── train
│   ├──	train_gauss_sp_blur_rotate_flip_random_erasing
│   ├── valid
├── data_inbalanced
│   ├── test
│   ├── train
│   ├── valid
├── data_augmentation.py
├── data_loader.py
├── data_split.py
├── model.pys
├── pipeline.py
├── raw_data
│   ├── Corals
│   ├── Crabs
│   ├── ...
├── readme.md
├── report.pdf
├── requirements.txt
├── show_grad_cam.py
├── test.py
├── train.py
└── utils.py
```





# 配置文件

配置文件位于`config`文件夹下, 共有:
- `mynet_inbalanced`. 使用原始数据集. 训练数据集位于data/train文件夹下
- `mynet_balanced`. 使用平衡各类后的数据集. 训练数据集位于data/train文件夹下
- `mynet_pro`. 使用手工定义函数进行数据增强, 训练数据集位于`data/train_gauss_sp_blur_rotate_flip_random_erasing`文件夹下.
- `mynet_balanced_transform`
- `mynet_pro_batchnorm`
- `mynet_pro_batchnorm_adam`
- `mynet_pro_batchnorm_adam_transform`
- `resnet_pro`
- `vit_pro`

点开看看配置文件, 就可以改参数跑自己的饰演了

# 训练

从零开始训练, 可以使用

```bash
python train.py -c 配置名
```

配置名即为`config`文件夹下配置的名字.

如想尝试使用了Batch Normalization与Adam优化器的使用了增强版数据集的`MyNet`, 可以使用

```bash
python train.py -c mynet_pro_batchnorm_adam
```

## 使用训练脚本

如果想训练所有配置, 可以直接使用

```bash
python pipeline.py
```

# 测试

## 使用已训练好的结果

在https://cloud.tsinghua.edu.cn/d/59e5b4c121e54e81a1e1/下载`saved_model.zip`, 将其放置在项目文件夹下, 此时项目的结构为

```
├── config
│   ├── baseline.py
│   ├── mynet_balanced.py
│   ├── mynet_balanced_transform.py
│   ├── mynet_inbalanced.py
│   ├── mynet_pro.py
│   ├── mynet_pro_batchnorm.py
│   ├── mynet_pro_batchnorm_adam.py
│   ├── mynet_pro_batchnorm_adam_transform.py
│   ├── resnet_pro.py
│   └── vit_pro.py
├── data
│   ├── test
│   ├── train
│   ├──	train_gauss_sp_blur_rotate_flip_random_erasing
│   ├── valid
├── data_augmentation.py
├── data_loader.py
├── data_split.py
├── model.pys
├── pipeline.py
├── raw_data
│   ├── Corals
│   ├── Crabs
│   ├── ...
├── readme.md
├── report.pdf
├── requirements.txt
├── saved_model
│   ├── mynet_balanced_60_0.001_64_optim_None_BN_False_TF_False
│   ├── mynet_balanced_60_0.001_64_optim_None_BN_False_TF_True
│   ├── mynet_inbalanced_60_0.001_64_optim_None_BN_False_TF_False
│   ├── mynet_pro_60_0.001_64_optim_Adam_BN_True_TF_False
│   ├── mynet_pro_60_0.001_64_optim_None_BN_False_TF_False
│   ├── mynet_pro_60_0.001_64_optim_None_BN_True_TF_False
│   ├── resnet50_pro_60_0.001_64_optim_Adam_BN_False_TF_False
│   ├── vit_pro_60_0.001_64_optim_Adam_BN_False_TF_False
├── show_grad_cam.py
├── test.py
├── train.py
└── utils.py
```

为了节约空间, `saved_model`下的文件夹只保留了最终训练的结果.

如果想测试某个配置运行的结果, 可以指定该模型对应的配置文件

```bash
python test.py -c mynet_pro_batchnorm_adam
```

## 从头训练(train from scratch)

从头训练的话, 训练过程会自动在`saved_model`下生成checkpoint.

如果想指定某一个checkpoint观察其效果的话, 可以用`-n`参数指定某一个checkpoint.

```bash
python test.py -c mynet_pro_batchnorm_adam -n 9
```

## Confusion Matrix 图像

在项目文件夹的`confusion_matrix`文件夹下, 会自动生成本次测试对应的混淆矩阵, 文件名即为本次实验的配置名.

#  可视化

使用`show_grad_cam.py`脚本可实现GradCAM可视化. 需指定想要可视化的图片路径, 并且需要指定输出图片名称(默认为`grad_cam.png`). 示例:

```bash
python show_grad_cam.py --image_name data/test/Penguin/21.jpg --save_name penguin.png
```

输出结果在`gradcam_result`文件夹下.