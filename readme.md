# 生成数据集

将原始数据集解压缩, 重命名为raw_data, 放置于项目文件夹下.

运行`python data_split.py`生成训练集和测试集. 此时数据集位于项目文件夹的data文件夹下, 其中train文件夹下为训练集, test文件夹下为测试集, valid文件夹下为验证集.
运行




# 配置文件
配置文件位于config文件夹下, 共有:
- mynet_inbalanced. 使用原始数据集. 训练数据集位于data/train文件夹下
- mynet_balanced. 使用平衡各类后的数据集. 训练数据集位于data/train文件夹下
- mynet_pro. 使用手工定义函数进行数据增强, 训练数据集位于data/train_gauss_sp_blur_rotate_flip_random_erasing文件夹下.
- mynet_balanced_transform
- mynet_pro_batchnorm
- mynet_pro_batchnorm_adam
- mynet_pro_batchnorm_adam_transform
- resnet_pro
- vit_pro