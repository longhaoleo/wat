import torch
from torch.utils.data import DataLoader
from . import tiny_genimage
import os
# 假设数据集路径已经提供

source = "~/datasets/tiny_genimage/sdv5"
source = os.path.expanduser(source)
# 创建数据集实例
dataset = tiny_genimage.Dataset(
    source=source,  # 数据集路径
    resize=256,     # 图像初始大小调整为256x256
    imagesize=224,  # 图像裁剪为224x224
    split=tiny_genimage.DatasetSplit.VAL,  # 使用训练集
    train_val_split=0.8  # 训练集和验证集按8:2划分
)

# 打印数据集的长度，确认数据是否加载
print(f"Dataset length: {len(dataset)}")

# 打印第一项数据，检查图像路径和转换
sample = dataset[0]
print(f"Sample image path: {sample['image_path']}")
print(f"Sample image shape: {sample['image'].shape}")  # 应该输出：torch.Size([3, 224, 224])

# 检查数据加载是否正常
batch_size = 4  # 设置批次大小
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 测试 DataLoader 是否正常工作
for i, batch in enumerate(dataloader):
    if i == 0:
        # 打印批次中的第一个图像信息
        print(f"Batch {i}:")
        print(f"Image path: {batch['image_path'][0]}")
        print(f"Image shape: {batch['image'][0].shape}")  # torch.Size([3, 224, 224])
    if i >= 2:  # 加载两个批次后结束
        break
