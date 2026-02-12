# wat/datasets/tiny_genimage.py
import os, random
from enum import Enum
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
from torchvision import transforms as T


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL   = "val"
    TEST  = "test"  


class Dataset(TorchDataset):
    """
    Tiny-GenImage 风格数据集（支持 bank 相对标签）：
      - 训练集：仅使用 train/<bankname> 下的图像建立记忆库
      - 验证/测试：相对 bank 标签；属于 bankname→0，不属于→1
    返回：
      __getitem__ -> dict(image, is_ai, image_name, image_path)
      get_image_data() -> (imgpaths, labels_gt)
    """

    def __init__(
        self,
        source: str,
        resize: int = 256,
        imagesize: int = 224,
        split: DatasetSplit = DatasetSplit.TRAIN,
        bankname: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.source = source
        # 约定：generator 标签取数据集根目录名（split 的父目录），例如 .../tiny_genimage/sdv5/train/ai/*.png -> sdv5
        self.dataset_name = os.path.basename(os.path.normpath(self.source))
        self.resize = int(resize)
        self.imagesize = int(imagesize)
        self.split = split
        self.bankname = bankname
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.transform = T.Compose(
            [
                T.Resize(self.resize, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(self.imagesize),

                T.ToTensor(), # 将图像转换为 PyTorch 的张量（Tensor），并且会将图像的像素值从 [0, 255] 的范围转换到 [0.0, 1.0]
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        # 读取文件列表 + 标签
        self.imgpaths = self.get_image_data()

    def get_image_data(self) -> List[Tuple[str, str, str]]:
        """
        imgpaths: list of (type_name, abs_image_path)
        labels_gt:
            - 训练：保留 type_name（调试用，不参与监督，可存为字符串）
            - 验/测：相对 bank 标签，in-bank=0, out-of-bank=1
        """
        split_dir = os.path.join(self.source, self.split.value)
        if not os.path.isdir(split_dir):
            print(f"[tiny_genimage] Warning: split path not found: {split_dir}")
            return []

        # 所有子目录作为“类型”
        type_dirs = [
            d for d in sorted(os.listdir(split_dir))
            if os.path.isdir(os.path.join(split_dir, d))
        ]

        # 训练：只取指定 bank 的目录
        if self.split == DatasetSplit.TRAIN and self.bankname is not None:
            type_dirs = [self.bankname]

        imgpaths: List[Tuple[str, str, str]] = []

        for t in type_dirs:
            t_dir = os.path.join(split_dir, t)
            if not os.path.isdir(t_dir):
                continue

            # 支持一层生成器子目录：split/ai/<generator>/*.png
            subdirs = [
                d for d in sorted(os.listdir(t_dir))
                if os.path.isdir(os.path.join(t_dir, d))
            ]
            if subdirs:
                for gen in subdirs:
                    gen_dir = os.path.join(t_dir, gen)
                    for f in sorted(os.listdir(gen_dir)):
                        p = os.path.join(gen_dir, f)
                        imgpaths.append((t, gen, p))
            else:
                # 没有子目录则直接读文件：split/ai/*.png 或 split/nature/*.png
                for f in sorted(os.listdir(t_dir)):
                    p = os.path.join(t_dir, f)
                    # ai 的 generator 从数据集根目录名取；nature 统一为 nature
                    imgpaths.append((t, "nature" if t == "nature" else self.dataset_name, p))
        print(f"[tiny_genimage] Loaded {len(imgpaths)} items from {split_dir} "
            f"(memorybank={self.bankname}, split={self.split.value})")
        return imgpaths

    def __len__(self) -> int:
        return len(self.imgpaths)

    def __getitem__(self, idx: int):
        t, generator, image_path = self.imgpaths[idx]
        image_types = "RGB"
        try:
            # 读图更稳健一点
            with Image.open(image_path) as im:
                image = im.convert(image_types)  
        except (IOError, OSError) as e:
            # 如果图像无法加载，打印出错误信息并跳过
            print(f"Warning: Unable to load image {image_path}. Using default image.")
            image = Image.new(image_types, (self.imagesize, self.imagesize))  # 创建一个空白图像
    
        # 对图像进行转换
        image = self.transform(image)
        
        is_ai = 1 if t == "ai" else 0
        is_anomaly = 1 if (self.bankname is not None and t != self.bankname) else 0

        return {
            "image": image,
            "is_ai": is_ai,
            "is_anomaly": is_anomaly,
            "generator": generator if is_ai else "nature",
            "dataset_name": self.dataset_name,
            "image_name": "/".join(image_path.replace("\\", "/").split("/")[-4:]),
            "image_path": image_path,
        }
