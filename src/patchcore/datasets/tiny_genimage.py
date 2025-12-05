# patchcore/datasets/tiny_genimage.py
import os, random
from enum import Enum
from tkinter import image_types
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
from torchvision import transforms as T
import torchvision.transforms.functional as TF


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
                # 改为随机集剪裁
                # T.Lambda(lambda im: random_crop_single(im, self.imagesize)),

                T.ToTensor(), # 将图像转换为 PyTorch 的张量（Tensor），并且会将图像的像素值从 [0, 255] 的范围转换到 [0.0, 1.0]
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        # 读取文件列表 + 标签
        self.imgpaths = self.get_image_data()

    def get_image_data(self) -> List[Tuple[str, str]]:
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

        imgpaths: List[Tuple[str, str]] = []

        for t in type_dirs:
            t_dir = os.path.join(split_dir, t)
            files = sorted(os.listdir(t_dir))
            for f in files:
                p = os.path.join(t_dir, f)
                imgpaths.append((t, p))
        print(f"[tiny_genimage] Loaded {len(imgpaths)} items from {split_dir} "
            f"(memorybank={self.bankname}, split={self.split.value})")
        return imgpaths

    def __len__(self) -> int:
        return len(self.imgpaths)

    def __getitem__(self, idx: int):
        t, image_path = self.imgpaths[idx]
        image_types = 'HSV'   # 修改图片格式
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
            "image_name": "/".join(image_path.replace("\\", "/").split("/")[-4:]),
            "image_path": image_path,
        }


def random_crop_single(image: Image.Image, crop_size: int) -> Image.Image:
    """
    随机裁剪出一块 crop_size x crop_size 的区域；
    若原图较小则先等比放大到短边>=crop_size，尽量避免无谓的 resize。
    """
    w, h = image.size
    if w < crop_size or h < crop_size:
        scale = crop_size / min(w, h)
        new_w = max(crop_size, int(round(w * scale)))
        new_h = max(crop_size, int(round(h * scale)))
        image = TF.resize(
            image,
            size=(new_h, new_w),
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        )
        w, h = image.size

    if w == crop_size and h == crop_size:
        return image

    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    return TF.crop(image, top, left, crop_size, crop_size)
