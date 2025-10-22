#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理工具模块
包含RLE编解码、数据集类、数据增强等功能
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2


def rle_encode(im):
    """
    RLE编码：将二值图像编码为RLE格式字符串
    
    Args:
        im: 二值图像数组 (numpy.ndarray)
    
    Returns:
        str: RLE编码字符串
    """
    # 将二值图像展平，order='F' (Fortran order/列优先)
    pixels = im.flatten(order='F')
    # 在首尾添加0作为边界
    pixels = np.concatenate([[0], pixels, [0]])
    # 找到所有值发生变化的位置
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # runs中的数字用空格连接成字符串
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(512, 512)):
    """
    RLE解码：将RLE格式字符串解码为二值图像
    
    Args:
        mask_rle: RLE编码字符串
        shape: 输出图像形状
    
    Returns:
        numpy.ndarray: 二值图像数组
    """
    # 如果是空字符串，表示图像没有建筑物，返回全0的掩码图
    if not mask_rle:
        return np.zeros(shape, dtype=np.uint8)
    
    # 字符串按空格拆分为列表
    s = mask_rle.split()
    # 将字符串列表转换为整型数组，奇数项为起始位置，偶数项为长度
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    
    # 起始位置调整（基于1开始）
    starts = starts - 1
    # 计算结束位置
    ends = starts + lengths
    
    # 生成标记图像
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)  # 1D数组
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1  # 将指定范围内的像素设置为1
        
    # 重塑为shape对应的二维数组，order='F'
    return img.reshape(shape, order='F')


class TianChiDataset(Dataset):
    """
    天池数据集类
    支持训练和测试模式
    """
    
    def __init__(self, paths, rles=None, transform=None, test_mode=False, image_size=(512, 512)):
        """
        初始化数据集
        
        Args:
            paths: 图像路径列表
            rles: RLE标签列表（训练时使用）
            transform: 数据增强方法
            test_mode: 是否为测试模式
            image_size: 图像尺寸
        """
        self.paths = paths
        self.rles = rles if not test_mode else []
        self.transform = transform
        self.test_mode = test_mode
        self.image_size = image_size
        
        # 定义ToTensor和标准化操作
        self.to_tensor = T.Compose([
            T.ToPILImage(),
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize([0.625, 0.448, 0.688], [0.131, 0.177, 0.101]),
        ])
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, index):
        # 读取图像并转换RGB
        img = cv2.imread(self.paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if not self.test_mode:
            # 训练模式: 加载mask并进行增强
            mask = rle_decode(self.rles[index], self.image_size)
            if self.transform:
                augmented = self.transform(image=img, mask=mask)
                img, mask = augmented['image'], augmented['mask']
            return self.to_tensor(img), torch.from_numpy(mask[None]).float()
        else:
            # 测试模式: 仅返回图像
            return self.to_tensor(img),


def get_train_transforms():
    """
    获取训练时的数据增强
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Blur(blur_limit=3, p=0.3),
    ])


def get_val_transforms():
    """
    获取验证时的数据增强（无增强）
    """
    return None


def load_data(data_dir, train_mask_csv):
    """
    加载数据集
    
    Args:
        data_dir: 数据目录
        train_mask_csv: 训练标签CSV文件路径
    
    Returns:
        tuple: (train_paths, train_rles, test_paths)
    """
    # 加载训练数据
    train_mask_df = pd.read_csv(train_mask_csv)
    train_dir = os.path.join(data_dir, 'train')
    
    train_paths = []
    train_rles = []
    
    for idx, row in train_mask_df.iterrows():
        img_path = os.path.join(train_dir, row['name'])
        if os.path.exists(img_path):
            train_paths.append(img_path)
            train_rles.append(row['mask'])
    
    print(f"训练集大小: {len(train_paths)}")
    
    # 加载测试数据
    test_dir = os.path.join(data_dir, 'test_a')
    test_paths = []
    
    if os.path.exists(test_dir):
        for filename in os.listdir(test_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                test_paths.append(os.path.join(test_dir, filename))
    
    print(f"测试集大小: {len(test_paths)}")
    
    return train_paths, train_rles, test_paths


def create_data_loaders(train_paths, train_rles, test_paths, config):
    """
    创建数据加载器
    
    Args:
        train_paths: 训练图像路径
        train_rles: 训练标签
        test_paths: 测试图像路径
        config: 配置对象
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 划分训练集和验证集
    train_paths_split, val_paths_split, train_rles_split, val_rles_split = train_test_split(
        train_paths, train_rles, test_size=0.2, random_state=42
    )
    
    # 创建数据集
    train_dataset = TianChiDataset(
        train_paths_split, 
        train_rles_split, 
        transform=get_train_transforms(),
        test_mode=False,
        image_size=config.IMAGE_SIZE
    )
    
    val_dataset = TianChiDataset(
        val_paths_split, 
        val_rles_split, 
        transform=get_val_transforms(),
        test_mode=False,
        image_size=config.IMAGE_SIZE
    )
    
    test_dataset = TianChiDataset(
        test_paths, 
        transform=get_val_transforms(),
        test_mode=True,
        image_size=config.IMAGE_SIZE
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试RLE编解码
    test_mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
    rle_str = rle_encode(test_mask)
    decoded_mask = rle_decode(rle_str, (512, 512))
    print(f"RLE编解码测试: {np.array_equal(test_mask, decoded_mask)}")
