#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地表建筑物识别 - 语义分割项目
基于FCN-ResNet50的建筑物识别系统
"""

import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 配置参数
class Config:
    # 数据路径
    DATA_DIR = './data'
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    TEST_DIR = os.path.join(DATA_DIR, 'test_a')
    TRAIN_MASK_CSV = os.path.join(DATA_DIR, 'train_mask.csv')
    TEST_SAMPLE_CSV = os.path.join(DATA_DIR, 'test_a_samplesubmit.csv')
    
    # 模型参数
    IMAGE_SIZE = (512, 512)
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # 设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 保存路径
    MODEL_SAVE_DIR = './models'
    RESULT_SAVE_DIR = './results'
    
    def __init__(self):
        # 创建必要的目录
        os.makedirs(self.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(self.RESULT_SAVE_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)

config = Config()

def main():
    """
    主函数 - 完整的训练和预测流程
    """
    print("=" * 60)
    print("地表建筑物识别 - 语义分割项目")
    print("=" * 60)
    print(f"使用设备: {config.DEVICE}")
    print(f"图像尺寸: {config.IMAGE_SIZE}")
    print(f"批次大小: {config.BATCH_SIZE}")
    print(f"训练轮数: {config.NUM_EPOCHS}")
    print("=" * 60)
    
    # 检查数据是否存在
    if not os.path.exists(config.TRAIN_MASK_CSV):
        print("数据文件不存在，请先运行: python download_data.py")
        return
    
    # 导入训练相关模块
    from data_utils import load_data, create_data_loaders
    from model import get_model
    from train import Trainer, create_submission
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    train_paths, train_rles, test_paths = load_data(config.DATA_DIR, config.TRAIN_MASK_CSV)
    
    if len(train_paths) == 0:
        print("训练数据为空，请检查数据文件!")
        return
    
    # 2. 创建数据加载器
    print("\n2. 创建数据加载器...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_paths, train_rles, test_paths, config
    )
    
    # 3. 创建模型
    print("\n3. 创建模型...")
    model = get_model('fcn_resnet50', pretrained=True)
    model = model.to(config.DEVICE)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 4. 创建训练器
    print("\n4. 创建训练器...")
    trainer = Trainer(model, config)
    
    # 5. 开始训练
    print("\n5. 开始训练...")
    trainer.train(train_loader, val_loader)
    
    # 6. 预测测试集
    print("\n6. 预测测试集...")
    if len(test_paths) > 0:
        predictions = trainer.predict(test_loader, f"best_model_epoch_{trainer.best_epoch}.pth")
        
        # 7. 创建提交文件
        print("\n7. 创建提交文件...")
        submission_path = os.path.join(config.RESULT_SAVE_DIR, 'submission.csv')
        submission_df = create_submission(predictions, test_paths, submission_path)
        
        print(f"\n提交文件已创建: {submission_path}")
        print(f"预测样本数: {len(predictions)}")
    else:
        print("测试集为空，跳过预测步骤")
    
    print("\n项目完成!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='地表建筑物识别项目')
    parser.add_argument('--mode', choices=['train', 'predict', 'download'], 
                       default='train', help='运行模式')
    parser.add_argument('--model_path', type=str, help='模型路径（预测模式）')
    parser.add_argument('--download_sample', action='store_true', 
                       help='下载示例数据')
    
    args = parser.parse_args()
    
    if args.mode == 'download':
        if args.download_sample:
            from download_data import create_sample_data
            create_sample_data()
        else:
            from download_data import download_dataset
            download_dataset()
    elif args.mode == 'predict':
        if not args.model_path:
            print("预测模式需要指定模型路径: --model_path")
            sys.exit(1)
        
        # 预测模式
        from data_utils import load_data, create_data_loaders
        from model import get_model
        from train import Trainer, create_submission
        
        train_paths, train_rles, test_paths = load_data(config.DATA_DIR, config.TRAIN_MASK_CSV)
        _, _, test_loader = create_data_loaders(train_paths, train_rles, test_paths, config)
        
        model = get_model('fcn_resnet50', pretrained=True)
        model = model.to(config.DEVICE)
        
        trainer = Trainer(model, config)
        predictions = trainer.predict(test_loader, args.model_path)
        
        submission_path = os.path.join(config.RESULT_SAVE_DIR, 'submission.csv')
        create_submission(predictions, test_paths, submission_path)
        
    else:
        # 训练模式
        main()
