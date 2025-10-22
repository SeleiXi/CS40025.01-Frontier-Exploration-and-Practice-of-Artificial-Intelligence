#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练模块
包含训练、验证、测试等核心功能
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import get_model, get_loss_fn, get_optimizer, get_scheduler
from data_utils import TianChiDataset, rle_encode, rle_decode


class Trainer:
    """
    训练器类
    负责模型训练、验证和测试
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.DEVICE
        
        # 损失函数和优化器
        self.criterion = get_loss_fn('combined')
        self.optimizer = get_optimizer(
            model, 
            optimizer_name='adamw', 
            lr=config.LEARNING_RATE, 
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = get_scheduler(
            self.optimizer, 
            scheduler_name='cosine', 
            T_max=config.NUM_EPOCHS
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.val_scores = []
        self.best_score = 0.0
        self.best_epoch = 0
        
    def train_epoch(self, train_loader):
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
        
        Returns:
            float: 平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc="训练中")
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # 处理不同模型的输出格式
            if isinstance(outputs, dict):
                outputs = outputs['out']
            
            loss = self.criterion(outputs, masks)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader):
        """
        验证一个epoch
        
        Args:
            val_loader: 验证数据加载器
        
        Returns:
            tuple: (平均验证损失, Dice分数)
        """
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="验证中")
            for batch_idx, (images, masks) in enumerate(pbar):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    outputs = outputs['out']
                
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                
                # 计算Dice分数
                preds = torch.sigmoid(outputs)
                dice_score = self.calculate_dice(preds, masks)
                total_dice += dice_score
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice': f'{dice_score:.4f}'
                })
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        
        return avg_loss, avg_dice
    
    def calculate_dice(self, pred, target, smooth=1e-6):
        """
        计算Dice系数
        
        Args:
            pred: 预测结果 (B, 1, H, W)
            target: 真实标签 (B, 1, H, W)
            smooth: 平滑因子
        
        Returns:
            float: Dice系数
        """
        # 二值化预测结果
        pred = (pred > 0.5).float()
        
        # 计算交集和并集
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        # 计算Dice系数
        dice = (2.0 * intersection + smooth) / (union + smooth)
        
        return dice.mean().item()
    
    def train(self, train_loader, val_loader):
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
        """
        print("开始训练...")
        print(f"训练集大小: {len(train_loader.dataset)}")
        print(f"验证集大小: {len(val_loader.dataset)}")
        print(f"总epoch数: {self.config.NUM_EPOCHS}")
        
        start_time = time.time()
        
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            print("-" * 50)
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss, val_dice = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_scores.append(val_dice)
            
            # 学习率调度
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"训练损失: {train_loss:.4f}")
            print(f"验证损失: {val_loss:.4f}")
            print(f"验证Dice: {val_dice:.4f}")
            print(f"学习率: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_dice > self.best_score:
                self.best_score = val_dice
                self.best_epoch = epoch + 1
                self.save_model(f"best_model_epoch_{epoch + 1}.pth")
                print(f"新的最佳模型! Dice: {val_dice:.4f}")
            
            # 每10个epoch保存一次模型
            if (epoch + 1) % 10 == 0:
                self.save_model(f"model_epoch_{epoch + 1}.pth")
        
        # 训练完成
        training_time = time.time() - start_time
        print(f"\n训练完成! 总用时: {training_time:.2f}秒")
        print(f"最佳验证Dice: {self.best_score:.4f} (Epoch {self.best_epoch})")
        
        # 保存训练历史
        self.save_training_history()
        
        # 绘制训练曲线
        self.plot_training_curves()
    
    def save_model(self, filename):
        """
        保存模型
        
        Args:
            filename: 保存文件名
        """
        save_path = os.path.join(self.config.MODEL_SAVE_DIR, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': len(self.train_losses),
            'best_score': self.best_score,
            'config': self.config.__dict__
        }, save_path)
        print(f"模型已保存: {save_path}")
    
    def load_model(self, filename):
        """
        加载模型
        
        Args:
            filename: 模型文件名
        """
        load_path = os.path.join(self.config.MODEL_SAVE_DIR, filename)
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"模型已加载: {load_path}")
        return checkpoint
    
    def save_training_history(self):
        """
        保存训练历史
        """
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_scores': self.val_scores,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch
        }
        
        history_path = os.path.join(self.config.RESULT_SAVE_DIR, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"训练历史已保存: {history_path}")
    
    def plot_training_curves(self):
        """
        绘制训练曲线
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='训练损失', color='blue')
        ax1.plot(self.val_losses, label='验证损失', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.set_title('训练和验证损失')
        ax1.legend()
        ax1.grid(True)
        
        # Dice分数曲线
        ax2.plot(self.val_scores, label='验证Dice', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice分数')
        ax2.set_title('验证Dice分数')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # 保存图像
        plot_path = os.path.join(self.config.RESULT_SAVE_DIR, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"训练曲线已保存: {plot_path}")
    
    def predict(self, test_loader, model_path=None):
        """
        预测测试集
        
        Args:
            test_loader: 测试数据加载器
            model_path: 模型路径（可选）
        
        Returns:
            list: 预测结果列表
        """
        if model_path:
            self.load_model(model_path)
        
        self.model.eval()
        predictions = []
        
        print("开始预测...")
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="预测中")
            for images, _ in pbar:
                images = images.to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    outputs = outputs['out']
                
                # 应用sigmoid激活
                preds = torch.sigmoid(outputs)
                
                # 转换为numpy数组
                preds = preds.cpu().numpy()
                
                # 处理每个样本
                for pred in preds:
                    # 二值化
                    pred_binary = (pred[0] > 0.5).astype(np.uint8)
                    
                    # RLE编码
                    rle = rle_encode(pred_binary)
                    predictions.append(rle)
        
        print(f"预测完成! 共{len(predictions)}个样本")
        return predictions


def create_submission(predictions, test_paths, output_path):
    """
    创建提交文件
    
    Args:
        predictions: 预测结果列表
        test_paths: 测试图像路径列表
        output_path: 输出文件路径
    """
    # 提取文件名
    filenames = [os.path.basename(path) for path in test_paths]
    
    # 创建提交DataFrame
    submission_df = pd.DataFrame({
        'name': filenames,
        'mask': predictions
    })
    
    # 保存为CSV
    submission_df.to_csv(output_path, index=False)
    print(f"提交文件已保存: {output_path}")
    
    return submission_df


if __name__ == "__main__":
    # 测试训练器
    from main import config
    from data_utils import load_data, create_data_loaders
    
    # 加载数据
    train_paths, train_rles, test_paths = load_data(config.DATA_DIR, config.TRAIN_MASK_CSV)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        train_paths, train_rles, test_paths, config
    )
    
    # 创建模型
    model = get_model('fcn_resnet50')
    model = model.to(config.DEVICE)
    
    # 创建训练器
    trainer = Trainer(model, config)
    
    print("训练器创建成功!")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
