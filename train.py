#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Hugging Face训练框架的训练模块
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

# Hugging Face imports
from transformers import (
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from transformers.trainer_utils import get_last_checkpoint

from model import get_model, get_loss_fn, get_optimizer, get_scheduler
from data_utils import TianChiDataset, rle_encode, rle_decode


class CustomTrainer(Trainer):
    """
    基于Hugging Face的自定义训练器类
    负责模型训练、验证和测试
    """
    
    def __init__(self, model, config, train_dataset, eval_dataset=None, **kwargs):
        self.config = config
        self.device = config.DEVICE
        
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=config.MODEL_SAVE_DIR,
            num_train_epochs=config.NUM_EPOCHS,
            per_device_train_batch_size=config.BATCH_SIZE,
            per_device_eval_batch_size=config.BATCH_SIZE,
            learning_rate=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            warmup_steps=100,
            logging_dir=f"{config.MODEL_SAVE_DIR}/logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_dice",
            greater_is_better=True,
            report_to=None,  # 禁用wandb等
            dataloader_num_workers=4,
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False,
        )
        
        super().__init__(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.val_scores = []
        self.best_score = 0.0
        self.best_epoch = 0
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        计算损失函数
        
        Args:
            model: 模型
            inputs: 输入数据
            return_outputs: 是否返回输出
        
        Returns:
            loss或(loss, outputs)
        """
        images = inputs["images"]
        masks = inputs["masks"]
        
        # 前向传播
        outputs = model(images)
        
        # 处理不同模型的输出格式
        if isinstance(outputs, dict):
            outputs = outputs['out']
        
        # 计算损失
        criterion = get_loss_fn('combined')
        loss = criterion(outputs, masks)
        
        if return_outputs:
            return loss, outputs
        return loss
    
    def evaluate(self, eval_dataset=None):
        """
        评估模型性能
        
        Args:
            eval_dataset: 评估数据集
        
        Returns:
            dict: 评估指标
        """
        eval_dataset = eval_dataset or self.eval_dataset
        if eval_dataset is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in eval_dataset:
                if isinstance(batch, dict):
                    images = batch["images"]
                    masks = batch["masks"]
                else:
                    images, masks = batch
                
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    outputs = outputs['out']
                
                # 计算损失
                loss = self.compute_loss(self.model, {"images": images, "masks": masks})
                total_loss += loss.item()
                
                # 计算Dice分数
                preds = torch.sigmoid(outputs)
                dice_score = self.calculate_dice(preds, masks)
                total_dice += dice_score
                
                num_samples += images.size(0)
        
        avg_loss = total_loss / len(eval_dataset)
        avg_dice = total_dice / len(eval_dataset)
        
        return {
            "eval_loss": avg_loss,
            "eval_dice": avg_dice
        }
    
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
    
    def train_model(self):
        """
        使用Hugging Face框架进行训练
        """
        print("开始训练...")
        print(f"训练集大小: {len(self.train_dataset)}")
        if self.eval_dataset:
            print(f"验证集大小: {len(self.eval_dataset)}")
        print(f"总epoch数: {self.args.num_train_epochs}")
        
        start_time = time.time()
        
        # 添加回调函数
        callbacks = [
            EarlyStoppingCallback(early_stopping_patience=10),
            TrainingCallback(self)
        ]
        
        # 开始训练
        train_result = self.train(callbacks=callbacks)
        
        # 训练完成
        training_time = time.time() - start_time
        print(f"\n训练完成! 总用时: {training_time:.2f}秒")
        
        # 保存训练历史
        self.save_training_history()
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        return train_result
    
    def predict(self, test_dataset, model_path=None):
        """
        预测测试集
        
        Args:
            test_dataset: 测试数据集
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
            pbar = tqdm(test_dataset, desc="预测中")
            for batch in pbar:
                if isinstance(batch, dict):
                    images = batch["images"]
                else:
                    images, _ = batch
                
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


class TrainingCallback(TrainerCallback):
    """
    训练回调函数
    用于记录训练历史和保存最佳模型
    """
    
    def __init__(self, custom_trainer):
        self.custom_trainer = custom_trainer
    
    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        """评估完成后的回调"""
        if logs:
            eval_loss = logs.get("eval_loss", 0)
            eval_dice = logs.get("eval_dice", 0)
            
            self.custom_trainer.val_losses.append(eval_loss)
            self.custom_trainer.val_scores.append(eval_dice)
            
            if eval_dice > self.custom_trainer.best_score:
                self.custom_trainer.best_score = eval_dice
                self.custom_trainer.best_epoch = state.epoch
                print(f"新的最佳模型! Dice: {eval_dice:.4f}")
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """日志记录回调"""
        if logs and "train_loss" in logs:
            self.custom_trainer.train_losses.append(logs["train_loss"])
    
    def save_model(self, filename):
        """
        保存模型
        
        Args:
            filename: 保存文件名
        """
        save_path = os.path.join(self.config.MODEL_SAVE_DIR, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epoch': len(self.custom_trainer.train_losses),
            'best_score': self.custom_trainer.best_score,
            'config': self.custom_trainer.config.__dict__
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


def create_submission(predictions, test_paths, output_path):
    """
    创建提交文件
    
    Args:
        predictions: 预测结果列表
        test_paths: 测试图像路径列表
        output_path: 输出文件路径
    """
    import pandas as pd
    
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


def create_hf_trainer(model, config, train_dataset, eval_dataset=None):
    """
    创建Hugging Face训练器
    
    Args:
        model: 模型
        config: 配置对象
        train_dataset: 训练数据集
        eval_dataset: 验证数据集
    
    Returns:
        CustomTrainer: 自定义训练器
    """
    return CustomTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )


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
    
    # 创建Hugging Face训练器
    trainer = create_hf_trainer(model, config, train_loader.dataset, val_loader.dataset)
    
    print("Hugging Face训练器创建成功!")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
