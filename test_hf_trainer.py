#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Hugging Face训练框架
"""

import os
import sys
import torch
import numpy as np

def test_hf_imports():
    """测试Hugging Face导入"""
    try:
        from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
        print("✅ Hugging Face transformers 导入成功")
        return True
    except ImportError as e:
        print(f"❌ Hugging Face transformers 导入失败: {e}")
        return False

def test_custom_trainer():
    """测试自定义训练器"""
    try:
        from train import CustomTrainer, TrainingCallback, create_hf_trainer
        print("✅ 自定义训练器导入成功")
        return True
    except ImportError as e:
        print(f"❌ 自定义训练器导入失败: {e}")
        return False

def test_training_args():
    """测试训练参数"""
    try:
        from transformers import TrainingArguments
        
        args = TrainingArguments(
            output_dir="./test_output",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            learning_rate=1e-4,
            logging_steps=1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            remove_unused_columns=False,
        )
        
        print("✅ TrainingArguments 创建成功")
        print(f"   - 输出目录: {args.output_dir}")
        print(f"   - 训练轮数: {args.num_train_epochs}")
        print(f"   - 批次大小: {args.per_device_train_batch_size}")
        print(f"   - 学习率: {args.learning_rate}")
        
        return True
    except Exception as e:
        print(f"❌ TrainingArguments 创建失败: {e}")
        return False

def test_callback():
    """测试回调函数"""
    try:
        from train import TrainingCallback
        
        # 创建模拟的自定义训练器
        class MockCustomTrainer:
            def __init__(self):
                self.train_losses = []
                self.val_losses = []
                self.val_scores = []
                self.best_score = 0.0
                self.best_epoch = 0
        
        mock_trainer = MockCustomTrainer()
        callback = TrainingCallback(mock_trainer)
        
        print("✅ TrainingCallback 创建成功")
        return True
    except Exception as e:
        print(f"❌ TrainingCallback 创建失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("Hugging Face 训练框架测试")
    print("=" * 60)
    
    tests = [
        ("Hugging Face 导入", test_hf_imports),
        ("自定义训练器", test_custom_trainer),
        ("训练参数", test_training_args),
        ("回调函数", test_callback),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n测试: {test_name}")
        print("-" * 40)
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} 测试失败")
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过! Hugging Face 训练框架准备就绪")
        return True
    else:
        print("⚠️  部分测试失败，请检查依赖安装")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
