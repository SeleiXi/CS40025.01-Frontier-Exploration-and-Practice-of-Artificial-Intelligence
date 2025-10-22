#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目功能测试脚本
验证各个模块的基本功能
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from main import config
from data_utils import rle_encode, rle_decode, TianChiDataset
from model import get_model, get_loss_fn, SoftDiceLoss
from train import Trainer


def test_rle_functions():
    """测试RLE编解码功能"""
    print("测试RLE编解码功能...")
    
    # 创建测试掩码
    test_mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
    
    # 编码
    rle_str = rle_encode(test_mask)
    print(f"RLE编码长度: {len(rle_str)}")
    
    # 解码
    decoded_mask = rle_decode(rle_str, (512, 512))
    
    # 验证
    is_equal = np.array_equal(test_mask, decoded_mask)
    print(f"RLE编解码测试: {'通过' if is_equal else '失败'}")
    
    return is_equal


def test_model_creation():
    """测试模型创建"""
    print("\n测试模型创建...")
    
    try:
        # 创建模型
        model = get_model('fcn_resnet50', pretrained=False)
        model = model.to(config.DEVICE)
        
        # 测试前向传播
        x = torch.randn(2, 3, 512, 512).to(config.DEVICE)
        output = model(x)
        
        if isinstance(output, dict):
            output = output['out']
        
        print(f"模型输出形状: {output.shape}")
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"模型创建测试失败: {e}")
        return False


def test_loss_functions():
    """测试损失函数"""
    print("\n测试损失函数...")
    
    try:
        # 创建测试数据
        pred = torch.randn(2, 1, 512, 512).to(config.DEVICE)
        target = torch.randint(0, 2, (2, 1, 512, 512)).float().to(config.DEVICE)
        
        # 测试组合损失
        loss_fn = get_loss_fn('combined')
        loss = loss_fn(pred, target)
        print(f"组合损失: {loss.item():.4f}")
        
        # 测试Dice损失
        dice_fn = SoftDiceLoss()
        dice_loss = dice_fn(torch.sigmoid(pred), target)
        print(f"Dice损失: {dice_loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"损失函数测试失败: {e}")
        return False


def test_data_loading():
    """测试数据加载"""
    print("\n测试数据加载...")
    
    try:
        # 检查数据文件
        if not os.path.exists(config.TRAIN_MASK_CSV):
            print("数据文件不存在，跳过数据加载测试")
            return True
        
        # 加载数据
        from data_utils import load_data
        train_paths, train_rles, test_paths = load_data(config.DATA_DIR, config.TRAIN_MASK_CSV)
        
        print(f"训练数据: {len(train_paths)}")
        print(f"测试数据: {len(test_paths)}")
        
        if len(train_paths) > 0:
            # 测试数据集
            dataset = TianChiDataset(
                train_paths[:5], 
                train_rles[:5], 
                test_mode=False,
                image_size=config.IMAGE_SIZE
            )
            
            # 测试数据加载
            img, mask = dataset[0]
            print(f"图像形状: {img.shape}")
            print(f"掩码形状: {mask.shape}")
        
        return True
        
    except Exception as e:
        print(f"数据加载测试失败: {e}")
        return False


def test_training_setup():
    """测试训练设置"""
    print("\n测试训练设置...")
    
    try:
        # 创建模型
        model = get_model('fcn_resnet50', pretrained=False)
        model = model.to(config.DEVICE)
        
        # 创建训练器
        trainer = Trainer(model, config)
        
        print(f"优化器: {type(trainer.optimizer).__name__}")
        print(f"损失函数: {type(trainer.criterion).__name__}")
        print(f"学习率调度器: {type(trainer.scheduler).__name__}")
        
        return True
        
    except Exception as e:
        print(f"训练设置测试失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("开始项目功能测试")
    print("=" * 60)
    
    tests = [
        ("RLE编解码", test_rle_functions),
        ("模型创建", test_model_creation),
        ("损失函数", test_loss_functions),
        ("数据加载", test_data_loading),
        ("训练设置", test_training_setup),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"{test_name}测试异常: {e}")
            results.append((test_name, False))
    
    # 输出测试结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "通过" if result else "失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 个测试通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！项目可以正常运行。")
    else:
        print("⚠️  部分测试失败，请检查相关模块。")
    
    return passed == len(results)


if __name__ == "__main__":
    # 设置设备
    print(f"使用设备: {config.DEVICE}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    # 运行测试
    success = run_all_tests()
    
    if success:
        print("\n✅ 项目测试完成，可以开始训练！")
        print("\n使用方法:")
        print("1. 下载数据: python main.py --mode download --download_sample")
        print("2. 开始训练: python main.py --mode train")
        print("3. 模型预测: python main.py --mode predict --model_path models/best_model.pth")
    else:
        print("\n❌ 项目测试失败，请检查环境配置。")
        sys.exit(1)
