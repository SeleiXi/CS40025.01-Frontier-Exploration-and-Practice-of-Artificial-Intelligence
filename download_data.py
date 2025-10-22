#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据下载脚本
自动下载天池比赛数据集
"""

import os
import sys
import requests
import zipfile
from tqdm import tqdm
import pandas as pd


def download_file(url, filename, chunk_size=8192):
    """
    下载文件并显示进度条
    
    Args:
        url: 下载链接
        filename: 保存文件名
        chunk_size: 块大小
    """
    print(f"正在下载: {filename}")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"下载完成: {filename}")


def extract_zip(zip_path, extract_to):
    """
    解压zip文件
    
    Args:
        zip_path: zip文件路径
        extract_to: 解压目标目录
    """
    print(f"正在解压: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"解压完成: {extract_to}")


def download_dataset():
    """
    下载完整数据集
    """
    # 数据下载链接
    data_urls = {
        'train.zip': 'http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531872/%E5%9C%B0%E8%A1%A8%E5%BB%BA%E7%AD%91%E7%89%A9%E8%AF%86%E5%88%AB/train.zip',
        'train_mask.csv.zip': 'http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531872/%E5%9C%B0%E8%A1%A8%E5%BB%BA%E7%AD%91%E7%89%A9%E8%AF%86%E5%88%AB/train%20mask.csv.zip',
        'test_a.zip': 'http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531872/%E5%9C%B0%E8%A1%A8%E5%BB%BA%E7%AD%91%E7%89%A9%E8%AF%86%E5%88%AB/test%20a.zip',
        'test_a_samplesubmit.csv': 'http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531872/%E5%9C%B0%E8%A1%A8%E5%BB%BA%E7%AD%91%E7%89%A9%E8%AF%86%E5%88%AB/test%20a%20samplesubmit.csv'
    }
    
    # 创建数据目录
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    
    # 下载文件
    for filename, url in data_urls.items():
        file_path = os.path.join(data_dir, filename)
        
        # 如果文件已存在，跳过下载
        if os.path.exists(file_path):
            print(f"文件已存在，跳过下载: {filename}")
            continue
        
        try:
            download_file(url, file_path)
        except Exception as e:
            print(f"下载失败 {filename}: {e}")
            continue
    
    # 解压文件
    zip_files = ['train.zip', 'train_mask.csv.zip', 'test_a.zip']
    
    for zip_file in zip_files:
        zip_path = os.path.join(data_dir, zip_file)
        
        if os.path.exists(zip_path):
            if zip_file == 'train.zip':
                extract_to = data_dir
            elif zip_file == 'train_mask.csv.zip':
                extract_to = data_dir
            elif zip_file == 'test_a.zip':
                extract_to = data_dir
            
            try:
                extract_zip(zip_path, extract_to)
            except Exception as e:
                print(f"解压失败 {zip_file}: {e}")
                continue
    
    print("数据集下载和解压完成!")
    
    # 检查文件结构
    check_data_structure(data_dir)


def check_data_structure(data_dir):
    """
    检查数据目录结构
    
    Args:
        data_dir: 数据目录路径
    """
    print("\n检查数据目录结构...")
    
    # 检查训练集
    train_dir = os.path.join(data_dir, 'train')
    if os.path.exists(train_dir):
        train_images = [f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.png'))]
        print(f"训练集图像数量: {len(train_images)}")
    else:
        print("训练集目录不存在!")
    
    # 检查测试集
    test_dir = os.path.join(data_dir, 'test_a')
    if os.path.exists(test_dir):
        test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
        print(f"测试集图像数量: {len(test_images)}")
    else:
        print("测试集目录不存在!")
    
    # 检查标签文件
    mask_csv = os.path.join(data_dir, 'train_mask.csv')
    if os.path.exists(mask_csv):
        df = pd.read_csv(mask_csv)
        print(f"训练标签数量: {len(df)}")
        print(f"标签列: {df.columns.tolist()}")
    else:
        print("训练标签文件不存在!")
    
    # 检查提交样例
    sample_csv = os.path.join(data_dir, 'test_a_samplesubmit.csv')
    if os.path.exists(sample_csv):
        df = pd.read_csv(sample_csv)
        print(f"提交样例数量: {len(df)}")
        print(f"提交样例列: {df.columns.tolist()}")
    else:
        print("提交样例文件不存在!")


def create_sample_data():
    """
    创建示例数据（用于测试）
    """
    print("创建示例数据...")
    
    import numpy as np
    import cv2
    from data_utils import rle_encode
    
    # 创建示例目录
    sample_dir = './sample_data'
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(os.path.join(sample_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(sample_dir, 'test_a'), exist_ok=True)
    
    # 创建示例图像
    for i in range(10):
        # 创建随机图像
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        img_path = os.path.join(sample_dir, 'train', f'sample_{i:03d}.jpg')
        cv2.imwrite(img_path, img)
        
        # 创建随机掩码
        mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
        rle = rle_encode(mask)
        
        # 保存标签
        if i == 0:
            df = pd.DataFrame({'name': [f'sample_{i:03d}.jpg'], 'mask': [rle]})
        else:
            new_row = pd.DataFrame({'name': [f'sample_{i:03d}.jpg'], 'mask': [rle]})
            df = pd.concat([df, new_row], ignore_index=True)
    
    # 保存标签文件
    df.to_csv(os.path.join(sample_dir, 'train_mask.csv'), index=False)
    
    # 创建测试图像
    for i in range(5):
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        img_path = os.path.join(sample_dir, 'test_a', f'test_{i:03d}.jpg')
        cv2.imwrite(img_path, img)
    
    # 创建提交样例
    test_df = pd.DataFrame({
        'name': [f'test_{i:03d}.jpg' for i in range(5)],
        'mask': [''] * 5
    })
    test_df.to_csv(os.path.join(sample_dir, 'test_a_samplesubmit.csv'), index=False)
    
    print(f"示例数据已创建: {sample_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='下载天池比赛数据集')
    parser.add_argument('--sample', action='store_true', help='创建示例数据而不是下载真实数据')
    parser.add_argument('--data_dir', default='./data', help='数据保存目录')
    
    args = parser.parse_args()
    
    if args.sample:
        create_sample_data()
    else:
        download_dataset()
