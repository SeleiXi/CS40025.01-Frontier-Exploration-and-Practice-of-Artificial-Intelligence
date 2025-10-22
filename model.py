#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型定义模块
包含FCN-ResNet50、损失函数、优化器等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss实现
    用于语义分割任务的区域级损失
    """
    
    def __init__(self, smooth=1., dims=(-2, -1)):
        super().__init__()
        # 平滑因子，防止分母为0导致的计算错误
        self.smooth = smooth
        # 对高度(H)和宽度(W)求和
        self.dims = dims
        
    def forward(self, x, y):
        # True Positive (TP): 预测和真实是1的区域
        t_p = (x * y).sum(self.dims)
        # False Positive (FP): 预测是1真实是0的区域
        f_p = (x * (1 - y)).sum(self.dims)
        # False Negative (FN): 预测是0真实是1的区域
        f_n = ((1 - x) * y).sum(self.dims)

        # Dice系数计算
        dc = (2 * t_p + self.smooth) / (2 * t_p + f_p + f_n + self.smooth)
        
        # 返回Dice损失 (1 - Dice Coefficient)
        return 1 - dc.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss实现
    用于处理类别不平衡问题
    """
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # 计算BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算概率
        pt = torch.exp(-bce_loss)
        
        # 计算Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    组合损失函数
    结合BCE、Dice和Focal Loss
    """
    
    def __init__(self, bce_weight=0.8, dice_weight=0.2, focal_weight=0.0, 
                 focal_alpha=1, focal_gamma=2):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.dice_fn = SoftDiceLoss()
        if focal_weight > 0:
            self.focal_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.focal_fn = None
    
    def forward(self, y_pred, y_true):
        # BCE损失
        bce_loss = self.bce_fn(y_pred, y_true)
        
        # Dice损失（需要sigmoid激活）
        dice_loss = self.dice_fn(y_pred.sigmoid(), y_true)
        
        # 总损失
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        # 如果使用Focal Loss
        if self.focal_fn is not None:
            focal_loss = self.focal_fn(y_pred, y_true)
            total_loss += self.focal_weight * focal_loss
        
        return total_loss


def get_model(model_name='fcn_resnet50', num_classes=1, pretrained=True):
    """
    获取分割模型
    
    Args:
        model_name: 模型名称
        num_classes: 输出类别数
        pretrained: 是否使用预训练权重
    
    Returns:
        torch.nn.Module: 分割模型
    """
    if model_name == 'fcn_resnet50':
        # FCN-ResNet50
        model = models.segmentation.fcn_resnet50(pretrained=pretrained)
        # 修改分类器为1个输出通道（二分类）
        model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
        
    elif model_name == 'fcn_resnet101':
        # FCN-ResNet101
        model = models.segmentation.fcn_resnet101(pretrained=pretrained)
        model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
        
    elif model_name == 'deeplabv3_resnet50':
        # DeepLabV3-ResNet50
        model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        
    elif model_name == 'deeplabv3_resnet101':
        # DeepLabV3-ResNet101
        model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained)
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    return model


class UNet(nn.Module):
    """
    UNet模型实现
    经典的编码器-解码器结构
    """
    
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 下采样路径
        for feature in features:
            self.downs.append(self._conv_block(in_channels, feature))
            in_channels = feature
        
        # 上采样路径
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(self._conv_block(feature * 2, feature))
        
        self.bottleneck = self._conv_block(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        skip_connections = []
        
        # 下采样
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # 上采样
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            
            if x.shape != skip_connection.shape:
                x = F.resize(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        
        return self.final_conv(x)


def get_loss_fn(loss_type='combined', **kwargs):
    """
    获取损失函数
    
    Args:
        loss_type: 损失函数类型
        **kwargs: 损失函数参数
    
    Returns:
        torch.nn.Module: 损失函数
    """
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'dice':
        return SoftDiceLoss()
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'combined':
        return CombinedLoss(**kwargs)
    else:
        raise ValueError(f"不支持的损失函数: {loss_type}")


def get_optimizer(model, optimizer_name='adam', lr=1e-4, weight_decay=1e-4, **kwargs):
    """
    获取优化器
    
    Args:
        model: 模型
        optimizer_name: 优化器名称
        lr: 学习率
        weight_decay: 权重衰减
        **kwargs: 其他参数
    
    Returns:
        torch.optim.Optimizer: 优化器
    """
    if optimizer_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")


def get_scheduler(optimizer, scheduler_name='cosine', **kwargs):
    """
    获取学习率调度器
    
    Args:
        optimizer: 优化器
        scheduler_name: 调度器名称
        **kwargs: 其他参数
    
    Returns:
        torch.optim.lr_scheduler: 学习率调度器
    """
    if scheduler_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_name == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_name == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f"不支持的学习率调度器: {scheduler_name}")


if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试FCN-ResNet50
    model = get_model('fcn_resnet50')
    model = model.to(device)
    
    # 测试输入
    x = torch.randn(2, 3, 512, 512).to(device)
    output = model(x)['out']
    print(f"FCN-ResNet50输出形状: {output.shape}")
    
    # 测试UNet
    unet = UNet().to(device)
    unet_output = unet(x)
    print(f"UNet输出形状: {unet_output.shape}")
    
    # 测试损失函数
    loss_fn = get_loss_fn('combined')
    target = torch.randint(0, 2, (2, 1, 512, 512)).float().to(device)
    loss = loss_fn(output, target)
    print(f"损失值: {loss.item()}")
