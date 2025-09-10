# -*- coding: utf-8 -*-
"""
基于ViLD的开放世界室内物体检测 - 训练模块
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import gc
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip  
from torch.amp import autocast, GradScaler  # 更新为推荐的导入方式

from data_loader import ImprovedCOCOIndoorDataset, collate_fn
from config import TRAINING_CONFIG

class LossTracker:
    """损失追踪和可视化"""
    
    def __init__(self):
        self.train_losses = []
        self.epoch_losses = []
        self.best_loss = float('inf')
        self.best_epoch = 0
        
    def update(self, epoch_loss, epoch):
        """更新损失记录"""
        self.epoch_losses.append(epoch_loss)
        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.best_epoch = epoch
            
    def plot_losses(self, save_path=None, train_losses=None, val_losses=None, lr_history=None):
        """绘制损失曲线"""
        plt.figure(figsize=(15, 10))
        
        # 创建多子图
        gs = plt.GridSpec(2, 2, height_ratios=[2, 1])
        ax1 = plt.subplot(gs[0, :])
        ax2 = plt.subplot(gs[1, 0])
        ax3 = plt.subplot(gs[1, 1])
        
        # 主损失曲线
        epochs = range(1, len(self.epoch_losses) + 1)
        ax1.plot(epochs, self.epoch_losses, 'b-', linewidth=2.5, label='Validation Loss', marker='o')
        
        # 标注最佳损失点
        ax1.plot(self.best_epoch + 1, self.best_loss, 'r*', markersize=20, 
                label=f'Best Loss: {self.best_loss:.4f} (Epoch {self.best_epoch + 1})')
        
        # 移动平均线
        if len(self.epoch_losses) >= 3:
            window_size = min(3, len(self.epoch_losses))
            moving_avg = []
            for i in range(len(self.epoch_losses)):
                start_idx = max(0, i - window_size + 1)
                moving_avg.append(np.mean(self.epoch_losses[start_idx:i+1]))
            ax1.plot(epochs, moving_avg, 'g--', linewidth=2, alpha=0.7, label='Moving Average')
        
        # 设置样式
        ax1.set_xlabel('Epoch', fontsize=14)
        ax1.set_ylabel('Loss Value', fontsize=14)
        ax1.set_title('Validation Loss Curve', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=12, loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 训练/验证损失对比
        if train_losses and val_losses and len(train_losses) == len(val_losses):
            train_epochs = range(1, len(train_losses) + 1)
            ax2.plot(train_epochs, train_losses, 'b-', linewidth=2, label='Training')
            ax2.plot(train_epochs, val_losses, 'r-', linewidth=2, label='Validation')
            ax2.set_title('Training vs Validation Loss', fontsize=12)
            ax2.set_xlabel('Epoch', fontsize=10)
            ax2.set_ylabel('Loss', fontsize=10)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "训练/验证损失数据不可用", 
                    ha='center', va='center', transform=ax2.transAxes)
        
        # 学习率曲线
        if lr_history and len(lr_history) > 0:
            lr_epochs = range(1, len(lr_history) + 1)
            ax3.plot(lr_epochs, lr_history, 'g-', linewidth=2)
            ax3.set_title('Learning Rate Schedule', fontsize=12)
            ax3.set_xlabel('Epoch', fontsize=10)
            ax3.set_ylabel('Learning Rate', fontsize=10)
            ax3.grid(True, alpha=0.3)
            ax3.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
        else:
            ax3.text(0.5, 0.5, "学习率数据不可用", 
                    ha='center', va='center', transform=ax3.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Loss plot saved to: {save_path}")
        
        # 避免显示
        try:
            plt.close()
        except:
            pass

class StableTrainer:
    """稳定训练器"""
    
    def __init__(self, clip_model, device):
        self.clip_model = clip_model
        self.device = device
        
        # 创建投影器
        self.visual_projector = self.create_projector().to(device)
        self.text_projector = self.create_projector().to(device)
        
        # 使用恒等映射初始化
        self.initialize_as_identity()
        
        # 设置为训练模式
        self.visual_projector.train()
        self.text_projector.train()
        
        print("🎯 训练器初始化完成")
    
    def create_projector(self):
        """创建简化版投影器"""
        projector = nn.Sequential(
            nn.Linear(512, 512, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(512, 512, dtype=torch.float32)
        )
        
        # 确保使用float32
        for param in projector.parameters():
            param.data = param.data.float()
            
        return projector
    
    def initialize_as_identity(self):
        """初始化投影器为恒等映射"""
        with torch.no_grad():
            # 第一层
            torch.nn.init.eye_(self.visual_projector[0].weight)
            if self.visual_projector[0].bias is not None:
                torch.nn.init.zeros_(self.visual_projector[0].bias)
            
            torch.nn.init.eye_(self.text_projector[0].weight)
            if self.text_projector[0].bias is not None:
                torch.nn.init.zeros_(self.text_projector[0].bias)
            
            # 最后一层
            torch.nn.init.eye_(self.visual_projector[2].weight)
            if self.visual_projector[2].bias is not None:
                torch.nn.init.zeros_(self.visual_projector[2].bias)
                
            torch.nn.init.eye_(self.text_projector[2].weight)
            if self.text_projector[2].bias is not None:
                torch.nn.init.zeros_(self.text_projector[2].bias)
    
    def get_trainable_parameters(self):
        """获取可训练参数"""
        params = []
        params.extend(self.visual_projector.parameters())
        params.extend(self.text_projector.parameters())
        return params
    
    def compute_distillation_loss(self, visual_features, text_features, temperature=0.05):
        """计算知识蒸馏损失"""
        # L2归一化
        visual_features = F.normalize(visual_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.mm(visual_features, text_features.t()) / temperature
        
        # 对角线损失（自相似）
        batch_size = visual_features.size(0)
        targets = torch.arange(batch_size).to(self.device)
        
        # 如果文本特征数量不够，使用循环索引
        if text_features.size(0) < batch_size:
            targets = targets % text_features.size(0)
        
        # 带有标签平滑的交叉熵损失
        label_smoothing = 0.2
        loss_v2t = F.cross_entropy(similarity_matrix, targets, label_smoothing=label_smoothing)
        loss_t2v = F.cross_entropy(similarity_matrix.t(), targets[:text_features.size(0)], label_smoothing=label_smoothing)
        
        # 特征对齐损失
        alignment_loss = torch.diagonal(1 - similarity_matrix).mean()
        
        # 特征均匀性损失
        uniformity_loss = torch.log(torch.exp(torch.mm(visual_features, visual_features.t()) / temperature).mean())
        
        # 添加L2正则化
        l2_reg = 0.0005 * (
            torch.norm(self.visual_projector[0].weight, p=2) +
            torch.norm(self.text_projector[0].weight, p=2)
        )
        
        # 总损失
        total_loss = 0.3 * (loss_v2t + loss_t2v) / 2 + \
                    0.3 * alignment_loss + \
                    0.3 * uniformity_loss + \
                    0.1 * l2_reg
        
        return total_loss
    
    def encode_text_features_batch(self, categories, batch_size):
        """为每个批次编码文本特征"""
        all_text_features = []
        templates = ["a {}", "indoor {}", "a {} in a room"]
        
        for category in categories:
            category_features = []
            
            for template in templates:
                text = template.format(category)
                text_tokens = clip.tokenize([text]).to(self.device)
                
                with torch.no_grad():
                    text_features = self.clip_model.encode_text(text_tokens).float()
                
                # 应用文本投影器
                projected_text = self.text_projector(text_features)
                category_features.append(projected_text)
            
            # 平均多个模板的特征
            if category_features:
                avg_features = torch.stack(category_features).mean(dim=0)
                all_text_features.append(avg_features)
        
        if all_text_features:
            text_features = torch.cat(all_text_features, dim=0)
            
            # 随机选择文本特征（匹配batch size）
            if text_features.size(0) >= batch_size:
                selected_indices = torch.randperm(text_features.size(0))[:batch_size]
                selected_text_features = text_features[selected_indices]
            else:
                # 重复文本特征
                repeat_times = (batch_size + text_features.size(0) - 1) // text_features.size(0)
                repeated_text = text_features.repeat(repeat_times, 1)
                selected_text_features = repeated_text[:batch_size]
            
            return selected_text_features
        else:
            return torch.empty(batch_size, 512, dtype=torch.float32).to(self.device)
    
    def validate(self, dataloader):
        """在验证集上评估模型"""
        self.visual_projector.eval()
        self.text_projector.eval()
        
        val_losses = []
        
        # 室内类别
        indoor_categories = [
            "chair", "table", "bed", "sofa", "cabinet", "toilet", "sink",
            "refrigerator", "microwave", "bottle", "cup", "bowl",
            "lamp", "clock", "vase", "plant", "computer", "bookshelf"
        ]
        
        with torch.no_grad():
            for batch in dataloader:
                try:
                    # 获取图像
                    images = batch['images'].to(self.device)
                    batch_size = images.size(0)
                    
                    # 提取视觉特征
                    visual_features = []
                    for i in range(batch_size):
                        # 使用CLIP编码整个图像
                        image_features = self.clip_model.encode_image(images[i:i+1]).float()
                        
                        # 应用投影器
                        projected_features = self.visual_projector(image_features)
                        visual_features.append(projected_features)
                    
                    visual_features = torch.cat(visual_features, dim=0)
                    
                    # 编码文本特征
                    text_features = self.encode_text_features_batch(indoor_categories, batch_size)
                    
                    # 计算损失
                    loss = self.compute_distillation_loss(visual_features, text_features)
                    
                    # 记录损失
                    val_losses.append(loss.item())
                    
                except Exception as e:
                    print(f"⚠️ 验证批次处理失败: {e}")
                    continue
        
        avg_loss = np.mean(val_losses) if val_losses else float('inf')
        return avg_loss
    
    def train_epoch(self, dataloader, optimizer, scheduler=None, scaler=None, gradient_accumulation_steps=1):
        """训练一个epoch，支持混合精度和梯度累积"""
        self.visual_projector.train()
        self.text_projector.train()
        
        # 确保scheduler不为None时类型正确
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
            print("⚠️ 警告: 提供的调度器类型不正确，已禁用")
            scheduler = None
        
        epoch_losses = []
        use_amp = scaler is not None
        
        # 室内类别
        indoor_categories = [
            "chair", "table", "bed", "sofa", "cabinet", "toilet", "sink",
            "refrigerator", "microwave", "bottle", "cup", "bowl",
            "lamp", "clock", "vase", "plant", "computer", "bookshelf"
        ]
        
        with tqdm(total=len(dataloader), desc="🚀 训练进行中") as pbar:
            for batch_idx, batch in enumerate(dataloader):
                try:
                    # 仅在累积的第一步清零梯度
                    if (batch_idx % gradient_accumulation_steps) == 0:
                        optimizer.zero_grad()
                    
                    # 获取图像
                    images = batch['images'].to(self.device)
                    batch_size = images.size(0)
                    
                    # 使用自动混合精度
                    with autocast('cuda', enabled=use_amp):
                        # 批量处理视觉特征（更高效）
                        try:
                            with torch.no_grad():
                                # 直接批量编码所有图像
                                image_features = self.clip_model.encode_image(images).float()
                            
                            # 应用视觉投影器
                            visual_features = self.visual_projector(image_features)
                        except RuntimeError as e:
                            # 如果批处理失败，回退到单图像处理
                            if 'out of memory' in str(e):
                                torch.cuda.empty_cache()
                                print(f"⚠️ 批量处理内存不足，回退到单图像处理")
                                visual_features = []
                                for i in range(batch_size):
                                    with torch.no_grad():
                                        image_features = self.clip_model.encode_image(images[i:i+1]).float()
                                    projected_features = self.visual_projector(image_features)
                                    visual_features.append(projected_features)
                                visual_features = torch.cat(visual_features, dim=0)
                            else:
                                raise e
                        
                        # 文本特征
                        text_features = self.encode_text_features_batch(indoor_categories, batch_size)
                        
                        # 计算损失
                        loss = self.compute_distillation_loss(visual_features, text_features)
                        
                        # 检测异常损失值
                        if not torch.isfinite(loss):
                            print(f"⚠️ 警告: 损失值无效 {loss.item()}, 跳过此批次")
                            continue
                        
                        # 根据梯度累积调整损失
                        loss = loss / gradient_accumulation_steps
                    
                    # 使用梯度缩放器进行反向传播
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    # 仅在累积完成后更新参数
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        # 梯度裁剪
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        
                        torch.nn.utils.clip_grad_norm_(self.get_trainable_parameters(), max_norm=1.0)
                        
                        # 使用scaler更新权重
                        if scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        
                        # 这里不再调用scheduler.step()，统一在epoch结束后处理
                        # 这样可以确保optimizer.step()总是在scheduler.step()之前被调用
                    
                    # 记录损失
                    epoch_losses.append(loss.item() * gradient_accumulation_steps)
                    
                    # 更新进度条
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'avg_loss': f"{np.mean(epoch_losses):.4f}",
                        'lr': f"{current_lr:.2e}"
                    })
                    pbar.update(1)
                    
                    # 清理中间变量
                    del visual_features, text_features, loss
                    
                except Exception as e:
                    print(f"⚠️ 批次 {batch_idx} 处理失败: {e}")
                    continue
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        return avg_loss

def run_fixed_training(clip_model, device, images, image_root):
    """运行训练"""
    print("🚀 开始优化版ViLD训练 (RTX 4090优化版)")
    
    try:
        # 创建训练器
        trainer = StableTrainer(
            clip_model=clip_model,
            device=device
        )
        
        # 限制最大样本数以避免内存问题
        max_samples = TRAINING_CONFIG.get('max_samples', 20000)
        if len(images) > max_samples:
            print(f"⚙️ 限制训练样本数量为 {max_samples}（原始: {len(images)}）")
            images = images[:max_samples]
        
        # 创建训练数据集
        dataset = ImprovedCOCOIndoorDataset(
            images_data=images,
            image_root=image_root,
            image_size=TRAINING_CONFIG.get('image_size', 224),
            augment=TRAINING_CONFIG.get('augment', True)
        )
        
        if len(dataset) == 0:
            print("❌ 数据集为空")
            return False
        
        # 创建验证数据集 - 使用10%的数据
        val_size = int(len(dataset) * 0.1)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        # 从配置中获取参数
        max_epochs = TRAINING_CONFIG.get('max_epochs', 25)
        batch_size = TRAINING_CONFIG.get('batch_size', 64)  # 增大批量大小
        num_workers = TRAINING_CONFIG.get('num_workers', 4)  # 增加数据加载线程
        pin_memory = TRAINING_CONFIG.get('pin_memory', True)
        
        # 打印GPU信息
        if torch.cuda.is_available():
            print(f"\n💻 GPU信息:")
            print(f"   GPU设备: {torch.cuda.get_device_name(0)}")
            print(f"   内存分配: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"   内存缓存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
            print(f"   最大内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # 优化的数据加载器
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False,  # 保持worker进程活跃
            prefetch_factor=2 if num_workers > 0 else None,  # 预加载2个batch
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            persistent_workers=True if num_workers > 0 else False
        )
        
        # 优化器配置
        learning_rate = TRAINING_CONFIG.get('learning_rate', 2e-5)  # 更高的学习率
        weight_decay = TRAINING_CONFIG.get('weight_decay', 5e-5)
        
        # 优化器
        optimizer = torch.optim.AdamW(
            trainer.get_trainable_parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),  # 标准AdamW参数
            eps=1e-8,
            amsgrad=True
        )
        
        # 学习率调度器 - 使用OneCycleLR以获得更好的收敛
        steps_per_epoch = len(train_dataloader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            steps_per_epoch=steps_per_epoch,
            epochs=max_epochs,
            pct_start=0.1,  # 预热10%的训练时间
            anneal_strategy='cos',
            div_factor=10.0,  # 初始学习率 = max_lr / 10
            final_div_factor=100.0,  # 最终学习率 = max_lr / 1000
            last_epoch=-1  # 显式设置初始化状态，避免初始化时自动调用step()
        )
        
        # 创建梯度缩放器用于混合精度训练
        use_amp = TRAINING_CONFIG.get('use_amp', True)
        scaler = GradScaler(enabled=use_amp)
        
        # 梯度累积步数
        gradient_accumulation_steps = TRAINING_CONFIG.get('gradient_accumulation_steps', 1)
        
        # 损失追踪器
        loss_tracker = LossTracker()
        
        # 打印详细的训练配置
        print(f"📊 训练配置 (RTX 4090 优化):")
        print(f"   数据集大小: {len(dataset)}")
        print(f"   批次大小: {batch_size}")
        print(f"   最大epoch数: {max_epochs}")
        print(f"   学习率: {learning_rate}")
        print(f"   数据加载线程: {num_workers}")
        print(f"   梯度累积步数: {gradient_accumulation_steps}")
        print(f"   自动混合精度: {'启用' if use_amp else '禁用'}")
        print(f"   有效批大小: {batch_size * gradient_accumulation_steps}")
        
        # 训练统计
        train_losses = []
        val_losses = []
        lr_history = []
        
        # 最佳模型信息
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        patience = 5
        
        # 创建检查点目录
        checkpoint_dir = '/home/cui/rtdetr_indoor/src/vild/checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 开始训练
        for epoch in range(max_epochs):
            print(f"\n{'='*50}")
            print(f"🔄 Epoch {epoch + 1}/{max_epochs}")
            print(f"{'='*50}")
            
            # 在训练前禁用学习率调度器的批次级更新，后面手动统一处理
            # 这样可以确保先执行optimizer.step()再执行scheduler.step()
            train_loss = trainer.train_epoch(
                train_dataloader, 
                optimizer,
                scheduler=None,  # 明确设置为None，避免批次级更新
                scaler=scaler if use_amp else None,
                gradient_accumulation_steps=gradient_accumulation_steps
            )
            train_losses.append(train_loss)
            
            # 验证
            print(f"\n📊 运行验证...")
            val_loss = trainer.validate(val_dataloader)
            val_losses.append(val_loss)
            
            # 统一在这里处理所有学习率调度更新
            # 确保在optimizer.step()后调用scheduler.step()
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # 这类调度器需要根据验证损失调整学习率
                    scheduler.step(val_loss)
                elif isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    # OneCycleLR需要在每个批次后更新，这里进行一个epoch的步数更新
                    for _ in range(len(train_dataloader)):
                        scheduler.step()
                else:
                    # 其他调度器只需要每轮更新一次
                    scheduler.step()
            
            # 记录学习率
            current_lr = optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)
            
            # 更新损失追踪器
            loss_tracker.update(val_loss, epoch)
            
            print(f"📈 Epoch {epoch+1} 结果:")
            print(f"   训练损失: {train_loss:.6f}")
            print(f"   验证损失: {val_loss:.6f}")
            print(f"   学习率: {current_lr:.8f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                
                # 保存模型
                best_model_path = f'{checkpoint_dir}/best_model.pth'
                checkpoint = {
                    'epoch': epoch,
                    'visual_projector': trainer.visual_projector.state_dict(),
                    'text_projector': trainer.text_projector.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_loss': val_loss
                }
                
                torch.save(checkpoint, best_model_path)
                print(f"💾 保存最佳模型: 验证损失={val_loss:.6f} (第{epoch+1}轮)")
            else:
                patience_counter += 1
                print(f"⚠️ 验证损失未改善，当前耐心: {patience_counter}/{patience}")
            
            # 早停检查
            if patience_counter >= patience:
                print(f"\n⏹️ 早停触发! 连续{patience}个epoch无改善")
                break
            
            # 内存清理
            torch.cuda.empty_cache()
            gc.collect()
        
        # 训练完成后绘图
        final_loss_path = f'{checkpoint_dir}/training_loss.png'
        loss_tracker.plot_losses(
            save_path=final_loss_path,
            train_losses=train_losses,
            val_losses=val_losses,
            lr_history=lr_history
        )
        
        print(f"\n🎉 训练完成!")
        print(f"📈 最终成果:")
        print(f"   最佳验证损失: {best_val_loss:.6f}")
        print(f"   最佳epoch: {best_epoch + 1}")
        print(f"   损失图已保存: {final_loss_path}")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"❌ 训练失败: {e}")
        traceback.print_exc()
        return False
