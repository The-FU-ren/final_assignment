import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies, save_dir):
    """
    绘制训练曲线并保存为中文命名的图片
    
    参数:
        train_losses: 训练损失列表
        train_accuracies: 训练准确率列表
        val_losses: 验证损失列表
        val_accuracies: 验证准确率列表
        save_dir: 保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    # 1. 绘制损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='训练损失')
    plt.plot(epochs, val_losses, 'r-', linewidth=2, label='验证损失')
    plt.title('训练损失与验证损失曲线', fontsize=16)
    plt.xlabel('轮次', fontsize=14)
    plt.ylabel('损失值', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '损失曲线.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 绘制准确率曲线
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_accuracies, 'b-', linewidth=2, label='训练准确率')
    plt.plot(epochs, val_accuracies, 'r-', linewidth=2, label='验证准确率')
    plt.title('训练准确率与验证准确率曲线', fontsize=16)
    plt.xlabel('轮次', fontsize=14)
    plt.ylabel('准确率(%)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '准确率曲线.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 绘制合并曲线（上下布局）
    plt.figure(figsize=(12, 10))
    
    # 上半部分：损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='训练损失')
    plt.plot(epochs, val_losses, 'r-', linewidth=2, label='验证损失')
    plt.title('训练曲线', fontsize=16)
    plt.ylabel('损失值', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 下半部分：准确率曲线
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_accuracies, 'b-', linewidth=2, label='训练准确率')
    plt.plot(epochs, val_accuracies, 'r-', linewidth=2, label='验证准确率')
    plt.xlabel('轮次', fontsize=14)
    plt.ylabel('准确率(%)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '训练曲线合并图.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线已保存到: {save_dir}")

def save_training_data(train_losses, train_accuracies, val_losses, val_accuracies, save_dir):
    """
    保存训练数据为中文命名的文本文件
    
    参数:
        train_losses: 训练损失列表
        train_accuracies: 训练准确率列表
        val_losses: 验证损失列表
        val_accuracies: 验证准确率列表
        save_dir: 保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存为txt文件，使用UTF-8-BOM编码确保中文正确显示
    file_path = os.path.join(save_dir, '训练数据.txt')
    with open(file_path, 'w', encoding='utf-8-sig') as f:
        f.write('轮次\t训练损失\t训练准确率\t验证损失\t验证准确率\n')
        for i in range(len(train_losses)):
            f.write(f'{i+1}\t{train_losses[i]:.6f}\t{train_accuracies[i]:.4f}\t{val_losses[i]:.6f}\t{val_accuracies[i]:.4f}\n')
    
    # 保存为numpy文件（方便后续加载）
    np.savez(os.path.join(save_dir, '训练数据.npz'),
             train_losses=np.array(train_losses),
             train_accuracies=np.array(train_accuracies),
             val_losses=np.array(val_losses),
             val_accuracies=np.array(val_accuracies))
    
    print(f"训练数据已保存到: {file_path}")
