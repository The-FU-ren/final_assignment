import matplotlib.pyplot as plt
import numpy as np

# 生成理想的训练曲线数据
def generate_ideal_curve(num_epochs=20):
    # Softmax分类器 - 简单线性模型
    softmax_train_loss = np.exp(-np.linspace(0, 3, num_epochs)) + 0.1 * np.random.rand(num_epochs) * np.exp(-np.linspace(0, 2, num_epochs))
    softmax_val_loss = softmax_train_loss + 0.05 + 0.1 * np.random.rand(num_epochs) * np.exp(-np.linspace(0, 1.5, num_epochs))
    softmax_train_acc = 60 + 30 * (1 - np.exp(-np.linspace(0, 3, num_epochs))) + 2 * np.random.rand(num_epochs)
    softmax_val_acc = softmax_train_acc - 5 - 2 * np.random.rand(num_epochs) * np.exp(-np.linspace(0, 1.5, num_epochs))
    
    # 全连接神经网络 - Adam优化器
    fc_adam_train_loss = np.exp(-np.linspace(0, 3.5, num_epochs)) + 0.05 * np.random.rand(num_epochs) * np.exp(-np.linspace(0, 2.5, num_epochs))
    fc_adam_val_loss = fc_adam_train_loss + 0.03 + 0.08 * np.random.rand(num_epochs) * np.exp(-np.linspace(0, 2, num_epochs))
    fc_adam_train_acc = 70 + 28 * (1 - np.exp(-np.linspace(0, 3.5, num_epochs))) + 1.5 * np.random.rand(num_epochs)
    fc_adam_val_acc = fc_adam_train_acc - 3 - 1.5 * np.random.rand(num_epochs) * np.exp(-np.linspace(0, 2, num_epochs))
    
    # 全连接神经网络 - SGD优化器
    fc_sgd_train_loss = np.exp(-np.linspace(0, 2.5, num_epochs)) + 0.15 * np.random.rand(num_epochs) * np.exp(-np.linspace(0, 1.5, num_epochs))
    fc_sgd_val_loss = fc_sgd_train_loss + 0.07 + 0.12 * np.random.rand(num_epochs) * np.exp(-np.linspace(0, 1.2, num_epochs))
    fc_sgd_train_acc = 65 + 32 * (1 - np.exp(-np.linspace(0, 2.5, num_epochs))) + 2.5 * np.random.rand(num_epochs)
    fc_sgd_val_acc = fc_sgd_train_acc - 4 - 2 * np.random.rand(num_epochs) * np.exp(-np.linspace(0, 1.2, num_epochs))
    
    return {
        'softmax': {
            'train_loss': softmax_train_loss,
            'val_loss': softmax_val_loss,
            'train_acc': softmax_train_acc,
            'val_acc': softmax_val_acc
        },
        'fc_adam': {
            'train_loss': fc_adam_train_loss,
            'val_loss': fc_adam_val_loss,
            'train_acc': fc_adam_train_acc,
            'val_acc': fc_adam_val_acc
        },
        'fc_sgd': {
            'train_loss': fc_sgd_train_loss,
            'val_loss': fc_sgd_val_loss,
            'train_acc': fc_sgd_train_acc,
            'val_acc': fc_sgd_val_acc
        }
    }

# 绘制理想的学习曲线
def plot_learning_curve(results, model_name, title, save_dir='ideal_curves'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    epochs = range(1, len(results['train_loss']) + 1)
    
    # 绘制损失曲线
    ax1.plot(epochs, results['train_loss'], label='Train Loss', linewidth=2, color='blue')
    ax1.plot(epochs, results['val_loss'], label='Validation Loss', linewidth=2, color='red')
    ax1.set_title(f'{title} - Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(bottom=0)
    
    # 绘制准确率曲线
    ax2.plot(epochs, results['train_acc'], label='Train Accuracy', linewidth=2, color='blue')
    ax2.plot(epochs, results['val_acc'], label='Validation Accuracy', linewidth=2, color='red')
    ax2.set_title(f'{title} - Accuracy Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name.replace(" ", "_")}_ideal_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

# 主函数
def main():
    num_epochs = 20
    
    # 生成理想曲线数据
    ideal_results = generate_ideal_curve(num_epochs)
    
    # 绘制各个模型的理想学习曲线
    plot_learning_curve(ideal_results['softmax'], 'Softmax Classifier', 'Ideal Softmax Classifier')
    plot_learning_curve(ideal_results['fc_adam'], 'FC NN with Adam', 'Ideal Fully-Connected Neural Network (Adam)')
    plot_learning_curve(ideal_results['fc_sgd'], 'FC NN with SGD', 'Ideal Fully-Connected Neural Network (SGD)')
    
    print("理想学习曲线生成完成！")
    print("生成的图像文件：")
    print("- ideal_curves/Softmax_Classifier_ideal_learning_curve.png")
    print("- ideal_curves/FC_NN_with_Adam_ideal_learning_curve.png")
    print("- ideal_curves/FC_NN_with_SGD_ideal_learning_curve.png")
    print("\n理想训练曲线特征：")
    print("1. 损失曲线：")
    print("   - 训练损失随epoch增加而单调下降，最终趋于稳定")
    print("   - 验证损失随epoch增加而下降，在某个点后趋于稳定")
    print("   - 训练损失和验证损失之间的差距较小")
    print("2. 准确率曲线：")
    print("   - 训练准确率随epoch增加而单调上升，最终趋于稳定")
    print("   - 验证准确率随epoch增加而上升，在某个点后趋于稳定")
    print("   - 训练准确率和验证准确率之间的差距较小")
    print("3. 模型比较：")
    print("   - 全连接神经网络的性能优于Softmax分类器")
    print("   - Adam优化器收敛速度快于SGD优化器")
    print("   - SGD优化器在后期可能获得更好的泛化性能")

if __name__ == "__main__":
    main()
