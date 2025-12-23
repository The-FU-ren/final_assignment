import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.trial import TrialState

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 数据预处理和增强
def get_data_loaders(batch_size=64, use_augmentation=True):
    # 训练数据增强
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # 测试数据不需要增强
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载Fashion MNIST数据集
    # 检查数据集是否完整，如果完整则不下载
    import os
    fashion_mnist_files = [
        './data/FashionMNIST/raw/train-images-idx3-ubyte.gz',
        './data/FashionMNIST/raw/train-labels-idx1-ubyte.gz',
        './data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz',
        './data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz'
    ]
    
    # 检查所有必要文件是否存在
    all_files_exist = all(os.path.exists(f) for f in fashion_mnist_files)
    
    # 如果所有文件存在，则不下载；否则下载
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=not all_files_exist, transform=train_transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=not all_files_exist, transform=test_transform)
    
    # 划分训练集和验证集
    val_size = 10000
    indices = torch.randperm(len(train_dataset))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, train_indices),
        batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, val_indices),
        batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader, test_loader

# 定义优化后的全连接神经网络
class OptimizedFCNN(nn.Module):
    def __init__(self, input_size=784, hidden_size1=256, hidden_size2=128, output_size=10, dropout_rate=0.5):
        super(OptimizedFCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_size2, output_size)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x

# 训练函数（带早停和学习率衰减）
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30, patience=5):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # 学习率衰减
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}: '\
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% '\
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% '\
              f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            best_model = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break
    
    # 加载最佳模型
    model.load_state_dict(best_model)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }

# 测试函数
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_acc = 100. * correct / total
    print(f'Test Accuracy: {test_acc:.2f}%')
    return test_acc

# 绘制学习曲线
def plot_learning_curves(results, title, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    epochs = range(1, len(results['train_losses']) + 1)
    
    # 绘制损失曲线
    ax1.plot(epochs, results['train_losses'], label='Train Loss', linewidth=2, color='blue')
    ax1.plot(epochs, results['val_losses'], label='Validation Loss', linewidth=2, color='red')
    ax1.set_title(f'{title} - Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(bottom=0)
    
    # 绘制准确率曲线
    ax2.plot(epochs, results['train_accuracies'], label='Train Accuracy', linewidth=2, color='blue')
    ax2.plot(epochs, results['val_accuracies'], label='Validation Accuracy', linewidth=2, color='red')
    ax2.set_title(f'{title} - Accuracy Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# 贝叶斯优化目标函数
def objective(trial):
    # 超参数搜索空间
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    hidden_size1 = trial.suggest_int('hidden_size1', 32, 256, step=32)
    hidden_size2 = trial.suggest_int('hidden_size2', 32, 256, step=32)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.6)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    # 获取数据加载器
    train_loader, val_loader, _ = get_data_loaders(batch_size=batch_size, use_augmentation=True)
    
    # 初始化模型
    model = OptimizedFCNN(
        input_size=28*28,
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2,
        dropout_rate=dropout_rate
    ).to(device)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 训练模型
    results = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30, patience=5)
    
    # 获取最佳验证准确率
    best_val_acc = max(results['val_accuracies'])
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return best_val_acc

# 主函数
def main():
    # 1. 运行贝叶斯优化
    print("\n=== Running Bayesian Optimization ===")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20, timeout=3600)
    
    # 2. 打印最佳结果
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    print("\n=== Optimization Results ===")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len(pruned_trials)}")
    print(f"Number of complete trials: {len(complete_trials)}")
    
    best_trial = study.best_trial
    print(f"\nBest trial: {best_trial.number}")
    print(f"  Value (Validation Accuracy): {best_trial.value:.2f}%")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # 3. 使用最佳超参数训练最终模型
    print("\n=== Training Final Model with Best Hyperparameters ===")
    
    # 获取最佳超参数
    best_params = best_trial.params
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=best_params['batch_size'], 
        use_augmentation=True
    )
    
    # 初始化最终模型
    final_model = OptimizedFCNN(
        input_size=28*28,
        hidden_size1=best_params['hidden_size1'],
        hidden_size2=best_params['hidden_size2'],
        dropout_rate=best_params['dropout_rate']
    ).to(device)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = optim.Adam(
        final_model.parameters(), 
        lr=best_params['lr'], 
        weight_decay=best_params['weight_decay']
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 训练最终模型
    final_results = train_model(
        final_model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        scheduler, 
        num_epochs=30, 
        patience=5
    )
    
    # 4. 测试最终模型
    print("\n=== Testing Final Model ===")
    test_acc = test_model(final_model, test_loader)
    
    # 5. 绘制学习曲线
    print("\n=== Plotting Learning Curves ===")
    plot_learning_curves(
        final_results, 
        title=f"Optimized FCNN - Test Acc: {test_acc:.2f}%",
        save_path="optimized_fcnn_learning_curve.png"
    )
    
    # 6. 保存最佳模型
    torch.save(final_model.state_dict(), 'best_optimized_model.pth')
    print("\nBest model saved as 'best_optimized_model.pth'")
    
    # 7. 解释学习曲线
    print("\n=== Learning Curve Interpretation ===")
    print("Ideal learning curve characteristics:")
    print("1. Both train and validation losses should decrease over time and stabilize")
    print("2. Both train and validation accuracies should increase over time and stabilize")
    print("3. The gap between train and validation curves should be small")
    print("4. No significant increase in validation loss after stabilization")
    print("5. The final validation accuracy should be close to the final train accuracy")
    
    # 分析实际曲线
    final_train_loss = final_results['train_losses'][-1]
    final_val_loss = final_results['val_losses'][-1]
    final_train_acc = final_results['train_accuracies'][-1]
    final_val_acc = final_results['val_accuracies'][-1]
    
    print(f"\nActual curve analysis:")
    print(f"- Final train loss: {final_train_loss:.4f}")
    print(f"- Final validation loss: {final_val_loss:.4f}")
    print(f"- Final train accuracy: {final_train_acc:.2f}%")
    print(f"- Final validation accuracy: {final_val_acc:.2f}%")
    print(f"- Loss gap: {abs(final_train_loss - final_val_loss):.4f}")
    print(f"- Accuracy gap: {abs(final_train_acc - final_val_acc):.2f}%")
    
    if abs(final_train_acc - final_val_acc) < 5:
        print("✅ Good generalization - small gap between train and validation accuracy")
    else:
        print("⚠️  Possible overfitting - large gap between train and validation accuracy")
    
    if final_val_loss < 0.5 and final_val_acc > 85:
        print("✅ Excellent performance - low validation loss and high accuracy")
    elif final_val_loss < 0.8 and final_val_acc > 80:
        print("✅ Good performance - moderate validation loss and accuracy")
    else:
        print("⚠️  Performance could be improved - high validation loss or low accuracy")

if __name__ == "__main__":
    main()
