import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子以确保结果可复现
torch.manual_seed(42)

# 生成合成的MNIST风格数据
def generate_synthetic_data(num_samples=10000, image_size=28, num_classes=10):
    # 生成随机图像数据
    images = torch.rand(num_samples, 1, image_size, image_size)
    # 生成随机标签
    labels = torch.randint(0, num_classes, (num_samples,))
    return images, labels

# 数据预处理
transform = None

# 生成合成数据
train_images, train_labels = generate_synthetic_data(num_samples=60000)
test_images, test_labels = generate_synthetic_data(num_samples=10000)

# 创建数据集
train_dataset = TensorDataset(train_images, train_labels)
test_dataset = TensorDataset(test_images, test_labels)

# 划分训练集和验证集
val_size = 10000
train_size = len(train_dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 定义Softmax分类器
class SoftmaxClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SoftmaxClassifier, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        # 将输入展平为(batch_size, input_size)
        x = x.view(x.size(0), -1)
        return self.linear(x)

# 定义全连接神经网络
class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# 训练函数
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
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
        
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }

# 测试函数
def test(model, test_loader):
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
def plot_learning_curve(results, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    ax1.plot(results['train_losses'], label='Train Loss')
    ax1.plot(results['val_losses'], label='Validation Loss')
    ax1.set_title(f'{title} - Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制准确率曲线
    ax2.plot(results['train_accuracies'], label='Train Accuracy')
    ax2.plot(results['val_accuracies'], label='Validation Accuracy')
    ax2.set_title(f'{title} - Accuracy Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}_learning_curve.png')
    plt.close()

# 主函数
def main():
    # 模型参数
    input_size = 28 * 28  # MNIST图像大小为28x28
    num_classes = 10  # 10个类别
    
    print("\n=== Training Softmax Classifier ===")
    # 初始化Softmax分类器
    softmax_model = SoftmaxClassifier(input_size, num_classes).to(device)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器 - 使用Adam优化器
    optimizer = optim.Adam(softmax_model.parameters(), lr=0.001)
    # 训练模型
    softmax_results = train(softmax_model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
    # 绘制学习曲线
    plot_learning_curve(softmax_results, "Softmax Classifier")
    # 测试模型
    softmax_test_acc = test(softmax_model, test_loader)
    
    print("\n=== Training Fully-Connected Neural Network ===")
    # 初始化全连接神经网络
    fc_model = FullyConnectedNN(input_size, num_classes).to(device)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器 - 使用Adam优化器
    optimizer = optim.Adam(fc_model.parameters(), lr=0.001)
    # 训练模型
    fc_results = train(fc_model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
    # 绘制学习曲线
    plot_learning_curve(fc_results, "Fully-Connected Neural Network")
    # 测试模型
    fc_test_acc = test(fc_model, test_loader)
    
    # 额外实验：使用不同的优化算法（SGD）
    print("\n=== Additional Experiment: FC NN with SGD Optimizer ===")
    # 初始化全连接神经网络
    fc_sgd_model = FullyConnectedNN(input_size, num_classes).to(device)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器 - 使用SGD优化器
    optimizer = optim.SGD(fc_sgd_model.parameters(), lr=0.01, momentum=0.9)
    # 训练模型
    fc_sgd_results = train(fc_sgd_model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
    # 绘制学习曲线
    plot_learning_curve(fc_sgd_results, "FC NN with SGD")
    # 测试模型
    fc_sgd_test_acc = test(fc_sgd_model, test_loader)
    
    # 输出结果总结
    print("\n=== Final Results ===")
    print(f'Softmax Classifier Test Accuracy: {softmax_test_acc:.2f}%')
    print(f'FC NN with Adam Test Accuracy: {fc_test_acc:.2f}%')
    print(f'FC NN with SGD Test Accuracy: {fc_sgd_test_acc:.2f}%')
    
    # 保存最佳模型
    torch.save(fc_model.state_dict(), 'best_fc_model.pth')
    print("\nBest model saved as 'best_fc_model.pth'")

if __name__ == "__main__":
    main()