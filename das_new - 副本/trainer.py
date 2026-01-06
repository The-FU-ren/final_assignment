import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(self.device)
        
        # 定义损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=0.0001
        )
        
        # 定义学习率调度器
        # 学习率：前40轮0.1，中间40轮0.01，最后20轮0.001
        def lr_lambda(epoch):
            if epoch < 40:
                return 1.0
            elif epoch < 80:
                return 0.1
            else:
                return 0.01
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
    
    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"轮次 {epoch+1}/{100}", unit="批次")
        
        for signals, labels in progress_bar:
            # 将数据移到设备上
            signals, labels = signals.to(self.device), labels.to(self.device)
            
            # 清零参数梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(signals)
            loss = self.criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            self.optimizer.step()
            
            # 更新统计信息
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                '损失': f"{total_loss/len(progress_bar):.3f}",
                '准确率': f"{100.*correct/total:.2f}%"
            })
        
        # 学习率调度器步进
        self.scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for signals, labels in val_loader:
                # 将数据移到设备上
                signals, labels = signals.to(self.device), labels.to(self.device)
                
                # 前向传播
                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)
                
                # 更新统计信息
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # 处理空验证集
        if len(val_loader) == 0:
            avg_loss = 0.0
            accuracy = 0.0
        else:
            avg_loss = total_loss / len(val_loader)
            accuracy = 100. * correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def test(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for signals, labels in test_loader:
                # 将数据移到设备上
                signals, labels = signals.to(self.device), labels.to(self.device)
                
                # 前向传播
                outputs = self.model(signals)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # 处理空测试集
        accuracy = 100. * correct / total if total > 0 else 0.0
        
        return accuracy
    
    def save_model(self, save_path):
        """
        保存训练好的模型
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, save_path)
    
    def load_model(self, load_path):
        """
        加载预训练模型
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    def train(self, train_loader, val_loader, num_epochs=100, save_dir="./models"):
        """
        训练模型指定轮数
        """
        best_val_accuracy = 0.0
        
        # 添加训练曲线数据收集
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(num_epochs):
            # 训练一轮
            train_loss, train_accuracy = self.train_one_epoch(train_loader, epoch)
            
            # 验证
            val_loss, val_accuracy = self.validate(val_loader)
            
            # 收集训练曲线数据
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            # 打印统计信息
            print(f"轮次 {epoch+1}/{num_epochs}:")
            print(f"  训练损失: {train_loss:.3f}, 训练准确率: {train_accuracy:.2f}%")
            print(f"  验证损失: {val_loss:.3f}, 验证准确率: {val_accuracy:.2f}%")
            print()
            
            # 保存最佳模型
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_model(os.path.join(save_dir, "best_model.pth"))
                print(f"  已保存最佳模型，验证准确率: {best_val_accuracy:.2f}%")
        
        # 保存最终模型
        self.save_model(os.path.join(save_dir, "final_model.pth"))
        
        # 返回训练曲线数据和最佳准确率
        return best_val_accuracy, {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
