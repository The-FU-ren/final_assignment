import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os

class Trainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', learning_rate=0.0005):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.model.to(self.device)
        
        # 定义损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.001  # 增大权重衰减，提高正则化效果
        )
        
        # 定义余弦退火学习率调度器，带热重启
        # T_0：初始周期，T_mult：每次重启后周期翻倍
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=10,  # 初始周期为10轮
            T_mult=2,  # 每次重启后周期翻倍
            eta_min=1e-6  # 最小学习率
        )
        
        # 梯度裁剪阈值
        self.grad_clip = 1.0
    
    def train_one_epoch(self, train_loader, epoch, total_epochs):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        print(f"轮次 {epoch+1}/{total_epochs} 开始")
        
        for batch_idx, (signals, labels) in enumerate(train_loader):
            # 将数据移到设备上
            signals, labels = signals.to(self.device), labels.to(self.device)
            
            # 清零参数梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(signals)
            loss = self.criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            # 更新统计信息
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 打印批次信息
            if (batch_idx + 1) % 2 == 0:  # 每2个批次打印一次
                avg_batch_loss = total_loss / (batch_idx + 1)
                batch_accuracy = 100. * correct / total
                print(f"  批次 {batch_idx+1}/{len(train_loader)}: 损失={avg_batch_loss:.3f}, 准确率={batch_accuracy:.2f}%")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        print(f"轮次 {epoch+1}/{total_epochs} 完成: 平均损失={avg_loss:.3f}, 平均准确率={accuracy:.2f}%")
        
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
    
    def train(self, train_loader, val_loader, num_epochs=100, save_dir="./models", patience=20):
        """
        训练模型指定轮数，包含早停机制
        
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 最大训练轮数
            save_dir: 模型保存目录
            patience: 早停耐心值，连续多少轮验证准确率不提升就停止
            
        返回:
            best_val_accuracy: 最佳验证准确率
            training_curves: 训练曲线数据
        """
        best_val_accuracy = 0.0
        early_stop_counter = 0
        
        # 添加训练曲线数据收集
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        try:
            for epoch in range(num_epochs):
                print(f"\n{'='*50}")
                print(f"开始轮次 {epoch+1}/{num_epochs}")
                print(f"{'='*50}")
                
                # 训练一轮
                print(f"执行训练轮次 {epoch+1}...")
                train_loss, train_accuracy = self.train_one_epoch(train_loader, epoch, num_epochs)
                print(f"训练轮次 {epoch+1} 完成")
                
                # 验证
                print(f"执行验证轮次 {epoch+1}...")
                val_loss, val_accuracy = self.validate(val_loader)
                print(f"验证轮次 {epoch+1} 完成")
                
                # 学习率调度器步进
                print(f"更新学习率...")
                self.scheduler.step()
                print(f"当前学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                # 收集训练曲线数据
                train_losses.append(train_loss)
                train_accuracies.append(train_accuracy)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                
                # 打印统计信息
                print(f"\n轮次 {epoch+1}/{num_epochs} 统计信息:")
                print(f"  训练损失: {train_loss:.3f}, 训练准确率: {train_accuracy:.2f}%")
                print(f"  验证损失: {val_loss:.3f}, 验证准确率: {val_accuracy:.2f}%")
                
                # 保存最佳模型并检查早停
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    print(f"  验证准确率提升，保存最佳模型...")
                    self.save_model(os.path.join(save_dir, "best_model.pth"))
                    print(f"  已保存最佳模型，验证准确率: {best_val_accuracy:.2f}%")
                    early_stop_counter = 0  # 重置早停计数器
                else:
                    early_stop_counter += 1
                    print(f"  验证准确率未提升，早停计数器: {early_stop_counter}/{patience}")
                
                # 检查早停条件
                if early_stop_counter >= patience:
                    print(f"\n早停机制触发！连续 {patience} 轮验证准确率未提升，停止训练")
                    break
            
            # 保存最终模型
            print(f"\n保存最终模型...")
            self.save_model(os.path.join(save_dir, "final_model.pth"))
            print(f"最终模型已保存")
            
            print(f"\n{'='*50}")
            print(f"训练完成！")
            print(f"最佳验证准确率: {best_val_accuracy:.2f}%")
            print(f"{'='*50}")
            
        except Exception as e:
            print(f"\n训练过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 返回训练曲线数据和最佳准确率
        return best_val_accuracy, {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
