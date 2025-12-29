import torch
import numpy as np
import argparse
from model import DRSN_NTF
from train import train_one_epoch, validate
from data_processor import load_data, preprocess_data, DAADataset
from torch.utils.data import DataLoader

# 测试GPU支持
def test_gpu_support():
    print("\n" + "="*60)
    print("测试GPU支持")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # 创建一个简单的模型
    model = DRSN_NTF(num_classes=6)
    model = model.to(device)
    
    # 创建一个输入张量
    input_tensor = torch.randn(1, 1, 8000).to(device)
    
    # 前向传播
    output = model(input_tensor)
    print(f"模型输入形状: {input_tensor.shape}")
    print(f"模型输出形状: {output.shape}")
    print("模型在GPU上运行成功！")

# 测试超参数灵活性
def test_hyperparameter_flexibility():
    print("\n" + "="*60)
    print("测试超参数灵活性")
    print("="*60)
    
    # 测试不同的超参数组合
    hyperparam_combinations = [
        {'initial_channels': 8, 'depth': 34, 'kernel_size': 3, 'threshold_N': 1.0, 'dropout': 0.2, 'use_batchnorm': True},
        {'initial_channels': 12, 'depth': 24, 'kernel_size': 5, 'threshold_N': 1.5, 'dropout': 0.3, 'use_batchnorm': False},
        {'initial_channels': 16, 'depth': 48, 'kernel_size': 7, 'threshold_N': 0.8, 'dropout': 0.1, 'use_batchnorm': True}
    ]
    
    for i, hyperparams in enumerate(hyperparam_combinations):
        print(f"\n--- 超参数组合 {i+1} ---")
        print(hyperparams)
        
        # 创建模型
        model = DRSN_NTF(num_classes=6, **hyperparams)
        
        # 创建输入张量
        input_tensor = torch.randn(1, 1, 8000)
        
        # 前向传播
        output = model(input_tensor)
        print(f"输出形状: {output.shape}")
        print(f"模型创建成功！")

# 测试模型训练流程
def test_training_flow():
    print("\n" + "="*60)
    print("测试模型训练流程")
    print("="*60)
    
    # 加载数据
    data, labels = load_data()
    
    # 预处理数据
    from sklearn.model_selection import train_test_split
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )
    
    # 预处理
    train_data = preprocess_data(train_data, snr_db=0)
    test_data = preprocess_data(test_data, snr_db=0)
    
    # 创建数据集和数据加载器
    train_dataset = DAADataset(train_data, train_labels)
    test_dataset = DAADataset(test_data, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = DRSN_NTF(num_classes=len(np.unique(labels)))
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # 训练一个epoch
    print("开始训练一个epoch...")
    train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"训练损失: {train_loss:.4f}")
    print(f"训练准确率: {train_acc:.4f}")
    print(f"训练F1分数: {train_f1:.4f}")
    
    # 验证模型
    print("开始验证...")
    test_loss, test_acc, test_f1 = validate(model, test_loader, criterion, device)
    print(f"验证损失: {test_loss:.4f}")
    print(f"验证准确率: {test_acc:.4f}")
    print(f"验证F1分数: {test_f1:.4f}")
    
    print("训练流程测试成功！")

if __name__ == "__main__":
    print("开始测试所有功能...")
    
    # 运行所有测试
    test_gpu_support()
    test_hyperparameter_flexibility()
    test_training_flow()
    
    print("\n" + "="*60)
    print("所有功能测试完成！")
    print("="*60)
