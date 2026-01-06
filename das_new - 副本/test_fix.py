import torch
from model import DRSN_NTF
from data_loader import DASDataset

# 测试模型和标签是否匹配
def test_model_label_matching():
    data_dir = "G:/DAS-dataset_3/DAS-dataset/data"
    
    # 创建数据集，限制样本数量以便快速测试
    dataset = DASDataset(data_dir, max_samples_per_file=10)
    
    print(f"数据集类别数量: {len(dataset.label_encoder.classes_)}")
    print(f"类别: {dataset.label_encoder.classes_}")
    
    # 检查标签范围
    labels = [meta['label'] for meta in dataset.sample_metadata]
    min_label = min(labels)
    max_label = max(labels)
    print(f"标签范围: {min_label} - {max_label}")
    
    # 初始化模型，使用正确的类别数量
    model = DRSN_NTF(num_classes=len(dataset.label_encoder.classes_))
    print(f"模型输出维度: {model.fc.out_features}")
    
    # 测试前向传播
    sample_signal, sample_label = dataset[0]
    print(f"样本标签: {sample_label}")
    
    # 添加批次维度
    sample_signal = sample_signal.unsqueeze(0)
    
    # 前向传播
    output = model(sample_signal)
    print(f"模型输出形状: {output.shape}")
    print(f"模型输出: {output}")
    
    # 检查损失计算
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, torch.tensor([sample_label]))
    print(f"损失值: {loss.item()}")
    
    print("\n测试完成！模型和标签匹配正常。")

if __name__ == "__main__":
    test_model_label_matching()
