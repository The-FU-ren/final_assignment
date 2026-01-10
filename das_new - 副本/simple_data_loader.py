import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class SimplifiedDASDataset(Dataset):
    def __init__(self, data_dir, sample_length=8000, max_samples_per_file=20):
        """
        简化的DAS数据集，用于目录数据集加载
        
        参数:
            data_dir: 数据集目录路径
            sample_length: 单个样本的长度
            max_samples_per_file: 每个文件最多生成的样本数
        """
        self.data_dir = data_dir
        self.sample_length = sample_length
        self.max_samples_per_file = max_samples_per_file
        self.sample_info = []
        
        # 收集样本信息
        self._collect_sample_info()
        
        # 初始化标签编码器
        self.label_encoder = LabelEncoder()
        self.labels = [info['label'] for info in self.sample_info]
        self.label_encoder.fit(self.labels)
        
        print(f"\n=== 数据集统计 ===")
        print(f"类别: {self.label_encoder.classes_}")
        print(f"样本数量: {len(self.sample_info)}")
        
    def _collect_sample_info(self):
        """
        收集样本信息
        """
        print(f"从 {self.data_dir} 收集样本信息...")
        
        # 获取所有类别目录
        categories = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        categories.sort()
        print(f"找到 {len(categories)} 个类别: {categories}")
        
        # 遍历每个类别目录
        for category in categories:
            category_dir = os.path.join(self.data_dir, category)
            label = categories.index(category)  # 使用索引作为标签
            
            # 获取所有.h5文件
            h5_files = [f for f in os.listdir(category_dir) if f.endswith('.h5')]
            print(f"类别 {category} 有 {len(h5_files)} 个.h5文件")
            
            # 遍历每个.h5文件
            for h5_file in h5_files[:3]:  # 只处理前3个文件
                h5_path = os.path.join(category_dir, h5_file)
                
                # 收集样本信息，但不加载完整数据
                self.sample_info.append({
                    'h5_path': h5_path,
                    'label': label
                })
                
                if len(self.sample_info) >= 100:  # 限制样本数量
                    break
    
    def __len__(self):
        """
        获取样本数量
        
        返回:
            样本数量
        """
        return len(self.sample_info)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        参数:
            idx: 样本索引
            
        返回:
            signals: 信号数据，形状为(1, sample_length, 1)
            labels: 标签数据
        """
        import h5py
        
        # 获取样本信息
        sample_info = self.sample_info[idx]
        h5_path = sample_info['h5_path']
        label = sample_info['label']
        
        # 读取h5文件
        with h5py.File(h5_path, 'r') as f:
            # 获取数据路径
            if 'Acquisition/Raw[0]/RawData' in f:
                data_path = 'Acquisition/Raw[0]/RawData'
            elif 'data' in f:
                data_path = 'data'
            else:
                # 尝试获取第一个数据集
                data_keys = list(f.keys())
                print(f"警告: 无法确定数据路径，使用默认路径！")
                return torch.zeros(1, self.sample_length, 1), torch.tensor(label)
            
            # 读取数据
            raw_data = f[data_path]
            
            # 随机选择一个起始位置
            start_idx = np.random.randint(0, max(0, raw_data.shape[0] - self.sample_length + 1))
            end_idx = start_idx + self.sample_length
            
            # 确保不超出范围
            if end_idx > raw_data.shape[0]:
                start_idx = raw_data.shape[0] - self.sample_length
                end_idx = raw_data.shape[0]
            
            # 读取数据
            signals = raw_data[start_idx:end_idx, 0]  # 只取第一个通道
            
            # 确保信号形状正确
            if signals.shape[0] < self.sample_length:
                # 填充到指定长度
                pad_length = self.sample_length - signals.shape[0]
                signals = np.pad(signals, (0, pad_length), 'constant')
            
        # 重塑为 (1, sample_length, 1)
        signals = torch.tensor(signals.reshape(1, -1, 1), dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return signals, label

# 测试数据加载器
if __name__ == "__main__":
    data_dir = "G:/DAS-dataset_3/DAS-dataset/data"
    
    print(f"测试数据集: {data_dir}")
    dataset = SimplifiedDASDataset(data_dir)
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    
    print(f"\n数据加载器批次: {len(dataloader)}")
    
    # 测试训练循环
    for i, (signals, labels) in enumerate(dataloader):
        print(f"批次 {i}: 信号形状={signals.shape}, 标签形状={labels.shape}")
        if i >= 2:  # 只测试3个批次
            break
    
    print("\n测试完成！")