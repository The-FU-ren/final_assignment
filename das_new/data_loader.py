import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

class DASDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        self.labels = []
        
        # 从目录加载数据
        self._load_data()
        
    def _load_data(self):
        """
        从指定目录加载DAS信号数据
        数据存储在HDF5文件的Acquisition/Raw[0]/RawData路径下
        充分利用所有通道和数据，生成更多样本
        """
        import h5py
        from tqdm import tqdm
        
        # 根据目录名定义标签映射
        label_map = {
            'car': 0,       # 汽车
            'human': 1,     # 人类
            'train': 2,     # 火车
            'excavator': 3, # 挖掘机
            'dog': 4,       # 狗
            'bike': 5       # 自行车
        }
        
        # 首先，收集所有HDF5文件
        h5_files = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.h5') or file.endswith('.hdf5'):
                    file_path = os.path.join(root, file)
                    h5_files.append(file_path)
        
        print(f"找到 {len(h5_files)} 个HDF5文件需要处理")
        print("正在处理所有文件，充分利用所有通道和数据...")
        
        # 使用进度条处理每个文件
        processed_files = 0
        total_samples = 0
        
        for file_path in tqdm(h5_files, desc="加载数据", unit="文件"):
            # 为所有文件分配标签
            parts = file_path.split(os.sep)
            label = 0  # 默认标签为0（汽车）
            
            # 尝试从路径中查找任何可能的标签
            for part in parts:
                if part in label_map:
                    label = label_map[part]
                    break
            
            try:
                with h5py.File(file_path, 'r') as f:
                    # 读取指定路径的数据
                    data_path = 'Acquisition/Raw[0]/RawData'
                    if data_path in f:
                        # 获取数据形状
                        raw_data = f[data_path]
                        raw_data_shape = raw_data.shape
                        num_time_points = raw_data_shape[0]
                        num_channels = raw_data_shape[1]
                        
                        print(f"  文件 {os.path.basename(file_path)}: {num_time_points}时间点 × {num_channels}通道")
                        
                        # 调整样本提取参数，生成更多样本
                        sample_length = 8000  # 样本长度保持不变
                        step = 2000  # 减小步长，增加样本数量（75%重叠）
                        
                        # 遍历所有通道，充分利用数据
                        for channel_idx in range(min(num_channels, 10)):  # 处理前10个通道，可根据内存调整
                            print(f"    处理第 {channel_idx+1}/{num_channels} 通道...")
                            
                            # 计算可以提取的样本数
                            num_samples_per_channel = (num_time_points - sample_length) // step + 1
                            
                            # 逐块读取，不一次性加载整个文件
                            for i in range(num_samples_per_channel):
                                start_idx = i * step
                                end_idx = start_idx + sample_length
                                
                                if end_idx <= num_time_points:
                                    # 只读取当前样本所需的数据块
                                    raw_data_chunk = raw_data[start_idx:end_idx, channel_idx]
                                    signal = raw_data_chunk
                                else:
                                    # 如果在末尾，用零填充
                                    signal = np.zeros(sample_length, dtype=np.float32)
                                    raw_data_chunk = raw_data[start_idx:, channel_idx]
                                    signal[:len(raw_data_chunk)] = raw_data_chunk
                                
                                self.data.append(signal)
                                self.labels.append(label)
                                total_samples += 1
                        
                        processed_files += 1
                        print(f"  从 {os.path.basename(file_path)} 中提取了 {total_samples} 个样本")
                    else:
                        print(f"  文件 {os.path.basename(file_path)} 中找不到 {data_path} 路径")
            except Exception as e:
                print(f"  处理文件 {os.path.basename(file_path)} 时出错: {e}")
                print(f"  继续处理下一个文件...")
        
        # 转换为numpy数组
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        
        print(f"处理完成！共处理了 {processed_files}/{len(h5_files)} 个文件")
        print(f"生成了 {len(self.data)} 个训练样本")
        print(f"每个样本大小: {self.data.shape[1]} 点")
        print(f"总数据量: {self.data.nbytes / 1024 / 1024:.2f} MB")
        
        # 转换为numpy数组
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        
        # 归一化数据
        self._normalize_data()
    
    def _normalize_data(self):
        """
        归一化DAS信号数据
        """
        # 重塑数据以适应StandardScaler (样本数, 特征数)
        num_samples, num_features = self.data.shape
        data_reshaped = self.data.reshape(num_samples, num_features)
        
        # 应用StandardScaler
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_reshaped)
        
        # 重塑回原始形状
        self.data = data_scaled.reshape(num_samples, num_features)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        signal = self.data[idx]
        label = self.labels[idx]
        
        # 重塑信号以匹配模型输入 (1, 8000, 1)
        signal = signal.reshape(1, 8000, 1)
        
        if self.transform:
            signal = self.transform(signal)
        
        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def get_data_loaders(data_dir, batch_size=128, val_split=0.1, test_split=0.2):
    """
    获取训练、验证和测试的数据加载器
    
    参数:
        data_dir: 数据集所在目录
        batch_size: 训练批次大小
        val_split: 验证集划分比例
        test_split: 测试集划分比例
        
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    # 创建数据集
    dataset = DASDataset(data_dir)
    
    # 计算划分大小
    total_size = len(dataset)
    
    # 确保每个集合至少有1个样本
    test_size = max(1, int(total_size * test_split))
    val_size = max(1, int(total_size * val_split))
    train_size = max(1, total_size - test_size - val_size)
    
    # 如果大小不匹配，进行调整
    if train_size + val_size + test_size != total_size:
        diff = total_size - (train_size + val_size + test_size)
        train_size += diff
    
    print(f"数据集划分: {train_size} 训练样本, {val_size} 验证样本, {test_size} 测试样本")
    
    # 划分数据集
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader

# 示例用法
def test_data_loader():
    data_dir = "G:/DAS-dataset_3/DAS-dataset"
    train_loader, val_loader, test_loader = get_data_loaders(data_dir, batch_size=16)
    
    print(f"训练加载器: {len(train_loader)} 批次")
    print(f"验证加载器: {len(val_loader)} 批次")
    print(f"测试加载器: {len(test_loader)} 批次")
    
    # 检查第一个批次
    for signals, labels in train_loader:
        print(f"信号形状: {signals.shape}")
        print(f"标签形状: {labels.shape}")
        break

if __name__ == "__main__":
    test_data_loader()
