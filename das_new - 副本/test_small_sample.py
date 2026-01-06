import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from glob import glob

class DASTestDataset(Dataset):
    def __init__(self, data_dir, transform=None, sample_length=8000, window_step=4000, max_files_per_class=1):
        """
        初始化DAS测试数据集，只处理少量文件用于测试
        
        参数:
            data_dir: 数据集目录路径，包含各个类别子目录
            transform: 数据变换
            sample_length: 单个样本的长度
            window_step: 滑动窗口的步长
            max_files_per_class: 每个类别最多处理的文件数
        """
        self.data_dir = data_dir
        self.transform = transform
        self.sample_length = sample_length
        self.window_step = window_step
        self.max_files_per_class = max_files_per_class
        self.data = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        
        # 从目录加载数据
        self._load_data()
        
    def _load_data(self):
        """
        从目录加载少量DAS信号数据用于测试
        """
        import h5py
        from tqdm import tqdm
        
        print(f"从 {self.data_dir} 加载测试数据集，每个类别最多处理 {self.max_files_per_class} 个文件")
        
        # 获取所有类别目录
        categories = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        print(f"找到 {len(categories)} 个类别: {categories}")
        
        # 为每个类别编码标签
        self.label_encoder.fit(categories)
        
        all_samples = []
        all_labels = []
        
        # 遍历每个类别目录
        for category in categories:
            category_dir = os.path.join(self.data_dir, category)
            label = self.label_encoder.transform([category])[0]
            
            # 获取该类别下的所有.h5文件
            h5_files = glob(os.path.join(category_dir, "*.h5"))
            print(f"类别 {category} 有 {len(h5_files)} 个.h5文件，只处理前 {self.max_files_per_class} 个")
            
            # 只处理前max_files_per_class个文件
            h5_files = h5_files[:self.max_files_per_class]
            
            # 处理每个.h5文件
            for h5_file in tqdm(h5_files, desc=f"处理 {category}", unit="文件"):
                try:
                    # 读取h5文件
                    with h5py.File(h5_file, 'r') as f:
                        # 获取原始数据，形状为(time, channels)
                        raw_data = f["Acquisition/Raw[0]/RawData"][()]
                    
                    # 读取对应的.npy文件（位图文件）
                    npy_file = h5_file[:-2] + "npy"
                    if os.path.exists(npy_file):
                        bitmap = np.load(npy_file)
                    else:
                        print(f"警告: 未找到对应的.npy文件 {npy_file}，跳过该文件")
                        continue
                    
                    # 生成窗口样本，只生成少量样本
                    samples = self._generate_samples_from_file(raw_data, bitmap)
                    
                    # 只取前10个样本
                    samples = samples[:10]
                    
                    # 将样本添加到列表
                    all_samples.extend(samples)
                    all_labels.extend([label] * len(samples))
                    
                except Exception as e:
                    print(f"处理文件 {h5_file} 时出错: {e}")
                    continue
        
        # 转换为numpy数组
        self.data = np.array(all_samples)
        self.labels = np.array(all_labels)
        
        print(f"成功加载 {len(self.data)} 个样本，{len(np.unique(self.labels))} 个类别")
        print(f"类别分布: {dict(zip(self.label_encoder.classes_, np.bincount(self.labels)))} ")
        
        # 归一化数据
        self._normalize_data()
    
    def _generate_samples_from_file(self, raw_data, bitmap):
        """
        从单个文件的原始数据和位图生成样本
        """
        samples = []
        
        # 获取非零位置（事件发生的位置）
        event_positions = np.transpose(np.where(bitmap))
        
        # 如果没有事件位置，使用所有位置
        if len(event_positions) == 0:
            # 生成所有可能的窗口位置
            num_windows = (raw_data.shape[0] - self.sample_length) // self.window_step + 1
            event_positions = np.array([(i, 0) for i in range(num_windows)])
        
        # 只处理前20个事件位置
        event_positions = event_positions[:20]
        
        # 遍历事件位置，生成样本
        for pos_idx, channel in event_positions:
            # 计算窗口起始和结束位置
            start_time = pos_idx * self.window_step
            end_time = start_time + self.sample_length
            
            # 确保窗口不超出数据范围
            if end_time <= raw_data.shape[0]:
                # 获取窗口数据
                window_data = raw_data[start_time:end_time, channel]
                samples.append(window_data)
        
        return samples
    
    def _normalize_data(self):
        """
        归一化DAS信号数据
        """
        if len(self.data) == 0:
            return
        
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
        signal = signal.reshape(1, self.sample_length, 1)
        
        if self.transform:
            signal = self.transform(signal)
        
        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# 测试数据加载器
def test_data_loader():
    data_dir = "G:/DAS-dataset_3/DAS-dataset/data"
    
    # 创建测试数据集
    test_dataset = DASTestDataset(data_dir, max_files_per_class=1)
    
    # 创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    
    print(f"测试加载器: {len(test_loader)} 批次")
    
    # 检查第一个批次
    for signals, labels in test_loader:
        print(f"信号形状: {signals.shape}")
        print(f"标签形状: {labels.shape}")
        break

if __name__ == "__main__":
    test_data_loader()
