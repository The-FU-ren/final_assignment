import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

class SimpleDRSN_NTF:
    """简化版的DRSN-NTF模型，使用numpy实现"""
    
    def __init__(self, num_classes=6):
        self.num_classes = num_classes
        self.model = LogisticRegression(max_iter=1000, random_state=42)
    
    def extract_features(self, X):
        """简化的特征提取"""
        # 使用简单的统计特征
        features = []
        for sample in X:
            # 时域特征
            mean = np.mean(sample)
            std = np.std(sample)
            max_val = np.max(sample)
            min_val = np.min(sample)
            rms = np.sqrt(np.mean(sample**2))
            
            # 频域特征（使用FFT）
            fft = np.fft.fft(sample)
            fft_mag = np.abs(fft[:len(fft)//2])
            fft_mean = np.mean(fft_mag)
            fft_std = np.std(fft_mag)
            
            features.append([mean, std, max_val, min_val, rms, fft_mean, fft_std])
        
        return np.array(features)
    
    def fit(self, X, y):
        """训练模型"""
        features = self.extract_features(X)
        self.model.fit(features, y)
    
    def predict(self, X):
        """预测"""
        features = self.extract_features(X)
        return self.model.predict(features)
    
    def evaluate(self, X, y):
        """评估模型"""
        features = self.extract_features(X)
        y_pred = self.model.predict(features)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='macro')
        return acc, f1

def add_noise(data, snr_db):
    """添加指定信噪比的高斯噪声"""
    signal_power = np.mean(data ** 2, axis=1, keepdims=True)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
    return data + noise

def load_data():
    """加载模拟数据"""
    num_samples = 1000
    signal_length = 8000
    num_classes = 6
    
    data = np.random.randn(num_samples, signal_length)
    labels = np.random.randint(0, num_classes, num_samples)
    
    return data, labels

def main():
    """主函数"""
    print("使用简化版DRSN-NTF模型")
    
    # 加载数据
    data, labels = load_data()
    
    # 十折交叉验证
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    
    all_acc = []
    all_f1 = []
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(data)):
        print(f"\n=== Fold {fold+1}/10 ===")
        
        # 划分数据
        X_train, X_test = data[train_idx], data[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        # 添加噪声（0dB）
        X_train = add_noise(X_train, snr_db=0)
        X_test = add_noise(X_test, snr_db=0)
        
        # 创建模型
        model = SimpleDRSN_NTF()
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 评估模型
        train_acc, train_f1 = model.evaluate(X_train, y_train)
        test_acc, test_f1 = model.evaluate(X_test, y_test)
        
        all_acc.append(test_acc)
        all_f1.append(test_f1)
        
        print(f"Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")
    
    # 计算平均结果
    avg_acc = np.mean(all_acc)
    avg_f1 = np.mean(all_f1)
    
    print(f"\n=== 十折交叉验证平均结果 ===")
    print(f"Average Test Acc: {avg_acc:.4f}")
    print(f"Average Test F1: {avg_f1:.4f}")

if __name__ == "__main__":
    main()
