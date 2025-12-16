# DRSN-NTF 模型复现项目

## 项目简介

本项目复现了论文 "基于新阈值函数的深度残差收缩网络 (DRSN-NTF) 用于 DAS 信号事件识别" 的实验结果。

## 项目结构

```
das/
├── 0506002.pdf              # 原始论文
├── DRSN-NTF模型复现提示词.md  # 模型复现提示
├── data_processor.py        # 数据处理模块
├── main.py                  # 主脚本
├── model.py                 # 模型定义
├── requirements.txt         # 依赖列表
├── train.py                 # 训练脚本
└── visualization.py         # 可视化模块
```

## 快速开始

### 1. 安装依赖

首先，安装项目所需的依赖：

```bash
pip install -r requirements.txt
```

### 2. 准备数据

当前项目使用的是模拟数据。要使用真实数据，需要修改 `data_processor.py` 中的 `load_data` 函数：

```python
def load_data(data_path=None):
    """加载数据集"""
    # 替换为真实数据加载逻辑
    # 例如，从文件中加载数据：
    # data = np.load('data.npy')
    # labels = np.load('labels.npy')
    
    # 返回格式：数据形状为 (样本数, 8000)，标签形状为 (样本数,)
    return data, labels
```

### 3. 运行训练

使用主脚本运行训练：

```bash
python main.py
```

这将运行所有信噪比（0dB~5dB）下的实验，每个信噪比使用十折交叉验证。

### 4. 查看结果

训练完成后，将生成以下结果文件：

- **training_curves_0dB.png** 到 **training_curves_5dB.png**：各信噪比下的训练曲线
- **snr_vs_accuracy.png**：信噪比与准确率关系图
- **experiment_report.txt**：实验报告，包含各信噪比下的平均准确率
- **experiment_results.npy**：完整的实验结果数据

## 自定义训练

### 调整训练参数

在 `main.py` 中可以调整以下参数：

```python
snr_values = [0, 1, 2, 3, 4, 5]  # 选择要运行的信噪比
batch_size = 128  # 批量大小
epochs = 100  # 训练轮数
```

### 单独运行特定信噪比

可以修改 `train.py` 中的 `if __name__ == "__main__":` 部分，单独运行特定信噪比下的训练：

```python
if __name__ == "__main__":
    train_model(snr_db=0, batch_size=128, epochs=100)
```

然后运行：

```bash
python train.py
```

## 模型架构

### 网络整体结构

```
输入层 (1×8000×1)
↓
特征提取阶段1 (4通道)
├─ RSBU-NTF (4, 3, /2) → 4×2000×1
└─ RSBU-NTF (4, 3) → 4×2000×1
↓
特征提取阶段2 (8通道)
├─ RSBU-NTF (8, 3, /2) → 8×1000×1
└─ RSBU-NTF (8, 3) → 8×1000×1
↓
特征提取阶段3 (16通道)
├─ RSBU-NTF (16, 3, /2) → 16×500×1
└─ RSBU-NTF (16, 3) → 16×500×1
↓
全局均值池化(GAP) → 16维向量
↓
全连接层 → 6维向量 (输出6个类别)
```

### RSBU-NTF 单元

RSBU-NTF 单元包含：
- 两个1×3卷积层
- 批量标准化和ReLU激活
- 自适应阈值计算模块
- 新阈值函数

## 新阈值函数

数学表达式：

```
y = sign(x) × (|x| - τ) × exp(-N × τ / |x|)  当 |x| > τ

y = 0  当 |x| ≤ τ
```

其中：
- τ: 自适应阈值
- N: 调节参数（默认为1.0）
- sign(x): 符号函数

## 实验结果

实验将在0dB~5dB不同信噪比下进行，使用十折交叉验证。结果包括：

- 训练准确率和损失曲线
- 测试准确率和损失曲线
- 各信噪比下的平均准确率
- 各事件类型的F1-score

## 技术栈

- **深度学习框架**: PyTorch
- **信号处理**: NumPy, SciPy
- **机器学习**: scikit-learn
- **可视化**: Matplotlib, Seaborn

## 注意事项

1. 确保数据形状为 (样本数, 8000)，每个样本是长度为8000的一维振动信号
2. 标签应为0~5之间的整数，对应6种事件类型
3. 训练时间较长，建议使用GPU加速
4. 可以根据需要调整模型参数，如学习率、批量大小、训练轮数等

## 许可证

本项目仅供学习和研究使用。
