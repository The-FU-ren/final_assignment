import torch
import torch.nn as nn
import torch.nn.functional as F

class NewThresholdFunction(nn.Module):
    def __init__(self, N=1.0):
        super(NewThresholdFunction, self).__init__()
        self.N = N
    
    def forward(self, x, tau):
        mask = torch.abs(x) > tau
        y = torch.sign(x) * (torch.abs(x) - tau) * torch.exp(-self.N * tau / (torch.abs(x) + 1e-8))
        y = y * mask.float()
        return y

class RSBU_NTF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=False):
        super(RSBU_NTF, self).__init__()
        self.downsample = downsample
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 特殊阈值模块
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(out_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.sigmoid = nn.Sigmoid()
        
        # 新阈值函数
        self.new_threshold = NewThresholdFunction()
        
        # 捷径连接
        if self.downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 计算自适应阈值
        alpha = self.global_avg_pool(out).squeeze(-1)  # (batch, channels)
        lambda_param = self.fc1(alpha)
        lambda_param = self.fc2(lambda_param)
        beta = self.sigmoid(lambda_param)  # (batch, channels)
        tau = beta * alpha  # (batch, channels)
        
        # 应用新阈值函数
        out = self.new_threshold(out, tau.unsqueeze(-1))
        
        out += residual
        out = self.relu(out)
        
        return out

class DRSN_NTF(nn.Module):
    def __init__(self, num_classes=6):
        super(DRSN_NTF, self).__init__()
        
        # 输入层: 1×8000×1 -> 4×2000×1
        self.layer1_down = RSBU_NTF(1, 4, kernel_size=3, stride=2, downsample=True)
        self.layer1 = RSBU_NTF(4, 4, kernel_size=3)
        
        # 特征提取阶段2: 4×2000×1 -> 8×1000×1
        self.layer2_down = RSBU_NTF(4, 8, kernel_size=3, stride=2, downsample=True)
        self.layer2 = RSBU_NTF(8, 8, kernel_size=3)
        
        # 特征提取阶段3: 8×1000×1 -> 16×500×1
        self.layer3_down = RSBU_NTF(8, 16, kernel_size=3, stride=2, downsample=True)
        self.layer3 = RSBU_NTF(16, 16, kernel_size=3)
        
        # 全局均值池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc = nn.Linear(16, num_classes)
    
    def forward(self, x):
        # x shape: (batch, 1, 8000)
        
        out = self.layer1_down(x)
        out = self.layer1(out)
        
        out = self.layer2_down(out)
        out = self.layer2(out)
        
        out = self.layer3_down(out)
        out = self.layer3(out)
        
        out = self.global_avg_pool(out).squeeze(-1)  # (batch, 16)
        out = self.fc(out)  # (batch, num_classes)
        
        return out

# 测试模型
if __name__ == "__main__":
    model = DRSN_NTF(num_classes=6)
    input_tensor = torch.randn(1, 1, 8000)
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print("Model created successfully!")
