import torch
import torch.nn as nn
import torch.nn.functional as F

class NewThresholdFunction(nn.Module):
    def __init__(self, N=1.0):
        super(NewThresholdFunction, self).__init__()
        self.N = N
    
    def forward(self, x, tau):
        # 确保tau是正值
        tau = torch.abs(tau)
        
        # 添加一个小的正值，确保tau不会为0
        tau = tau + 1e-10
        
        # 计算掩码
        mask = torch.abs(x) > tau
        
        # 安全计算，避免数值问题
        abs_x = torch.abs(x) + 1e-10
        exp_term = torch.exp(-self.N * tau / abs_x)
        y = torch.sign(x) * (abs_x - tau) * exp_term
        
        # 应用掩码
        y = y * mask.float()
        
        # 确保输出没有nan或inf值
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        return y

class RSBU_NTF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=False, threshold_N=1.0, dropout=0.0, use_batchnorm=True):
        super(RSBU_NTF, self).__init__()
        self.downsample = downsample
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        
        # 计算padding，确保卷积输出的长度与输入长度匹配（当stride=1时）
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        if use_batchnorm:
            self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if dropout > 0:
            self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        if use_batchnorm:
            self.bn2 = nn.BatchNorm1d(out_channels)
        if dropout > 0:
            self.dropout2 = nn.Dropout(dropout)
        
        # 特殊阈值模块
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(out_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.sigmoid = nn.Sigmoid()
        
        # 新阈值函数
        self.new_threshold = NewThresholdFunction(N=threshold_N)
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        
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
        if self.use_batchnorm:
            out = self.bn1(out)
        out = self.relu(out)
        if self.dropout > 0:
            out = self.dropout1(out)
        
        out = self.conv2(out)
        if self.use_batchnorm:
            out = self.bn2(out)
        
        # 计算自适应阈值
        alpha = self.global_avg_pool(out).squeeze(-1)  # (batch, channels)
        lambda_param = self.fc1(alpha)
        lambda_param = self.fc2(lambda_param)
        beta = self.sigmoid(lambda_param)  # (batch, channels)
        tau = beta * alpha  # (batch, channels)
        
        # 应用新阈值函数
        out = self.new_threshold(out, tau.unsqueeze(-1))
        
        # 确保out和residual大小匹配
        if out.size(2) != residual.size(2):
            # 使用自适应池化调整out大小以匹配residual
            out = nn.functional.adaptive_avg_pool1d(out, residual.size(2))
        
        out += residual
        out = self.relu(out)
        if self.dropout > 0:
            out = self.dropout2(out)
        
        return out

class DRSN_NTF(nn.Module):
    def __init__(self, num_classes=6, initial_channels=8, depth=34, kernel_size=3, 
                 threshold_N=1.0, dropout=0.2, use_batchnorm=True):
        super(DRSN_NTF, self).__init__()
        self.initial_channels = initial_channels
        self.depth = depth
        self.kernel_size = kernel_size
        self.threshold_N = threshold_N
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        
        # 计算每个阶段的RSBU-NTF单元数量
        layers_per_stage = depth // 3
        channels = [initial_channels, initial_channels*2, initial_channels*4]
        
        # 网络层次列表，用于动态构建网络
        self.network = nn.ModuleList()
        
        # 输入层：1×8000×1 -> initial_channels×4000×1
        self.network.append(RSBU_NTF(1, channels[0], kernel_size=kernel_size, stride=2, 
                                    downsample=True, threshold_N=threshold_N, 
                                    dropout=dropout, use_batchnorm=use_batchnorm))
        # 添加第一个阶段的剩余RSBU-NTF单元
        for _ in range(layers_per_stage - 1):
            self.network.append(RSBU_NTF(channels[0], channels[0], kernel_size=kernel_size, 
                                        threshold_N=threshold_N, dropout=dropout, 
                                        use_batchnorm=use_batchnorm))
        
        # 第二个阶段：channels[0]×4000×1 -> channels[1]×2000×1
        self.network.append(RSBU_NTF(channels[0], channels[1], kernel_size=kernel_size, stride=2, 
                                    downsample=True, threshold_N=threshold_N, 
                                    dropout=dropout, use_batchnorm=use_batchnorm))
        # 添加第二个阶段的剩余RSBU-NTF单元
        for _ in range(layers_per_stage - 1):
            self.network.append(RSBU_NTF(channels[1], channels[1], kernel_size=kernel_size, 
                                        threshold_N=threshold_N, dropout=dropout, 
                                        use_batchnorm=use_batchnorm))
        
        # 第三个阶段：channels[1]×2000×1 -> channels[2]×1000×1
        self.network.append(RSBU_NTF(channels[1], channels[2], kernel_size=kernel_size, stride=2, 
                                    downsample=True, threshold_N=threshold_N, 
                                    dropout=dropout, use_batchnorm=use_batchnorm))
        # 添加第三个阶段的剩余RSBU-NTF单元
        for _ in range(layers_per_stage - 1):
            self.network.append(RSBU_NTF(channels[2], channels[2], kernel_size=kernel_size, 
                                        threshold_N=threshold_N, dropout=dropout, 
                                        use_batchnorm=use_batchnorm))
        
        # 全局均值池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc = nn.Linear(channels[2], num_classes)
    
    def forward(self, x):
        # x shape: (batch, 1, 8000)
        
        out = x
        for layer in self.network:
            out = layer(out)
        
        out = self.global_avg_pool(out).squeeze(-1)  # (batch, channels[2])
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
