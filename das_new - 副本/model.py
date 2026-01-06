import torch
import torch.nn as nn
import torch.nn.functional as F

class NewThresholdFunction(nn.Module):
    def __init__(self, N=1.0):
        super(NewThresholdFunction, self).__init__()
        self.N = N
    
    def forward(self, x, tau):
        abs_x = torch.abs(x)
        mask = abs_x >= tau
        output = torch.zeros_like(x)
        
        # 应用新阈值函数
        output[mask] = torch.sign(x[mask]) * (abs_x[mask] - tau[mask] / torch.exp((abs_x[mask] - tau[mask]) / self.N))
        output[~mask] = 0
        
        return output

class RSBU_NTF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(RSBU_NTF, self).__init__()
        
        # 特征提取路径
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=(1, stride), padding=(0, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_size), stride=(1, 1), padding=(0, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 阈值估计模块
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(out_channels, out_channels // 4)
        self.fc2 = nn.Linear(out_channels // 4, out_channels)
        self.sigmoid = nn.Sigmoid()
        
        # 新阈值函数
        self.new_threshold = NewThresholdFunction(N=1.0)
        
        #  shortcut连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, stride), bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        # 特征提取
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 阈值估计
        abs_out = torch.abs(out)
        alpha = self.gap(abs_out)
        alpha = alpha.view(alpha.size(0), -1)
        
        lambda_ = self.fc1(alpha)
        lambda_ = F.relu(lambda_)
        lambda_ = self.fc2(lambda_)
        
        beta = self.sigmoid(lambda_)
        beta = beta.view(beta.size(0), beta.size(1), 1, 1)
        
        # 计算阈值
        tau = beta * self.gap(abs_out).expand_as(out)
        
        # 应用新阈值函数
        out = self.new_threshold(out, tau)
        
        # shortcut连接
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

class DRSN_NTF(nn.Module):
    def __init__(self, num_classes=9):
        super(DRSN_NTF, self).__init__()
        
        # 输入层
        self.input_conv = nn.Conv2d(1, 4, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), bias=False)
        self.input_bn = nn.BatchNorm2d(4)
        self.input_relu = nn.ReLU(inplace=True)
        
        # RSBU-NTF层
        self.layer1 = RSBU_NTF(4, 4, kernel_size=3, stride=2)
        self.layer2 = RSBU_NTF(4, 4, kernel_size=3, stride=1)
        self.layer3 = RSBU_NTF(4, 8, kernel_size=3, stride=2)
        self.layer4 = RSBU_NTF(8, 8, kernel_size=3, stride=1)
        self.layer5 = RSBU_NTF(8, 16, kernel_size=3, stride=2)
        self.layer6 = RSBU_NTF(16, 16, kernel_size=3, stride=1)
        
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 全连接层
        self.fc = nn.Linear(16, num_classes)
    
    def forward(self, x):
        # 输入层
        out = self.input_conv(x)
        out = self.input_bn(out)
        out = self.input_relu(out)
        
        # RSBU-NTF层
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        
        # 全局平均池化和全连接
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out

# 测试模型
def test_model():
    model = DRSN_NTF(num_classes=9)
    x = torch.randn(2, 1, 8000, 1)  # 批次大小2，1通道，8000宽度，1高度
    y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print("模型测试通过!")

if __name__ == "__main__":
    test_model()
