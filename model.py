import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    一维特征图的通道注意力模块
    通过平均池化和最大池化生成通道权重，用于增强重要特征通道
    """
    
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    一维特征图的空间注意力模块
    用于学习序列维度上的重要位置权重
    """
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    """
    卷积块注意力模块
    """
    
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class Channel(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(Channel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio),
            nn.ReLU(),
            nn.Linear(in_planes // ratio, in_planes)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.fc(x)
        return self.sigmoid(attn)

class Spatial(nn.Module):
    def __init__(self, in_dim):
        super(Spatial, self).__init__()
        self.fc = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.fc(x)
        return self.sigmoid(attn)

class CBAM1(nn.Module):
    """
    用于空间分支的CBAM结构
    """
    
    def __init__(self, in_planes):
        super(CBAM1, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention(in_planes)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class Attention(nn.Module):
    """
    时间注意力池化模块
    """
    
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        weights = self.attention(x)
        weights = F.softmax(weights, dim=1)
        output = torch.sum(x * weights, dim=1)
        return output


class SpectralBranch(nn.Module):
    """
    光谱特征提取分支
    该分支结合卷积、双向LSTM、CBAM注意力和时间注意力池化
    用于提取月尺度光谱时间序列特征
    """
    
    def __init__(self, input_dim=12, hidden_dims=256, num_layers=2, bidirectional=True, dropout=0.5):
        super(SpectralBranch, self).__init__()
        self.tcn = nn.Conv1d(input_dim, hidden_dims, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(hidden_dims)  # 添加BN
        self.relu = nn.ReLU(inplace=False)  # 添加ReLU
        self.lstm = nn.LSTM(input_size=hidden_dims, hidden_size=hidden_dims, num_layers=num_layers,
                            bias=True, batch_first=True, bidirectional=bidirectional)
        lstm_output_dim = hidden_dims * (2 if bidirectional else 1)
        self.cbam = CBAM(lstm_output_dim)  # 使用CBAM模块

        spectral_bilstm = hidden_dims * (2 if bidirectional else 1)
        self.spectral_bilstm = nn.LSTM(input_size=hidden_dims * 2, hidden_size=hidden_dims, num_layers=num_layers,
                                       bias=True, batch_first=True, bidirectional=bidirectional)
        self.attention = Attention(spectral_bilstm)
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Linear(hidden_dims * 2, 128)  # 输出为 512 → 256

    def forward(self, x):
        x = self.tcn(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)
        x = self.cbam(x)

        x = x.transpose(1, 2)
        x, _ = self.spectral_bilstm(x)
        x = self.attention(x)
        x = self.project(x)
        return x


class SpatialBranch(nn.Module):
    """
    空间纹理特征提取分支
    该分支利用一维卷积和CBAM注意力提取空间纹理特征表征
    """
    
    def __init__(self, input_dim=3, hidden_dims=128, kernel_size=3, num_init_features=128, dropout=0.5):
        super(SpatialBranch, self).__init__()

        self.tcn = nn.Conv1d(input_dim, hidden_dims, kernel_size=kernel_size, padding=1)
        self.bn = nn.BatchNorm1d(hidden_dims)

        self.conv1 = nn.Conv1d(input_dim, hidden_dims, kernel_size=kernel_size, padding=1)
        self.cbam1 = CBAM(hidden_dims)

        self.conv2 = nn.Conv1d(hidden_dims, hidden_dims, kernel_size=kernel_size, padding=1)
        self.cbam2 = CBAM(hidden_dims)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(hidden_dims, num_init_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.cbam1(x)

        x = self.conv2(x)
        x = self.cbam2(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SARBranch(nn.Module):
    """
    SAR特征提取分支
    该分支结合卷积、双向LSTM、CBAM注意力和时间注意力池化
    用于提取月尺度SAR时间序列特征
    """
    
    def __init__(self, input_dim=2, hidden_dims=128, num_layers=2, bidirectional=True, dropout=0.5):
        super(SARBranch, self).__init__()

        self.tcn = nn.Conv1d(input_dim, hidden_dims, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(hidden_dims)
        self.relu = nn.ReLU(inplace=False)
        self.lstm = nn.LSTM(input_size=hidden_dims, hidden_size=hidden_dims, num_layers=num_layers,
                            bias=True, batch_first=True, bidirectional=bidirectional)
        lstm_output_dim = hidden_dims * (2 if bidirectional else 1)
        self.cbam = CBAM(lstm_output_dim)

        sar_bilstm = hidden_dims * (2 if bidirectional else 1)
        self.sar_bilstm = nn.LSTM(input_size=hidden_dims * 2, hidden_size=hidden_dims, num_layers=num_layers,
                                  bias=True, batch_first=True, bidirectional=bidirectional)
        self.attention = Attention(sar_bilstm)
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Linear(hidden_dims * 2, 128)

    def forward(self, x):
        x = self.tcn(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)
        x = self.cbam(x)

        x = x.transpose(1, 2)
        x, _ = self.sar_bilstm(x)
        x = self.attention(x)
        x = self.project(x)
        return x


class phenology_branch(nn.Module):
    """
    物候特征提取分支
    """
    
    def __init__(self, input_dim=5, output_dim=128, dropout=0.5):
        super(phenology_branch, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class Model(nn.Module):
    def __init__(self, spectral_dim=12, spatial_dim=3, sar_dim=2, phenology_dim=5, hidden_dims=256, num_layers=2, bidirectional=True,
                 num_classes=3, dropout=0.5):
        super(Model, self).__init__()
        self.spectral_branch = SpectralBranch(input_dim=spectral_dim, hidden_dims=hidden_dims, num_layers=num_layers,
                                              bidirectional=bidirectional)

        self.spatial_branch = SpatialBranch(input_dim=spatial_dim, hidden_dims=hidden_dims)

        self.sar_branch = SARBranch(input_dim=sar_dim, hidden_dims=hidden_dims, num_layers=num_layers,
                                    bidirectional=bidirectional)

        self.phe_branch = phenology_branch(input_dim=phenology_dim)

        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)

        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, spectral_data, spatial_data, sar_data, phe_data):
        spectral_data = spectral_data.transpose(1, 2)
        spatial_data = spatial_data.transpose(1, 2)
        sar_data = sar_data.transpose(1, 2)

        spectral_output = self.spectral_branch(spectral_data)
        sar_output = self.sar_branch(sar_data)
        spatial_output = self.spatial_branch(spatial_data)
        phenology_output = self.phe_branch(phe_data)

        tokens = torch.stack([spectral_output, spatial_output, sar_output, phenology_output], dim=1)

        attn_out, _ = self.attention(tokens, tokens, tokens)
        attn_out = attn_out.mean(dim=1)

        x = self.dropout(attn_out)
        logits = self.fc(x)
        return logits
