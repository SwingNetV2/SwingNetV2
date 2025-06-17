import torch
import torch.nn as nn
import torch.nn.functional as F

class EventDetector_clstm_lr(nn.Module):
    def __init__(self, n_conv=64, num_classes=9):
        super(EventDetector_clstm_lr, self).__init__()
        self.n_conv = n_conv
        
        # 모든 시간 프레임에 공유될 2D Conv 네트워크
        self.shared_conv = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(20, momentum=0.8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(20, 30, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(30, momentum=0.8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(30, 30, kernel_size=3, padding=1)
        )
        
        # 3D convolution
        self.conv3d = nn.Conv3d(30, 30, kernel_size=3, padding=1)
        self.batchnorm = nn.BatchNorm3d(30, momentum=0.8)
        
        # Bidirectional GRU
        self.gru1 = nn.GRU(30 * 40 * 40, 256, batch_first=True, bidirectional=True)
        
        # 최종 분류기
        self.fc1 = nn.Linear(512, num_classes)
        
        # GRU 가중치 초기화
        self._initialize_gru_weights()
    
    def _initialize_gru_weights(self):
        """GRU 가중치를 Xavier 초기화로 설정"""
        for name, param in self.gru1.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, x):
        # x: [B, T, 3, H, W]
        B, T, C, H, W = x.shape
        assert T == self.n_conv, f"Expected {self.n_conv} frames, got {T}"
        
        # 모든 프레임을 배치로 한번에 처리
        x_reshaped = x.view(B * T, C, H, W)  # [B*T, 3, H, W]
        
        # 단일 forward pass로 모든 프레임 처리
        conv_out = self.shared_conv(x_reshaped)  # [B*T, 30, H/4, W/4]
        
        # 시간 차원 복원
        _, C_out, H_out, W_out = conv_out.shape
        conv_out = conv_out.view(B, T, C_out, H_out, W_out)  # [B, T, 30, H/4, W/4]
        
        # 3D Conv를 위한 차원 변경: [B, 30, T, H/4, W/4]
        t = conv_out.permute(0, 2, 1, 3, 4).contiguous()
        
        # 3D convolution + batch normalization
        t = self.conv3d(t)
        t = self.batchnorm(t)
        
        # 다시 원래 차원으로: [B, T, 30, H/4, W/4]
        t = t.permute(0, 2, 1, 3, 4).contiguous()
        
        # GRU 입력을 위한 차원 변경
        B, T, C3, H3, W3 = t.size()
        feat = t.view(B, T, C3 * H3 * W3)  # [B, T, features]
        
        # GRU 처리
        out, _ = self.gru1(feat)  # [B, T, 512]
        
        # 최종 분류
        out = out.contiguous().view(B * T, 512)
        out = self.fc1(out)  # [B*T, num_classes]
        
        return out
