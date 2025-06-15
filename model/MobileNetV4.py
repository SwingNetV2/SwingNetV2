import torch
import torch.autograd.profiler as profiler
import torch.nn as nn
from torch.autograd          import Variable
import timm


class EventDetector_mb4(nn.Module):
    def __init__(self, pretrain=True, width_mult=1.,
                 lstm_layers=1, lstm_hidden=64,
                 bidirectional=True, dropout=True):
        super(EventDetector_mb4, self).__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.backbone = timm.create_model('mobilenetv4_conv_medium',
                                          pretrained=True, features_only=True)
        self.cnn_out_dim = self.backbone.feature_info[-1]['num_chs']  
        self.rnn = nn.LSTM(self.cnn_out_dim,
                           self.lstm_hidden, self.lstm_layers,
                           batch_first=True, bidirectional=bidirectional)

        out_dim = 2 * self.lstm_hidden if self.bidirectional else self.lstm_hidden
        self.lin = nn.Linear(out_dim, 9)

        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def init_hidden(self, batch_size):
        directions = 2 if self.bidirectional else 1
        return (Variable(torch.zeros(directions * self.lstm_layers,
                                     batch_size, self.lstm_hidden).cuda(),
                                     requires_grad=True),
                                        Variable(torch.zeros(directions * self.lstm_layers,
                                     batch_size, self.lstm_hidden).cuda(),
                                     requires_grad=True))

    def forward(self, x, lengths=None):
        batch_size, timesteps, C, H, W = x.size()
        self.hidden = self.init_hidden(batch_size)

        # CNN
        with profiler.record_function("CNN_PASS"):
            x = x.view(batch_size * timesteps, C, H, W)       # [B*T, 3, 160, 160]
            feats = self.backbone(x)[-1]                      # [B*T, 960, h, w]
            feats = feats.mean([2, 3])                        # GlobalAvgPool → [B*T, 960]
            if self.dropout:
                feats = self.drop(feats)

        # RNN
        with profiler.record_function("LSTM_PASS"):
            feats = feats.view(batch_size, timesteps, -1)     # [B, T, 960]
            r_out, states = self.rnn(feats, self.hidden)      # [B, T, hidden]
            out = self.lin(r_out)                             # [B, T, 9]
            out = out.view(batch_size * timesteps, 9)         # [B*T, 9]

        return out
