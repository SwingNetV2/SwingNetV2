import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class EventDetector(nn.Module):
    def __init__(self, lstm_layers, lstm_hidden, bidirectional=True, dropout=True):
        super(EventDetector, self).__init__()
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout

        weights = MobileNet_V2_Weights.IMAGENET1K_V2
        net = mobilenet_v2(weights=weights)

        transform = weights.transforms()
        self.cnn_out_dim = 1280


        self.cnn = nn.Sequential(*list(net.children())[0][:19])
        self.lstm = nn.LSTM(self.cnn_out_dim,
                           self.lstm_hidden, self.lstm_layers,
                           batch_first=True, bidirectional=bidirectional)
        if self.bidirectional:
            self.lin = nn.Linear(2*self.lstm_hidden, 9)
        else:
            self.lin = nn.Linear(self.lstm_hidden, 9)
        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def init_hidden(self, batch_size):
        if self.bidirectional:
            return (Variable(torch.zeros(2*self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True),
                    Variable(torch.zeros(2*self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True))
        else:
            return (Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True),
                    Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True))

    def forward(self, x, lengths=None):
        batch_size, timesteps, C, H, W = x.size()
        self.hidden = self.init_hidden(batch_size)

        # CNN forward
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        c_out = c_out.mean(3).mean(2)
        if self.dropout:
            c_out = self.drop(c_out)

        # LSTM forward
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, states = self.lstm(r_in, self.hidden)
        out = self.lin(r_out)
        out = out.view(batch_size*timesteps,9)

        return out
