import torch
import torch.nn as nn
from torch.autograd import Variable as V

class LSTMModel(nn.Module):
    def __init__(self, no_outputs=181, hidden_size=512, num_layers=3):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.features = nn.Sequential(
            nn.Conv2d(3, 8, (5, 5), padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(8, 16, (5, 5), padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, (5, 5), padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, (3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.lstm = nn.LSTM(
            input_size=3712,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2
        )

        self.acc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, no_outputs)
        )

        self.brake = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, no_outputs)
        )

        self.steer = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, no_outputs)
        )

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        state = self._init_state(b_size=batch_size)
        convs = []
        for t in range(timesteps):
            conv = self.features(x[:, t, :, :, :])
            conv = conv.view(batch_size, -1)
            convs.append(conv)
        convs = torch.stack(convs, 0)
        lstm, _ = self.lstm(convs, state)

        return self.acc(lstm[-1]), self.brake(lstm[-1]), self.steer(lstm[-1])
        '''
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.features(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.lstm(r_in)
        #r_out2 = self.fc(r_out[:, -1, :])
        return self.acc(r_out[:, -1, :]), self.brake(r_out[:, -1, :]), self.steer(r_out[:, -1, :])
        '''

    def _init_state(self, b_size=1):
        weight = next(self.parameters()).data
        return (
            V(weight.new(self.num_layers, b_size, self.hidden_size).normal_(0.0, 0.01)).cuda(),
            V(weight.new(self.num_layers, b_size, self.hidden_size).normal_(0.0, 0.01)).cuda()
        )