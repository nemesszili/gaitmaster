import torch
import torch.nn as nn
import numpy as np


# http://www.jessicayung.com/lstms-for-time-series-in-pytorch/
# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/
# https://discuss.pytorch.org/t/lstm-autoencoder-architecture/8524
class LSTMAutoencoder(nn.Module):
    def __init__(self, encode_dim):
        super(LSTMAutoencoder, self).__init__()

        self.hidden_dim = 16
        self.layer_dim = 2

        self.encoder = nn.LSTM(input_size=3, hidden_size=self.hidden_dim,
                               num_layers=self.layer_dim, dropout=0.4, 
                               batch_first=True)

        self.decoder = nn.LSTM(input_size=self.hidden_dim, hidden_size=3, 
                               batch_first=True)

        nn.init.xavier_uniform_(self.encoder.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.encoder.weight_hh_l0, gain=np.sqrt(2))

        nn.init.xavier_uniform_(self.decoder.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.decoder.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, x):
        out, _ = self.encoder(x)
        x_enc = out[:, -1, :]
        x_res = x_enc.repeat(1, 128).view(-1, 128, self.hidden_dim)
        out, _ = self.decoder(x_res)

        return out, x_enc