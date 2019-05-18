import torch
import torch.nn as nn


# http://www.jessicayung.com/lstms-for-time-series-in-pytorch/
# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/
class LSTMAutoencoder(nn.Module):
    def __init__(self, encode_dim, activation='tanh'):
        super(LSTMAutoencoder, self).__init__()

        self.hidden_dim = 16
        self.layer_dim = 2

        self.encoder = nn.LSTM(input_size=128, hidden_size=self.hidden_dim,
                               num_layers=2, dropout=0.4, batch_first=True)

        self.decoder = nn.LSTM(input_size=self.hidden_dim, hidden_size=384, 
                               batch_first=True)

    def forward(self, x):
        # Initialize hidden state with zeros
        # Initialize cell state
        enc_h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        enc_c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        dec_h0 = torch.zeros(1, x.size(0), 384).requires_grad_()
        dec_c0 = torch.zeros(1, x.size(0), 384).requires_grad_()

        # print(x.view(-1, 3, 128).shape)
        out, _ = self.encoder(x.view(-1, 3, 128), (enc_h0.detach(), enc_c0.detach()))
        x_enc = out[:, -1, :]
        # print(x_enc.shape)
        x_res = x_enc.repeat(1, 128).view(-1, 128, 16)
        # print(x_res.shape)
        out, _ = self.decoder(x_res, (dec_h0.detach(), dec_c0.detach()))
        # print(out[:, -1, :].shape)

        return out[:, -1, :], x_enc