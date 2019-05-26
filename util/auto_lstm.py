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
        # Initialize hidden state with zeros
        # Initialize cell state
        # enc_h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        # enc_c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        # dec_h0 = torch.zeros(1, x.size(0), 3).requires_grad_()
        # dec_c0 = torch.zeros(1, x.size(0), 3).requires_grad_()

        # print(x[0])
        # print(x[0].shape)
        # print(x.view(-1, 128, 3).shape)
        # print(x.view(-1, 128, 3)[:])
        # out, _ = self.encoder(x, (enc_h0.detach(), enc_c0.detach()))
        out, _ = self.encoder(x)
        # out, _ = self.encoder(x.view(-1, 128, 3))
        x_enc = out[:, -1, :]
        # print(x_enc.shape)
        # print(x_enc)
        # x_res = x_enc.repeat(1, 128).view(-1, 128, 16)
        # print(x_enc.shape)
        # print(x_enc[0])
        # print(x_enc[1])
        # x_res = x_enc.repeat(1, 128, 1)
        # x_res = x_enc.repeat(128, 1, 1)
        # x_res = out
        x_res = x_enc.repeat(1, 128).view(-1, 128, 16)
        # x_enc = out[:, -1, :]
        # print(x_enc.shape)
        # x_res = out
        # print(x_res[0])
        # print(x_res[1])
        # print(x_res.shape)
        out, _ = self.decoder(x_res)
        # out, _ = self.decoder(x_res)
        # print(out[:, -1, :].shape)
        # print(out.shape)
        # print(out)

        # return out.reshape(-1, 384), x_enc
        return out, x_enc