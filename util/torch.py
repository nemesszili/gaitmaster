import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from util.const import *
from util.process import get_feat_csv, get_raw_csv

# https://www.oreilly.com/library/view/natural-language-processing/9781491978221/ch04.html
# https://towardsdatascience.com/pytorch-how-and-when-to-use-module-sequential-modulelist-and-moduledict-7a54597b5f17
def dense_block(in_f, out_f, activation='relu'):
    activations = nn.ModuleDict([
        ['relu', nn.ReLU()]
    ])

    return nn.Sequential(
        nn.Linear(in_f, out_f),
        activations[activation]
    )

class DenseAutoencoder(nn.Module):
    def __init__(self, encode_dim, activation='relu'):
        super(DenseAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            *[dense_block(in_f, out_f, activation=activation) \
                for in_f, out_f in zip(encode_dim, encode_dim[1:])]
        )

        rev = encode_dim[::-1][:-1]
        self.decoder = nn.Sequential(
            *[dense_block(in_f, out_f, activation=activation) \
                for in_f, out_f in zip(rev, rev[1:])],
            nn.Linear(encode_dim[1], encode_dim[0])
        )

    def forward(self, x):
        x_enc = self.encoder(x)
        x_dec = self.decoder(x_enc)
        return x_dec, x_enc


def conv_block(in_f, out_f, forw, activation='relu'):
    activations = nn.ModuleDict([
        ['relu', nn.ReLU()]
    ])

    summary = nn.ModuleDict([
        ['maxpool', nn.MaxPool1d(2)],
        ['upsample', nn.Upsample(2)]
    ])

    return nn.Sequential(
        nn.Conv1d(in_channels=in_f, out_channels=out_f, kernel_size=1),
        # nn.Conv1d(in_channels=in_f, out_channels=out_f),
        activations[activation],
        summary[forw]
    )


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class CNN1DAutoencoder(nn.Module):
    def __init__(self, encode_dim, activation='relu'):
        super(CNN1DAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            *[conv_block(in_f, out_f, forw='maxpool', activation=activation)  \
                for in_f, out_f in zip(encode_dim, encode_dim[1:])],
        )

        rev = encode_dim[::-1]
        self.decoder = nn.Sequential(
            *[conv_block(in_f, out_f, forw='upsample', activation=activation) \
                for in_f, out_f in zip(rev, rev[1:])]
        )

    def forward(self, x):
        x_enc = self.encoder(x)
        x_dec = self.decoder(x_enc)
        return x_dec, x_enc


class GaitDataset(Dataset):
    def __init__(self, df):
        y = df[df.columns[-1]].values
        t_df = df.drop([df.columns[-1]], axis=1)
        
        self.Xdata = t_df
        self.Ydata = y

    def __len__(self):
        return len(self.Xdata)
    
    def __getitem__(self, index):
        vector = self.Xdata.iloc[index, :].values.astype(np.float32)
        label  = self.Ydata[index]

        return vector, label


class FeatureExtractor():
    def __init__(self, train_df, feat_ext, encode_dim, activation='relu'):
        if feat_ext == 'dense':
            self.model = DenseAutoencoder(encode_dim, activation).cpu()
        elif feat_ext == 'cnn':
            self.model = CNN1DAutoencoder(encode_dim, activation).cpu()

        print(self.model)
        
        self.distance = nn.MSELoss()
        lr = 1e-3
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr = lr, momentum = 0.9)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.optimizer = torch.optim.Adadelta(self.model.parameters())
        dataset = GaitDataset(train_df)
        self.train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        self._train_feat_ext()

    def _train_feat_ext(self):
        for epoch in range(FEAT_EPOCHS):
            self._train(epoch)

        print('FEAT: Done training!')

    def _train(self, epoch):
        for data in self.train_loader:
            vec, _ = data
            vec = vec.unsqueeze(2)
            # print(vec.shape)
            vec = Variable(vec, requires_grad=True).cpu()
            
            dec, enc = self.model(vec)
            loss = self.distance(dec, vec)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        print('epoch [{}/{}], loss: {:.4f}'.format(epoch + 1, FEAT_EPOCHS, loss.item()))

    def to_latent(self, df):
        loader = DataLoader(GaitDataset(df), batch_size=BATCH_SIZE)

        self.model.eval()
        encs = torch.Tensor([])
        labels = []
        for data in loader:
            vec, label = data
            vec = vec.unsqueeze(2)
            vec = Variable(vec, requires_grad=False).cpu()
            _, enc = self.model(vec)
            
            encs = torch.cat((encs, enc.squeeze()))
            labels.extend([l.tolist() for l in label])

        # encs = list(map(lambda x: x.squeeze(), encs))
        # encs = 
        # print(list(map(lambda x: x.shape, encs)))
        latent_df = pd.DataFrame(data=np.c_[encs.detach().numpy(), labels])
            
        return latent_df