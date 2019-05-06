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
from util.process import get_csv


class Autoencoder(nn.Module):
    def __init__(self, input_size, encode_dim):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, encode_dim[0]),
            nn.ReLU(True),
            nn.Linear(encode_dim[0], encode_dim[1]),
            nn.ReLU(True))

        self.decoder = nn.Sequential(             
            nn.Linear(encode_dim[1], encode_dim[0]),
            nn.ReLU(True),
            nn.Linear(encode_dim[0], input_size),
            nn.Sigmoid())

    def forward(self, x):
        x_enc = self.encoder(x)
        x_dec = self.decoder(x_enc)
        return x_dec, x_enc


# TODO: add raw support
class GaitDataset(Dataset):
    def __init__(self, filename):
        df = pd.read_csv(Path(FEAT_PATH).joinpath(Path(filename)), header=None)
        y = df[df.columns[-1]].values
        df.drop([df.columns[-1]], axis=1, inplace=True)
        
        self.Xdata = df
        self.Ydata = None
        
    def __len__(self):
        return len(self.Xdata)
    
    def __getitem__(self, index):
        vector = self.Xdata.iloc[index, :].values.astype(np.float32)
        return vector, 0


class TransDataset(Dataset):
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
    def __init__(self, input_size, encode_dim):
        self.model = Autoencoder(input_size, encode_dim).cpu()
        self.distance = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = 1, momentum = 0.9)
        train_dataset = GaitDataset(get_csv(1, 128))
        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        self._train_feat_ext()

    def _train_feat_ext(self):
        for epoch in range(FEAT_EPOCHS):
            self._train(epoch)

        print('FEAT: Done training!')

    def _train(self, epoch):
        for data in self.train_loader:
            vec, _ = data
            vec = Variable(vec, requires_grad=True).cpu()
            
            dec, enc = self.model(vec)
            loss = self.distance(dec, vec)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        print('epoch [{}/{}], loss: {:.4f}'.format(epoch + 1, FEAT_EPOCHS, loss.item()))

    def to_latent(self, df):
        loader = DataLoader(TransDataset(df), batch_size=BATCH_SIZE)

        self.model.eval()
        encs = torch.Tensor([])
        labels = []
        for data in loader:
            vec, label = data
            vec = Variable(vec, requires_grad=False).cpu()
            _, enc = self.model(vec)
            
            encs = torch.cat((encs, enc))
            labels.extend([l.tolist() for l in label])

        latent_df = pd.DataFrame(data=np.c_[encs.detach().numpy(), labels])
            
        return latent_df