import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from util.const import *
from util.auto_dense import DenseAutoencoder
from util.auto_lstm import LSTMAutoencoder

# https://www.oreilly.com/library/view/natural-language-processing/9781491978221/ch04.html
# https://towardsdatascience.com/pytorch-how-and-when-to-use-module-sequential-modulelist-and-moduledict-7a54597b5f17


##
#  Class implementing torch.utils.data.Dataset specifically for a session of
#  the ZJU-GaitAcc dataset
#
class GaitDataset(Dataset):
    def __init__(self, df, is_lstm):
        y = df[df.columns[-1]].values
        t_df = df.drop([df.columns[-1]], axis=1)
        
        if is_lstm == True:
            self.Xdata = t_df.values.astype(np.float32).reshape(-1, 128, 3, order='F')
        else:
            self.Xdata = t_df.values.astype(np.float32)
        self.Ydata = y

    def __len__(self):
        return len(self.Xdata)
    
    def __getitem__(self, index):
        vector = self.Xdata[index, :]
        label  = self.Ydata[index]

        return vector, label

##
#  Class for training autoencoders and transforming dataframes to latent space
#
class FeatureExtractor():
    def __init__(self, train_df, feat_ext, epochs, same_day, activation='relu'):
        if feat_ext == 'dense':
            self.model = DenseAutoencoder([384, 256, 128, 64, 32], activation).cpu()
        elif feat_ext == 'lstm':
            self.model = LSTMAutoencoder(128).cpu()

        self.is_lstm = (feat_ext == 'lstm')
        self.epochs = epochs

        self.distance = nn.MSELoss()
        lr = 1e-3
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.optimizer = torch.optim.Adadelta(self.model.parameters())
        dataset = GaitDataset(train_df, self.is_lstm)
        self.train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        self._train_feat_ext()

    ##
    #  Trains the autoencoder for specified number of training epochs
    #
    def _train_feat_ext(self):
        for epoch in range(self.epochs):
            self._train(epoch)

        print('FEAT: Done training!')

    ##
    #  Runs one epoch of batch training
    #
    def _train(self, epoch):
        for data in self.train_loader:
            vec, labels = data
            vec = Variable(vec, requires_grad=True).cpu()

            dec, enc = self.model(vec)
            loss = self.distance(dec, vec)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        print('epoch [{}/{}], loss: {:.4f}'.format(epoch + 1, self.epochs, loss.item()))

    ##
    #  Converts the given dataframe with the trained autoencoder to latent space
    #
    def to_latent(self, df):
        loader = DataLoader(GaitDataset(df, self.is_lstm), batch_size=BATCH_SIZE)

        with torch.no_grad():
            encs = torch.Tensor([])
            labels = []
            for data in loader:
                vec, label = data
                vec = Variable(vec, requires_grad=False).cpu()
                curr_batch = vec.shape[0]
                _, enc = self.model(vec)

                encs = torch.cat((encs, enc.squeeze()))
                labels.extend([l.tolist() for l in label])

            latent_df = pd.DataFrame(data=np.c_[encs.detach().numpy(), labels])
        
        return latent_df
