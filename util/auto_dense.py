import torch.nn as nn

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