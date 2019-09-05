import torch
from torch import nn

from .convlstm import ConvLSTM


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.convlstm1 = ConvLSTM(
            input_size=(args.height, args.width), input_dim=args.channels,
            hidden_dim=[128, 64, 64], kernel_size=(5, 5), num_layers=3,
            batch_first=True, bias=True, return_all_layers=True)

    def forward(self, x):
        out1, hidden1 = self.convlstm1(x)
        return out1[-1], [hidden1]


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.convlstm1 = ConvLSTM(
            input_size=(args.height, args.width), input_dim=64,
            hidden_dim=[128, 64, 64], kernel_size=(5, 5), num_layers=3,
            batch_first=True, bias=True, return_all_layers=True)

    def forward(self, x, hidden_list=None):
        out1, hidden1 = self.convlstm1(x, hidden_list[0])
        return out1


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.conv1x1 = nn.Conv2d(
            128 + 64 + 64, args.channels, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out_e, hidden_list = self.encoder(x)
        out1 = self.decoder(out_e, hidden_list)
        out_d = torch.cat(out1, dim=2)
        bs, t, c, h, w = out_d.size()
        out = self.conv1x1(out_d.view(bs * t, c, h, w))
        return out.view(bs, t, -1, h, w)


def convlstm_3_layers(args):
    return Model(args)
