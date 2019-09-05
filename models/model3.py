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
        self.convlstm2 = ConvLSTM(
            input_size=(args.height, args.width), input_dim=64,
            hidden_dim=[128, 64, 64], kernel_size=(5, 5), num_layers=3,
            batch_first=True, bias=True, return_all_layers=True)

    def forward(self, x):
        out1, hidden1 = self.convlstm1(x)
        out2, hidden2 = self.convlstm2(out1[-1])
        return out2[-1], [hidden1, hidden2]


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.convlstm1 = ConvLSTM(
            input_size=(args.height, args.width), input_dim=64,
            hidden_dim=[128, 64, 64], kernel_size=(5, 5), num_layers=3,
            batch_first=True, bias=True, return_all_layers=False)
        self.convlstm2 = ConvLSTM(
            input_size=(args.height, args.width), input_dim=64,
            hidden_dim=[128, 64, 64], kernel_size=(5, 5), num_layers=3,
            batch_first=True, bias=True, return_all_layers=False)

    def forward(self, x, hidden_list=None):
        out1, hidden1 = self.convlstm1(x, hidden_list[0])
        out2, hidden2 = self.convlstm2(out1[0], hidden_list[1])
        return out1[-1], out2[-1]


class Model3(nn.Module):
    def __init__(self, args):
        super(Model3, self).__init__()

        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.conv1x1 = nn.Conv2d(
            2 * 64, args.channels, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out_e, hidden_list = self.encoder(x)
        out1, out2 = self.decoder(out_e, hidden_list)
        out_d = torch.cat([out1, out2], dim=2)
        bs, t, c, h, w = out_d.size()
        out = self.conv1x1(out_d.view(bs * t, c, h, w))
        return out.view(bs, t, -1, h, w)


def model3(args):
    model3 = Model3(args)
    return model3
