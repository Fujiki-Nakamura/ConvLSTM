import argparse
from collections import OrderedDict
import os

import numpy as np

import torch
from torch.utils.data import DataLoader

from MovingMNIST import MovingMNIST
import models


def main(args):
    os.chmod(args.log_dir, 0o0777)

    test_set = MovingMNIST(root='./data/test', train=False, download=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    checkpoint = torch.load(args.checkpoint)
    new_state_dict = OrderedDict()
    for k, v in iter(checkpoint['state_dict'].items()):
        new_k = k.replace('module.', '')
        new_state_dict[new_k] = v
    model = models.__dict__[args.model](args)
    model.load_state_dict(new_state_dict)
    model.to(args.device)
    model.eval()

    inpts = np.zeros((len(test_set), 10, args.height, args.width))
    preds = np.zeros((len(test_set), 10, args.height, args.width))
    trues = np.zeros((len(test_set), 10, args.height, args.width))
    for batch_i, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.unsqueeze(2), targets.unsqueeze(2)
        inputs, targets = inputs.float() / 255., targets.float() / 255.
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        with torch.no_grad():
            outputs = model(inputs)
            outputs = outputs.squeeze(2).cpu().numpy()
            targets = targets.squeeze(2).cpu().numpy()
            inputs = inputs.squeeze(2).cpu().numpy()
            inpts[batch_i * args.batch_size:(batch_i + 1) * args.batch_size] = inputs
            preds[batch_i * args.batch_size:(batch_i + 1) * args.batch_size] = outputs
            trues[batch_i * args.batch_size:(batch_i + 1) * args.batch_size] = targets

    inpts.dump(os.path.join(args.log_dir, 'inpts.npy'))
    preds.dump(os.path.join(args.log_dir, 'preds.npy'))
    trues.dump(os.path.join(args.log_dir, 'trues.npy'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # network
    parser.add_argument('--model', type=str, default='convlstm_3_layers')
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--channels', type=int, default=1)
    # training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=int, default=0)
    # optim
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--betas', nargs='+', type=float, default=(0.9, 0.999))
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--scheduler', type=str, default='')
    parser.add_argument('--milestones', nargs='+', type=int, default=[30, ])
    parser.add_argument('--gamma', nargs='+', type=float, default=0.9)
    # misc
    parser.add_argument('--log_dir', type=str, default='./log')

    parser.add_argument('--checkpoint', type=str)

    args, _ = parser.parse_known_args()
    main(args)
