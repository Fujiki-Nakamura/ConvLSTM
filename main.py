import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm  # noqa

from MovingMNIST import MovingMNIST
import models
from utils import get_logger, get_optimizer, save_checkpoint
from utils import get_scheduler, get_loss_fn


def main(args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    os.chmod(args.log_dir, 0o0777)
    logger = get_logger(os.path.join(args.log_dir, 'main.log'))
    logger.info(args)

    writer = SummaryWriter(args.log_dir)

    # data
    train_set = MovingMNIST(root='./data/train', train=True, download=True)
    test_set = MovingMNIST(root='./data/test', train=False, download=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # network
    model = models.__dict__[args.model](args=args)
    model = model.to(args.device)

    # training
    criterion = get_loss_fn(args)
    mse_loss_fn = nn.MSELoss(reduction='mean')
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    best_loss = 1e+6
    for epoch_i in range(1, 1 + args.epochs):
        model.train()
        losses = 0.
        mse_losses = 0.
        for i, (inputs, targets) in enumerate(train_loader):
            bs, ts, h, w = targets.size()
            inputs = inputs.unsqueeze(2)
            inputs, targets = inputs.float() / 255., targets.float() / 255.
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)

            # (bs ,ts, c, h, w) -> (bs, ts, h, w) -> (ts, bs, h, w)
            outputs = outputs.squeeze(2).permute(1, 0, 2, 3)
            # (bs, ts, h, w) -> (ts, bs, h, w)
            targets = targets.permute(1, 0, 2, 3)
            loss = 0.
            mse_loss = 0.
            for t_i in range(ts):
                loss += criterion(outputs[t_i], targets[t_i]) / bs
                mse_loss += mse_loss_fn(outputs[t_i], targets[t_i])

            losses += loss.item() * bs
            mse_losses += mse_loss.item() * bs

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.debug('Train/Batch {}/{}'.format(i + 1, len(train_loader)))

        model.eval()
        test_losses = 0.
        test_mse_losses = 0.
        for i, (inputs, targets) in enumerate(test_loader):
            bs, ts, h, w = targets.size()
            inputs = inputs.unsqueeze(2)
            inputs, targets = inputs.float() / 255., targets.float() / 255.
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            with torch.no_grad():
                outputs = model(inputs)
                # (bs ,ts, c, h, w) -> (bs, ts, h, w) -> (ts, bs, h, w)
                outputs = outputs.squeeze(2).permute(1, 0, 2, 3)
                # (bs, ts, h, w) -> (ts, bs, h, w)
                targets = targets.permute(1, 0, 2, 3)
                loss = 0.
                mse_loss = 0.
                for t_i in range(ts):
                    loss += criterion(outputs[t_i], targets[t_i]) / bs
                    mse_loss += mse_loss_fn(outputs[t_i], targets[t_i])
            test_losses += loss.item() * bs
            test_mse_losses += mse_loss.item() * bs
            logger.debug('Test/Batch {}/{}'.format(i + 1, len(test_loader)))

        test_loss = test_losses / len(test_set)
        is_best = test_loss < best_loss
        if test_loss < best_loss:
            best_loss = test_loss
        save_checkpoint({
            'epoch': epoch_i,
            'state_dict': model.state_dict(),
            'test_loss': test_loss,
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.log_dir)

        if scheduler is not None:
            scheduler.step()

        train_loss = losses / len(train_set)
        train_mse_loss = mse_losses / len(train_set)
        test_mse_loss = test_mse_losses / len(test_set)
        writer.add_scalar('Train/BCE', train_loss, epoch_i)
        writer.add_scalar('Test/BCE', test_loss, epoch_i)
        writer.add_scalar('Train/Loss', train_mse_loss, epoch_i)
        writer.add_scalar('Test/Loss', test_mse_loss, epoch_i)

        logger.info('Epoch {} Train/Loss {:.4f} Test/Loss {:.4f}'.format(
            epoch_i,
            losses / len(train_set), test_loss))


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
    parser.add_argument('--loss', type=str, default='mseloss')
    parser.add_argument('--reduction', type=str, default='mean')
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

    args, _ = parser.parse_known_args()
    main(args)
