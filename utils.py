import os
import shutil
import torch
from torch import optim


def get_optimizer(model, args):
    if args.optim.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
    elif args.optim.lower() == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=args.lr, alpha=0.99, weight_decay=args.weight_decay)
    return optimizer


def get_scheduler(optimizer, args):
    if args.scheduler.lower() == 'multisteplr':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, args.milestones, args.gamma)
    else:
        return None
    return scheduler


def get_logger(log_file):
    from logging import getLogger, FileHandler, StreamHandler, Formatter, DEBUG, INFO, WARNING  # noqa
    fh = FileHandler(log_file)
    fh.setLevel(DEBUG)
    sh = StreamHandler()
    sh.setLevel(INFO)
    for handler in [fh, sh]:
        formatter = Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
    logger = getLogger('log')
    logger.setLevel(INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def save_checkpoint(state, is_best, log_dir):
    filename = os.path.join(log_dir, 'checkpoint.pt')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(log_dir, 'best.pt'))
