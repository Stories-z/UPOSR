import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()

        loss_function = nn.L1Loss()
        self.loss.append({
            'type': 'L1',
            'function': loss_function}
        )        
        module = import_module('loss.vgg')
        loss_function = getattr(module, 'VGG')(
            '22',
            rgb_range=args.rgb_range
        )
        self.loss.append({
            'type': 'VGG',
            'function': loss_function}
        )      
        module = import_module('loss.adversarial')
        loss_function = getattr(module, 'Adversarial')(
            args,
            'GAN'
        )            
        self.loss.append({
            'type': 'GAN',
            'function': loss_function}
        )      
        self.loss.append({'type': 'DIS', 'function': None})
        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{}'.format( l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half': self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )

        if args.load != '': self.load(ckp.dir, cpu=args.cpu)

    def forward(self, sr, hr, alpha):
        losses = []

        loss = self.loss[0]['function'](sr, hr)
        effective_loss = alpha * loss # L1 loss
        losses.append(effective_loss)
        self.log[-1, 0] += effective_loss.item()

        loss = self.loss[1]['function'](sr, hr)
        effective_loss = alpha * loss # VGG loss
        losses.append(effective_loss)
        self.log[-1, 1] += effective_loss.item()

        loss = self.loss[2]['function'](sr, hr)
        effective_loss = (1 - alpha) * 10 * loss # GAN loss
        losses.append(effective_loss)
        self.log[-1, 2] += effective_loss.item()

        self.log[-1, 3] += self.loss[2]['function'].loss
        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type'])))
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()

