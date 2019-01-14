import os
import math
import logging
import torch
from tensorboardX import SummaryWriter

from ..model.model import FOTSModel


def save_checkpoint(epoch, model, optimizer, lr_scheduler, best_score, folder, save_as_best):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if save_as_best:
        torch.save(model.state_dict(), os.path.join(folder, 'best_model.pt'))
        print('Updated best_model')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'best_score': best_score  # not current score
    }, os.path.join(folder, 'last_checkpoint.pt'))


def restore_checkpoint(folder):
    model = FOTSModel().to(torch.device("cuda"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5,
                                 amsgrad=False)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.94)

    if os.path.isfile(os.path.join(folder, 'last_checkpoint.pt')):
        checkpoint = torch.load(os.path.join(folder, 'last_checkpoint.pt'))
        epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        best_score = checkpoint['best_score']
        return epoch, model, optimizer, lr_scheduler, best_score
    else:
        return 0, model, optimizer, lr_scheduler, -math.inf


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, metrics, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics = metrics
        self.name = config['name']
        self.epochs = config['trainer']['epochs']
        self.summyWriter = SummaryWriter()
        self.checkpoint_dir = os.path.join(config['trainer']['save_dir'], self.name)

        self.start_epoch, self.model, self.optimizer, self.lr_scheduler, self.best_score = \
            restore_checkpoint(self.checkpoint_dir)
        # self.model.summary()

        self.device = torch.device("cuda")
        self.monitor = config['trainer']['monitor']

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    for i, metric in enumerate(self.metrics):
                        log[metric.__name__] = result['metrics'][i]
                elif key == 'val_metrics':
                    for i, metric in enumerate(self.metrics):
                        log['val_' + metric.__name__] = result['val_metrics'][i]
                else:
                    log[key] = value

            self.lr_scheduler.step()
            self.logger.info('New Learning Rate: {:.8f}'.format(self.lr_scheduler.get_lr()[0]))

            self.summyWriter.add_scalars('Train', {'train_' + self.monitor: result[self.monitor],
                                                   'val_' + self.monitor: result[self.monitor]}, epoch)

            if self.best_score < result['hmean']:
                self.best_score = result['hmean']
                save_as_best = True
            else:
                save_as_best = False
            save_checkpoint(epoch, self.model, self.optimizer, self.lr_scheduler, self.best_score,
                            self.checkpoint_dir, save_as_best)

        self.summyWriter.close()


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError
