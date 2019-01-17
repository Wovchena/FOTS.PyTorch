import argparse
import json
import logging
import os
import pathlib
import torch

from FOTS.data_loader import SynthTextDataLoaderFactory
from FOTS.data_loader import OCRDataLoaderFactory
from FOTS.data_loader import ICDAR
from FOTS.data_loader.datautils import collate_fn
from FOTS.trainer import Trainer, fots_metrics
from FOTS.utils.bbox import Toolbox

logging.basicConfig(level=logging.DEBUG, format='')


def main(config):
    data_root = "C:\\Users\\vzlobin\\Documents\\repo\\FOTS.PyTorch\\data"
    bs = 10
    train_ICDARDataset2015 = ICDAR(os.path.join(data_root, 'icdar/icdar2015/4.4/training'), year='2015', type='training')
    test_ICDARDataset2015 = ICDAR(os.path.join(data_root, 'icdar/icdar2015/4.4/training'), year='2015', type='test')
    train_ICDARDataset2013 = ICDAR(os.path.join(data_root, 'icdar/icdar2013'), year='2013')  # no gt for test icdar2013
    train_ds = train_ICDARDataset2015 + train_ICDARDataset2013
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True, sampler=None, batch_sampler=None,
                                           num_workers=4, collate_fn=collate_fn, pin_memory=False, drop_last=False,
                                           timeout=0, worker_init_fn=None)
    test_dl = torch.utils.data.DataLoader(test_ICDARDataset2015, batch_size=bs * 2, shuffle=False, sampler=None, batch_sampler=None,
                                           num_workers=4, collate_fn=collate_fn, pin_memory=False, drop_last=False,
                                           timeout=0, worker_init_fn=None)
    # # elif config['data_loader']['dataset'] == 'synth800k':
    # data_loader = SynthTextDataLoaderFactory(config)
    # train = data_loader.train()
    # val = data_loader.val()

    Trainer(metrics=fots_metrics,
            config=config,
            data_loader=train_dl,
            valid_data_loader=test_dl,
            toolbox=Toolbox).train()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')

    args = parser.parse_args()

    config = None
    if args.config is not None:
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    assert config is not None

    main(config)
