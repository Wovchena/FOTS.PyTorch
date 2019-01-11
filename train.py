import argparse
import json
import logging
import os
import pathlib

from FOTS.data_loader import SynthTextDataLoaderFactory
from FOTS.data_loader import OCRDataLoaderFactory
from FOTS.data_loader import ICDAR
from FOTS.trainer import Trainer, fots_metrics
from FOTS.utils.bbox import Toolbox

logging.basicConfig(level=logging.DEBUG, format='')


def main(config):
    if config['data_loader']['dataset'] == 'icdar2015':
        # ICDAR 2015
        data_root = pathlib.Path(config['data_loader']['data_dir'])
        ICDARDataset2015 = ICDAR(data_root, year='2015')
        data_loader = OCRDataLoaderFactory(config, ICDARDataset2015)
        train = data_loader.train()
        val = data_loader.val()
    elif config['data_loader']['dataset'] == 'synth800k':
        data_loader = SynthTextDataLoaderFactory(config)
        train = data_loader.train()
        val = data_loader.val()

    metrics = fots_metrics

    trainer = Trainer(metrics,
                      config=config,
                      data_loader=train,
                      valid_data_loader=val,
                      toolbox=Toolbox)

    trainer.train()


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
