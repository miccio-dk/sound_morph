import warnings
warnings.simplefilter("ignore", UserWarning)

import os
import json
import numpy as np
import librosa as lr
import torch
import pytorch_lightning as pl

from argparse import ArgumentParser
from pprint import pprint
from scipy.signal.windows import hann
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

from datasets.urbansounds_datamodule import UrbanSoundsModule
from models.cvae_resnet import CvaeResnet
from models.cvae_inception import CvaeInception
from models.vae_inception import VaeInception


def main(args):
    pl.seed_everything(42)
    
    # load configs
    with open(args.cfg_path, 'r') as fp:
        cfg = json.load(fp)
    cfg_train = cfg['train']
    print('### TRAIN CONFIGS:')
    pprint(cfg_train)
    print('### MODEL CONFIGS:')
    pprint(cfg['model'])
    #os.environ['CUDA_VISIBLE_DEVICES'] = cfg_train['trainer_kwargs']['gpus']
        
    # init model
    ModelClass = {
        'cvae': CvaeInception,
        'vae': VaeInception
    }[cfg_train['type']]
    model = ModelClass(cfg['model'])
    
    # init data loader
    dm = UrbanSoundsModule(
        cfg['dataset'], 
        num_workers=cfg_train['num_workers'], 
        batch_size=cfg_train['batch_size'])
    dm.setup()
    
    # logger
    log_name = 'usnds_{}_{}'.format(ModelClass.model_name, cfg_train['descr'])
    logger = TensorBoardLogger(save_dir='logs', name=log_name)
    
    # callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=cfg_train['patience'])
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # trainer
    trainer = pl.Trainer(
        max_epochs=cfg_train['max_epochs'],
        logger=logger,
        callbacks=[early_stop, lr_monitor],
        **cfg_train['trainer_kwargs'])
    
    # train
    trainer.fit(model=model, datamodule=dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('cfg_path', type=str)
    args = parser.parse_args()

    main(args)
