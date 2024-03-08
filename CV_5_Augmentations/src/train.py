import argparse
import logging
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pytorch_lightning as pl
from clearml import Task
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from src.config import Config
from src.constants import EXPERIMENTS_PATH
from src.datamodule import BitterflyDM
from src.lightning_module import ButterflyModule
from src.callbacks.experiment_tracking import LogConfusionMatrix, SaveMetricsCallback
from src.callbacks.debug import DisplayDebugImagesCallback


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    parser.add_argument('--aug_yaml', type=str, default=None, help='aug file')
    return parser.parse_args()


def train(config: Config, aug_yaml: str):
    datamodule = BitterflyDM(config.data_config, aug_yaml)
    model = ButterflyModule(config, datamodule.label_names)

    task = Task.init(
        project_name=config.project_name,
        task_name=f'{config.experiment_name}',
        auto_connect_frameworks=True,
    )
    task.connect(config.dict())

    experiment_save_path = os.path.join(EXPERIMENTS_PATH, config.experiment_name)
    os.makedirs(experiment_save_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        experiment_save_path,
        monitor=config.monitor_metric,
        mode=config.monitor_mode,
        save_top_k=1,
        filename=f'epoch_{{epoch:02d}}-{{{config.monitor_metric}:.3f}}',
    )
    
    # save_metrics_callback = SaveMetricsCallback(experiment_save_path,  f'{config.experiment_name}.csv')
    
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator=config.accelerator,
        devices=[config.device],
        log_every_n_steps=10,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor=config.monitor_metric, patience=20, mode=config.monitor_mode),
            LearningRateMonitor(logging_interval='epoch'),
            DisplayDebugImagesCallback(frequency=5),
            # save_metrics_callback,
            LogConfusionMatrix()
        ],
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(ckpt_path=checkpoint_callback.best_model_path, datamodule=datamodule)


if __name__ == '__main__':
    args = arg_parse()
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(100, workers=True)
    config = Config.from_yaml(args.config_file)
    augs = args.aug_yaml
    train(config, augs)
