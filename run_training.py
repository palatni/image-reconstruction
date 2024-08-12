"""
this script is designed to run a training process for the image reconstruction.
Currently the training supports the model suggested in the article [1].

The training is performed on the regularly interspaced grid containing 1/4 of
the initial image pixels, as also suggested in the article [1].
The validation is performed on all other pixels.

[1] Tancik, Matthew, et al. "Fourier features let networks learn high
frequency functions in low dimensional domains." Advances in neural
information processing systems 33 (2020): 7537-7547.
"""
from typing import Dict, List
from argparse import ArgumentParser, Namespace
from pathlib import Path
from torch import optim
from image_reconstruction.trainer import ReconstructionTrainer
from image_reconstruction.models import FurierMLP
from image_reconstruction.loggers import (
    CSVLogger,
    VideoLogger,
    StateDictLogger,
    CurrentImgLogger
)
from utils import load_image, load_yaml


def _parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '-c',
        '--config',
        help='Specify a path to a config file',
        default='config.yaml',
    )
    return parser.parse_args()


def _init_loggers(config: Dict) -> List:
    loggers = []
    for logger_key, logger_type in zip(
        ['csv_metric', 'video_file', 'predicted_image', 'state_dict'],
        [CSVLogger, VideoLogger, CurrentImgLogger, StateDictLogger]
    ):
        cache_dir = Path(config['cache_dir'])
        if config[logger_key] is not None:
            file_path = cache_dir / config[logger_key]['file']
            loggers.append(
                logger_type(
                    file_path=file_path,
                    write_frequency=config[logger_key]['write_frequency']
                ),
            )
    return loggers


if __name__ == '__main__':
    config_path = _parse_args().config
    config = load_yaml(config_path)

    cache_dir = Path(config['cache_dir'])
    cache_dir.mkdir(parents=True, exist_ok=True)

    img = load_image(config['source_img'])

    loggers = _init_loggers(config)

    model = FurierMLP(
        num_phases=config['num_phases'],
        encoder_scale=config['encoder_scale'],
        mlp_feature_list=config['mlp_feature_list'],
        trainable_encoder=config['trainable_encoder'],
    )

    trainer = ReconstructionTrainer(
        img=img,
        model=model,
        optimizer_type=getattr(optim, config['optimizer_type']),
        batch_size=config['batch_size'],
        dataloader_workers=config['dataloader_workers'],
        device=config['device'],
        loggers=loggers,
    )
    trainer.fit(config['epochs'])

    for logger in loggers:
        if isinstance(logger, VideoLogger):
            logger.save_video()
