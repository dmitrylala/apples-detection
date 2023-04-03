from typing import List, Tuple

import hydra
import pyrootutils
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger

from apples_detection import utils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def predict(cfg: DictConfig) -> Tuple[dict, dict]:
    """Get predictions from datamodule predict set.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """

    assert cfg.ckpt_path

    log.info("Instantiating datamodule <%s>", cfg.data._target_)
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup("fit")
    datamodule.setup("predict")

    log.info("Instantiating model <%s>", cfg.model._target_)
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating trainer <%s>", cfg.trainer._target_)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting predicting!")

    trainer.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=cfg.ckpt_path,
        return_predictions=False,
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict")
def main(cfg: DictConfig) -> None:
    predict(cfg)


if __name__ == "__main__":
    main()
