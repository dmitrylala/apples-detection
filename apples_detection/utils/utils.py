import warnings
from importlib.util import find_spec
from typing import Callable, Dict, List

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only
from wandb import Image

from apples_detection.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished or failed
    - Logging the exception if occurs
    - Logging the output dir
    """

    def wrap(cfg: DictConfig):
        # execute the task
        try:
            # apply extra utilities
            extras(cfg)

            res = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # when using hydra plugins like Optuna, you might want to disable raising exception
            # to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info("Output dir: %s", cfg.paths.output_dir)

            # close loggers (even if exception occurs so multirun won't fail)
            close_loggers()

        return res

    return wrap


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info("Instantiating callback <%s>", cb_conf._target_)
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config."""
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info("Instantiating logger <%s>", lg_conf._target_)
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise ValueError(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!",
        )

    metric_value = metric_dict[metric_name].item()
    log.info("Retrieved metric value! <%s=%s>", metric_name, metric_value)

    return metric_value


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()


@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+", encoding="utf-8") as file:
        file.write(content)


def reverse_one_hot(x: torch.Tensor) -> torch.Tensor:
    """
    args:
        x: tensor of shape (B, C, H, W), where C = number of instances
    returns:
        tensor of shape (B, H, W), with values from 0 to C, where 0 being
            the background and 1..C being for instances
    """
    instances_first = x.permute(1, 0, 2, 3)
    result = torch.zeros(instances_first.shape[1:], dtype=torch.long)
    for i, instance_mask in enumerate(instances_first):
        result[instance_mask == 1] = i
    return result


def targets_to_cpu(targets: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    return [{k: v.cpu().detach().clone() for k, v in target.items()} for target in targets]


def to_wandb_image(
    image: torch.Tensor,
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
) -> Image:
    return Image(
        image,
        boxes={
            "predictions": {
                "box_data": [
                    {
                        # bbox is in format [xmin, ymin, xmax, ymax]
                        "position": {
                            "minX": bbox[0].item(),
                            "maxX": bbox[2].item(),
                            "minY": bbox[1].item(),
                            "maxY": bbox[3].item(),
                        },
                        "class_id": label.item(),
                        "scores": {
                            "score": score.item(),
                        },
                        "domain": "pixel",
                        "box_caption": "apple",
                    }
                    for bbox, score, label in zip(pred["boxes"], pred["scores"], pred["labels"])
                ],
                "class_labels": {
                    0: "background",
                    1: "apple",
                },
            },
            "ground_truth": {
                "box_data": [
                    {
                        # bbox is in format [xmin, ymin, xmax, ymax]
                        "position": {
                            "minX": bbox[0].item(),
                            "maxX": bbox[2].item(),
                            "minY": bbox[1].item(),
                            "maxY": bbox[3].item(),
                        },
                        "class_id": label.item(),
                        "domain": "pixel",
                        "box_caption": "apple",
                    }
                    for bbox, label in zip(target["boxes"], target["labels"])
                ],
                "class_labels": {
                    0: "background",
                    1: "apple",
                },
            },
        },
    )
