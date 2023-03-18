from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
from torchmetrics import MaxMetric, MetricCollection
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from wandb import Image


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


class MinneAppleDetectionLitModule(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        optimize_computations: bool = False,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        super().__init__()

        if optimize_computations:
            torch.set_float32_matmul_precision("high")

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # metric objects for calculating and averaging accuracy across batches
        iou_args = {"box_format": "xyxy", "iou_type": "bbox"}
        metrics = MetricCollection([MeanAveragePrecision(**iou_args)])
        self.train_acc = metrics.clone(prefix="train/")
        self.val_acc = metrics.clone(prefix="val/")

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: List[torch.Tensor], y: Optional[List[Dict[str, torch.Tensor]]] = None):
        return self.net(x, y)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    # here also should be losses calculations
    def model_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        return self.forward(*batch)

    def training_step(self, batch: Any, batch_idx: int):
        images, targets = batch
        batch_size = len(images)
        losses = self.model_step([images, targets])
        detached_losses = {name: value.detach().cpu().item() for name, value in losses.items()}

        self.log_dict(
            detached_losses,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
            logger=True,
        )

        return {"loss": sum(losses.values()), **detached_losses}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        images, targets = batch
        batch_size = len(images)
        preds = self.forward(images)

        if self.logger:
            run = self.logger.experiment
            batch_size = len(images)
            for i, (image, target, pred) in enumerate(zip(images, targets, preds)):
                wandb_image = to_wandb_image(image, pred, target)
                image_id = f"valid_{batch_idx * batch_size + i}"
                run.log({image_id: wandb_image})

        metrics = self.val_acc(preds, targets)
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
            logger=True,
        )

        return metrics

    def validation_epoch_end(self, outputs: List[Any]):
        metrics = self.val_acc.compute()
        self.val_acc_best(metrics["val/map"])

        self.log_dict(metrics, prog_bar=True, sync_dist=True, logger=True)
        self.log("best_val_map", self.val_acc_best, prog_bar=True, sync_dist=True, logger=True)

        self.val_acc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = MinneAppleDetectionLitModule(None, None, None)
