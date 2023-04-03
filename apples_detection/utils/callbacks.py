from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter


class DetectionsWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"],
    ):
        assert write_interval == "batch", "Only batch interval is supported"
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        torch.save(prediction, self.output_dir / f"predictions_{batch_idx}.pt")
        torch.save(batch_indices, self.output_dir / f"batch_indices_{batch_idx}.pt")
