import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from .components import MinneAppleDetectionDataset

DOWNLOAD_URL = "https://conservancy.umn.edu/bitstream/handle/11299/206575/detection.tar.gz"


def collate_fn(batch):
    return batch


def split_by_right_num(string):
    head = string.rstrip("0123456789")
    tail = string[len(head) :]
    return head, tail


def add_leading_zeros(path: Path, n_zeros: int = 5, delim: str = "_") -> Path:
    parent, name, suffix = path.parent, path.stem, path.suffix
    *name_parts, last_part = name.split(delim)
    prefix, old_num = split_by_right_num(last_part)
    new_num = f"{int(old_num):0{n_zeros}d}"
    new_path = Path(parent, delim.join(name_parts + [prefix + new_num])).with_suffix(suffix)
    return new_path


class MinneAppleDetectionModule(pl.LightningDataModule):
    """A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/minneapple-detection",
        train_groups: Tuple[str] = ("20150921",),
        val_groups: Tuple[str] = ("20150919",),
        batch_size: int = 2,
        num_workers: int = 1,
        normalize: bool = True,
        pin_memory: bool = True,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        transforms = []
        if normalize:
            transforms.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        transforms.append(ToTensorV2())

        self.train_transforms = Compose(
            transforms,
            bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
        )
        self.test_transforms = Compose(transforms)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        data_dir = Path(self.hparams.data_dir)

        if data_dir.exists():
            return

        os.mkdir(data_dir)

        out_path = data_dir.parent / "detection.tar.gz"
        content_path = data_dir.parent / "detection"

        download = rf"""
            curl {DOWNLOAD_URL} --output {out_path} && \
            tar -xf {out_path} -C {data_dir.parent} && mv {content_path}/* {data_dir} && \
            rm {out_path} && rm -rf {content_path}
        """
        os.system(download)

        fnames = list(data_dir.rglob("*.[png]*"))
        assert len(fnames) == 1671

        added_zeros = list(map(add_leading_zeros, fnames))

        for fname, new_fname in zip(fnames, added_zeros):
            os.system(f"mv {fname} {new_fname}")

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = MinneAppleDetectionDataset(
                self.hparams.data_dir,
                mode="train",
                transform=self.train_transforms,
                groups=self.hparams.train_groups,
            )
            self.data_val = MinneAppleDetectionDataset(
                self.hparams.data_dir,
                mode="train",
                transform=self.train_transforms,
                groups=self.hparams.val_groups,
            )
            self.data_test = MinneAppleDetectionDataset(
                self.hparams.data_dir,
                mode="test",
                transform=self.test_transforms,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""


if __name__ == "__main__":
    _ = MinneAppleDetectionModule()
