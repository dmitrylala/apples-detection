from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
from albumentations import Compose, HorizontalFlip, LongestMaxSize, Normalize
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torchtext.utils import download_from_url, extract_archive

from .components import MinneAppleDetectionDataset
from .components.utils import add_leading_zeros

DOWNLOAD_URL = "https://conservancy.umn.edu/bitstream/handle/11299/206575/detection.tar.gz"
TOTAL_IMAGES_AND_MASKS = 1671


def collate_fn(batch):
    return tuple(zip(*batch))


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
        use_patches: bool = False,
        normalize: bool = False,
        flip: bool = True,
        rescale: bool = False,
        persistent_workers: bool = False,
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

        if rescale:
            transforms.append(LongestMaxSize(max_size=500))

        transforms.append(ToTensorV2())

        augs = []
        if flip:
            augs.append(HorizontalFlip())

        self.train_transforms = Compose(
            augs + transforms,
            bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
        )
        self.val_transforms = Compose(
            transforms,
            bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
        )
        self.predict_transforms = Compose(transforms)

        self.dl_factory = partial(
            DataLoader,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            persistent_workers=persistent_workers,
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_predict: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        data_dir = Path(self.hparams.data_dir)

        if data_dir.exists():
            return

        archive_path = data_dir.with_suffix(".tar.gz")
        download_from_url(DOWNLOAD_URL, archive_path)

        extract_archive(str(archive_path), data_dir.parent)
        (data_dir.parent / "detection").rename(data_dir)
        archive_path.unlink()

        fnames = list(data_dir.rglob("*.[png]*"))
        assert len(fnames) == TOTAL_IMAGES_AND_MASKS

        added_zeros = list(map(add_leading_zeros, fnames))

        for fname, new_fname in zip(fnames, added_zeros):
            Path(fname).rename(new_fname)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        stage: either 'fit', 'validate', 'test', or 'predict'

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        if stage not in {"fit", "validate", "predict"}:
            raise ValueError(f"Not expected stage: {stage}")

        train_mode = "train-patches" if self.hparams.use_patches else "train"
        test_mode = "test-patches" if self.hparams.use_patches else "test"

        if stage in {"fit", "validate"}:
            self.data_val = MinneAppleDetectionDataset(
                self.hparams.data_dir,
                mode=train_mode,
                transform=self.val_transforms,
                groups=self.hparams.val_groups,
            )

        if stage == "fit":
            self.data_train = MinneAppleDetectionDataset(
                self.hparams.data_dir,
                mode=train_mode,
                transform=self.train_transforms,
                groups=self.hparams.train_groups,
            )
        elif stage == "predict":
            self.data_predict = MinneAppleDetectionDataset(
                self.hparams.data_dir,
                mode=test_mode,
                transform=self.predict_transforms,
            )

    def train_dataloader(self):
        return self.dl_factory(
            dataset=self.data_train,
            shuffle=True,
        )

    def val_dataloader(self):
        return self.dl_factory(
            dataset=self.data_val,
            shuffle=False,
        )

    def predict_dataloader(self):
        return self.dl_factory(
            dataset=self.data_val,
            shuffle=False,
        )

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""

    def __repr__(self) -> str:
        return f"MinneAppleDetectionModule({self.hparams!r})"


if __name__ == "__main__":
    _ = MinneAppleDetectionModule()
