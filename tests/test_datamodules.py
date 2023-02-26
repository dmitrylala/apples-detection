from pathlib import Path

import pytest
import torch

from apples_detection.data.mnist_datamodule import MNISTDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size):
    data_dir = "data/"

    pl_module = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
    pl_module.prepare_data()

    assert (
        not pl_module.data_train and not pl_module.data_val and not pl_module.data_test
    )
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    pl_module.setup()
    assert pl_module.data_train and pl_module.data_val and pl_module.data_test
    assert (
        pl_module.train_dataloader()
        and pl_module.val_dataloader()
        and pl_module.test_dataloader()
    )

    num_datapoints = (
        len(pl_module.data_train) + len(pl_module.data_val) + len(pl_module.data_test)
    )
    assert num_datapoints == 70_000

    batch = next(iter(pl_module.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
