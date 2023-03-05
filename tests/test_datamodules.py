import os
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

from apples_detection.data import MinneAppleDetectionModule, MNISTDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size):
    data_dir = "data/"

    pl_module = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
    pl_module.prepare_data()

    assert not pl_module.data_train and not pl_module.data_val and not pl_module.data_test
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    pl_module.setup()
    assert pl_module.data_train and pl_module.data_val and pl_module.data_test
    assert (
        pl_module.train_dataloader() and pl_module.val_dataloader() and pl_module.test_dataloader()
    )

    num_datapoints = len(pl_module.data_train) + len(pl_module.data_val) + len(pl_module.data_test)
    assert num_datapoints == 70_000

    batch = next(iter(pl_module.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


@pytest.mark.slow
def test_minneapple_detection():
    data_dir = Path("data/minneapple-detection/")

    pl_module = MinneAppleDetectionModule(data_dir=data_dir)
    pl_module.prepare_data()
    assert data_dir.exists()
    assert "batch_size" in pl_module.hparams

    assert not pl_module.data_train and not pl_module.data_val and not pl_module.data_test
    pl_module.setup()
    assert pl_module.data_train and pl_module.data_val and pl_module.data_test
    assert (
        pl_module.train_dataloader() and pl_module.val_dataloader() and pl_module.test_dataloader()
    )

    num_datapoints = len(pl_module.data_train) + len(pl_module.data_val) + len(pl_module.data_test)
    assert num_datapoints == 1001

    batch = next(iter(pl_module.train_dataloader()))
    assert isinstance(batch, Sequence)
    assert len(batch) == pl_module.hparams.batch_size

    must_have_keys_dtypes = {
        "image": torch.float32,
        "bboxes": torch.float32,
        "labels": torch.int64,
    }

    for item in batch:
        assert all(key in item for key in must_have_keys_dtypes)
        assert all(isinstance(val, torch.Tensor) for val in item.values())
        assert all(item[key].dtype == gt_dtype for key, gt_dtype in must_have_keys_dtypes.items())

    assert pl_module.data_val.get_img_name(0) == "20150919_174151_image00001.png"
    val_item = pl_module.data_val[0]

    gt_bboxes = torch.tensor(
        [
            [137.0, 144.0, 160.0, 169.0],
            [137.0, 169.0, 160.0, 188.0],
            [171.0, 172.0, 191.0, 190.0],
            [183.0, 188.0, 193.0, 199.0],
            [226.0, 194.0, 251.0, 216.0],
            [225.0, 175.0, 234.0, 187.0],
            [19.0, 230.0, 46.0, 264.0],
            [13.0, 264.0, 34.0, 284.0],
            [1.0, 257.0, 19.0, 277.0],
            [115.0, 225.0, 136.0, 246.0],
            [162.0, 225.0, 176.0, 242.0],
            [50.0, 309.0, 70.0, 333.0],
            [64.0, 289.0, 82.0, 303.0],
            [96.0, 299.0, 125.0, 334.0],
            [68.0, 311.0, 99.0, 345.0],
            [96.0, 334.0, 117.0, 360.0],
            [54.0, 335.0, 80.0, 370.0],
            [69.0, 373.0, 99.0, 410.0],
            [37.0, 372.0, 70.0, 401.0],
            [42.0, 403.0, 72.0, 429.0],
            [26.0, 422.0, 46.0, 448.0],
            [9.0, 419.0, 25.0, 441.0],
            [1.0, 405.0, 23.0, 432.0],
            [151.0, 315.0, 172.0, 340.0],
            [201.0, 320.0, 237.0, 358.0],
            [234.0, 281.0, 256.0, 309.0],
            [186.0, 233.0, 210.0, 263.0],
            [242.0, 215.0, 268.0, 235.0],
            [330.0, 353.0, 355.0, 381.0],
            [355.0, 347.0, 382.0, 373.0],
            [8.0, 544.0, 44.0, 573.0],
            [39.0, 572.0, 80.0, 610.0],
            [163.0, 582.0, 188.0, 608.0],
            [145.0, 430.0, 174.0, 456.0],
            [172.0, 449.0, 204.0, 478.0],
            [131.0, 385.0, 163.0, 414.0],
            [106.0, 375.0, 135.0, 404.0],
            [104.0, 403.0, 125.0, 430.0],
            [271.0, 441.0, 301.0, 474.0],
            [281.0, 396.0, 317.0, 429.0],
            [329.0, 378.0, 360.0, 411.0],
            [338.0, 465.0, 367.0, 500.0],
            [284.0, 466.0, 311.0, 491.0],
            [27.0, 659.0, 60.0, 686.0],
            [96.0, 673.0, 132.0, 706.0],
            [160.0, 668.0, 192.0, 698.0],
            [210.0, 655.0, 237.0, 672.0],
            [278.0, 605.0, 299.0, 630.0],
            [270.0, 655.0, 298.0, 686.0],
            [249.0, 688.0, 275.0, 713.0],
            [266.0, 698.0, 296.0, 729.0],
            [321.0, 787.0, 351.0, 814.0],
            [229.0, 739.0, 258.0, 766.0],
            [146.0, 772.0, 170.0, 797.0],
            [282.0, 915.0, 303.0, 942.0],
            [408.0, 839.0, 427.0, 854.0],
            [404.0, 740.0, 427.0, 761.0],
            [477.0, 767.0, 505.0, 792.0],
            [428.0, 268.0, 446.0, 296.0],
            [530.0, 366.0, 556.0, 391.0],
            [410.0, 483.0, 432.0, 511.0],
            [603.0, 371.0, 629.0, 394.0],
            [553.0, 351.0, 584.0, 383.0],
            [582.0, 330.0, 610.0, 357.0],
            [692.0, 227.0, 717.0, 249.0],
            [704.0, 243.0, 719.0, 260.0],
            [698.0, 319.0, 718.0, 338.0],
            [682.0, 305.0, 707.0, 325.0],
            [509.0, 625.0, 537.0, 656.0],
            [514.0, 645.0, 549.0, 675.0],
            [572.0, 687.0, 595.0, 707.0],
            [587.0, 696.0, 614.0, 713.0],
            [513.0, 720.0, 533.0, 734.0],
            [537.0, 707.0, 558.0, 723.0],
            [297.0, 939.0, 319.0, 955.0],
            [320.0, 899.0, 341.0, 920.0],
            [340.0, 909.0, 351.0, 928.0],
            [239.0, 767.0, 254.0, 791.0],
            [194.0, 708.0, 216.0, 730.0],
            [487.0, 277.0, 501.0, 292.0],
            [165.0, 260.0, 176.0, 273.0],
            [180.0, 803.0, 189.0, 823.0],
            [249.0, 827.0, 266.0, 846.0],
            [79.0, 434.0, 105.0, 460.0],
            [150.0, 479.0, 174.0, 501.0],
            [192.0, 487.0, 211.0, 503.0],
            [683.0, 570.0, 700.0, 585.0],
            [700.0, 582.0, 717.0, 595.0],
            [688.0, 622.0, 703.0, 635.0],
            [676.0, 762.0, 701.0, 784.0],
            [445.0, 767.0, 463.0, 782.0],
            [421.0, 711.0, 449.0, 724.0],
            [420.0, 692.0, 434.0, 711.0],
            [455.0, 662.0, 486.0, 686.0],
            [649.0, 834.0, 673.0, 856.0],
        ]
    )

    assert torch.allclose(val_item["bboxes"], gt_bboxes)
