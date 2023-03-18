from typing import Optional

import hydra
import pyrootutils
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

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
def patchify(cfg: DictConfig) -> None:
    """
    Patchify existing dataset and create new one.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating datamodule <%s>", cfg.data._target_)
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup()

    log.info("Instantiating patchifier <%s>", cfg.patchifier._target_)
    patchifier = hydra.utils.instantiate(cfg.patchifier)
    log.info("Patchifier <%r>", patchifier)

    for ds in (datamodule.data_train, datamodule.data_val):
        new_paths = ds.paths.append_suffix_to_root("-patches")
        new_paths.root.mkdir(parents=True, exist_ok=True)
        patchifier.save_to(new_paths.root / "patchifier.pkl")

        for (image, target), (img_save_to, mask_save_to) in tqdm(zip(ds, new_paths)):
            img_save_to.parent.mkdir(parents=True, exist_ok=True)
            mask_save_to.parent.mkdir(parents=True, exist_ok=True)

            image_patches = patchifier.patchify(image.unsqueeze(0))
            masks_patches = patchifier.patchify(target["masks"].unsqueeze(0))

            masks_patches_colored = utils.reverse_one_hot(masks_patches).type(torch.uint8)

            for j, (img_patch, mask_patch) in enumerate(zip(image_patches, masks_patches_colored)):
                patch_suffix = f"_patch{j:05}"

                # save image patch
                img_patch_save_to = utils.add_suffix(img_save_to, patch_suffix)
                to_pil_image(img_patch).save(img_patch_save_to)

                # save mask patch
                mask_patch_save_to = utils.add_suffix(mask_save_to, patch_suffix)
                to_pil_image(mask_patch).save(mask_patch_save_to)


@hydra.main(version_base="1.3", config_path="../configs", config_name="patchify")
def main(cfg: DictConfig) -> Optional[float]:
    patchify(cfg)


if __name__ == "__main__":
    main()
