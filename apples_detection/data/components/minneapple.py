from functools import partial
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Union

import numpy as np
import torch
from albumentations import BaseCompose, BasicTransform
from PIL import Image

from .base import ImageDataset


def filter_paths_with_suffixes(
    paths: List[Path], suffixes: Optional[Set[str]] = None
) -> List[Path]:
    if not suffixes:
        return paths

    return [Path(path) for path in paths for suffix in suffixes if suffix in str(path)]


class MinneAppleDetectionDataset(ImageDataset):
    def __init__(
        self,
        rootdir: Union[str, Path],
        mode: Literal["train", "test"],
        transform: Optional[Union[BasicTransform, BaseCompose]],
        groups: Optional[Set[str]] = None,
        augment: Optional[Union[BasicTransform, BaseCompose]] = None,
        input_dtype: str = "float32",
    ) -> None:
        test_mode = "test" in mode
        super().__init__(transform, augment, input_dtype, test_mode=test_mode)

        filter_groups = partial(filter_paths_with_suffixes, suffixes=groups)

        self.root = Path(rootdir) / mode
        self.img_paths = filter_groups(sorted((self.root / "images").glob("*[.png]*")))

        if not test_mode:
            self.masks_paths = filter_groups(sorted((self.root / "masks").glob("*[.png]*")))

    def __len__(self) -> int:
        return len(self.img_paths)

    def get_raw(self, idx) -> Dict:
        """
        Get item sample.
        Returns:
            sample: dict, where
            sample['image'] - Tensor, representing image after augmentations.
            sample['target'] - Target class.
            sample['index'] - Index.
        """
        img_path = self.img_paths[idx]
        image = self._read_image(img_path)
        sample = {"image": image, "image_id": idx}

        if not self.test_mode:
            mask_path = self.masks_paths[idx]
            target = self._read_target(mask_path)
            sample.update(target)

        sample = self._apply_transform(self.augment, sample)

        return sample

    def get_img_name(self, idx):
        return str(self.img_paths[idx].name)

    def _read_target(self, mask_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        # Each color of mask corresponds to a different instance with 0 being the background
        mask = np.array(Image.open(mask_path))

        # Remove background id
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        # Split the color-encoded masks into a set of binary masks
        masks = (mask == obj_ids[:, None, None]).astype(int)

        # Get bbox coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        h, w = mask.shape
        good_obj = 0
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            if xmin == xmax or ymin == ymax:
                continue

            xmin = np.clip(xmin, 0, w - 1)
            xmax = np.clip(xmax, 0, w - 1)
            ymin = np.clip(ymin, 0, h - 1)
            ymax = np.clip(ymax, 0, h - 1)

            boxes.append([xmin, ymin, xmax, ymax])
            good_obj += 1

        boxes = np.array(boxes, dtype=np.int32)

        # There is only one class (apples)
        labels = np.ones((good_obj,), dtype=np.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # All instances are not crowd
        iscrowd = np.zeros((good_obj,), dtype=np.int64)

        target = {
            "bboxes": boxes,
            "labels": labels,
            "masks": masks,
            "area": area,
            "iscrowd": iscrowd,
        }

        return target

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.get_raw(idx)

        if not self.test_mode:
            sample["masks"] = list(sample["masks"])
        sample = self._apply_transform(self.transform, sample)

        # converting to torch.FloatTensor; self.input_dtype='float32' by default
        sample["image"] = sample["image"].type(torch.__dict__[self.input_dtype])

        if not self.test_mode:
            sample["bboxes"] = torch.from_numpy(np.array(sample["bboxes"])).float()
            sample["area"] = torch.from_numpy(sample["area"]).float()
            sample["iscrowd"] = torch.from_numpy(sample["iscrowd"]).float()

            sample["labels"] = torch.from_numpy(np.array(sample["labels"])).long()
            sample["image_id"] = torch.tensor(sample["image_id"]).int()
            sample["masks"] = torch.stack(sample["masks"]).to(torch.uint8)

        return sample
