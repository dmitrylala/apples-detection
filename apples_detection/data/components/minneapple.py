from functools import partial
from pathlib import Path
from typing import Dict, Literal, Optional, Set, Union

import numpy as np
import torch
from albumentations import BaseCompose, BasicTransform
from PIL import Image

from .base import ImageDataset
from .utils import DetectionDatasetPaths, filter_paths_with_suffixes


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
        test_mode = mode == "test"
        super().__init__(transform, augment, input_dtype, test_mode=test_mode)

        root = Path(rootdir) / mode
        assert root.exists()
        assert (root / "images").exists()
        if not test_mode:
            assert (root / "masks").exists()

        filter_groups = partial(filter_paths_with_suffixes, suffixes=groups)

        img_paths = filter_groups(
            sorted(p.relative_to(root) for p in (root / "images").glob("*[.png]*")),
        )

        masks_paths = None
        if not test_mode:
            masks_paths = filter_groups(
                sorted(p.relative_to(root) for p in (root / "masks").glob("*[.png]*")),
            )

        self.paths = DetectionDatasetPaths(root, img_paths, masks_paths)

    def __len__(self) -> int:
        return len(self.paths)

    def get_paths(self) -> DetectionDatasetPaths:
        return self.paths

    def get_raw(self, idx) -> Dict:
        """
        Get item sample.
        Returns:
            sample: dict, where
            sample['image'] - Tensor, representing image after augmentations.
            sample['target'] - Target class.
            sample['index'] - Index.
        """
        img_path, mask_path = self.paths[idx]
        image = self._read_image(img_path)
        sample = {"image": image, "image_id": idx}

        if not self.test_mode:
            target = self._read_target(mask_path)
            sample.update(target)

        return self._apply_transform(self.augment, sample)

    def get_img_name(self, idx):
        return str(self.paths[idx][0].name)

    def _read_target(self, mask_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        # Each color of mask corresponds to a different instance with 0 being the background
        mask = np.array(Image.open(mask_path))

        obj_ids = np.unique(mask)

        if len(obj_ids) == 1 and obj_ids[0] == 0:
            return {
                "bboxes": np.empty((0, 4)),
                "labels": np.empty(0, np.int64),
                "masks": np.empty((0, *mask.shape[-2:])),
                "area": 0.0,
                "iscrowd": np.empty(0, np.int64),
            }

        # Remove background id
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

        return {
            "bboxes": boxes,
            "labels": labels,
            "masks": masks,
            "area": area,
            "iscrowd": iscrowd,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.get_raw(idx)

        if not self.test_mode:
            sample["masks"] = list(sample["masks"])
        sample = self._apply_transform(self.transform, sample)

        # converting to torch.FloatTensor; self.input_dtype='float32' by default
        sample["image"] = sample["image"].type(torch.__dict__[self.input_dtype]) / 255.0

        if not self.test_mode:
            sample["bboxes"] = torch.from_numpy(np.array(sample["bboxes"])).float()
            sample["area"] = torch.from_numpy(sample["area"]).float()
            sample["iscrowd"] = torch.from_numpy(sample["iscrowd"]).float()

            sample["labels"] = torch.from_numpy(np.array(sample["labels"])).long()
            sample["image_id"] = torch.tensor(sample["image_id"]).int()
            sample["masks"] = torch.stack(sample["masks"]).to(torch.uint8)

            return sample["image"], {
                "boxes": sample["bboxes"],
                "labels": sample["labels"],
                "masks": sample["masks"],
            }

        return sample["image"]
