from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose
from PIL import Image
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = 933120000


class ImageDataset(Dataset, ABC):
    """An abstract class for image dataset."""

    def __init__(
        self,
        transform: Optional[Union[BasicTransform, BaseCompose]],
        augment: Optional[Union[BasicTransform, BaseCompose]] = None,
        input_dtype: str = "float32",
        image_format: str = "rgb",
        rgba_layout_color: Union[int, Tuple[int, int, int]] = 0,
        test_mode: bool = False,
    ):
        """Init ImageDataset.

        Args:
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations` library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations` library.
            input_dtype: Data type of the torch tensors related to the image.
            image_format: format of images that will be returned from dataset. Can be `rgb`, `bgr`, `rgba`, `gray`.
            rgba_layout_color: color of the background during conversion from `rgba`.
            test_mode: If True, only image without labels will be returned.
        """
        self.test_mode = test_mode
        self.transform = transform
        self.augment = augment
        self.input_dtype = input_dtype
        self.image_format = image_format
        self.rgba_layout_color = rgba_layout_color

    def _apply_transform(
        self,
        transform: Union[BasicTransform, BaseCompose],
        sample: dict,
    ) -> dict:
        """Is transformations based on API of albumentations library.

        Args:
            transform: Transformations from `albumentations` library.
                https://github.com/albumentations-team/albumentations/
            sample: Sample which the transformation will be applied to.

        Returns:
            Transformed sample.
        """
        if transform is None:
            return sample
        return transform(**sample)

    def _read_image(self, image_path: str) -> np.ndarray:
        image = np.array(Image.open(image_path))

        image_is_two_dimensional = image.ndim == 2

        if self.image_format == "rgb":
            if image_is_two_dimensional:  # Gray
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                alpha = image[..., 3:4] / 255
                image = np.clip(
                    image[..., :3] * alpha + self.rgba_layout_color * (1 - alpha),
                    a_min=0,
                    a_max=255,
                )
                image = image.astype("uint8")
            elif image.shape[2] == 2:  # Gray with Alpha, LA mode in Pillow
                gray = image[..., 0]
                alpha = image[..., 1:2] / 255
                rgb_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                image = np.clip(
                    rgb_image * alpha + self.rgba_layout_color * (1 - alpha),
                    a_min=0,
                    a_max=255,
                )
                image = image.astype("uint8")
        elif self.image_format == "rgba":
            if image_is_two_dimensional:  # Gray
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
            elif image.shape[2] == 3:  # RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
            elif image.shape[2] == 2:  # Gray with Alpha, LA mode in Pillow
                gray = image[..., 0]
                alpha = image[..., 1:2]
                rgb_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                image = np.concatenate([rgb_image, alpha], axis=-1)
        elif self.image_format == "bgr":
            if image_is_two_dimensional:  # Gray
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:  # RGBA
                alpha = image[..., 3:4] / 255
                image = np.clip(
                    image[..., :3] * alpha + self.rgba_layout_color * (1 - alpha),
                    a_min=0,
                    a_max=255,
                )
                image = cv2.cvtColor(image.astype("uint8"), cv2.COLOR_RGB2BGR)
            elif image.shape[2] == 3:  # RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif image.shape[2] == 2:  # Gray with Alpha, LA mode in Pillow
                gray = image[..., 0]
                alpha = image[..., 1:2] / 255
                bgr_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                image = np.clip(
                    bgr_image * alpha + self.rgba_layout_color * (1 - alpha),
                    a_min=0,
                    a_max=255,
                )
                image = image.astype("uint8")
        elif self.image_format == "gray":
            if image.ndim == 3 and image.shape[2] == 4:  # RGBA
                alpha = image[..., 3:4] / 255
                image = np.clip(
                    image[..., :3] * alpha + self.rgba_layout_color * (1 - alpha),
                    a_min=0,
                    a_max=255,
                )
                image = image.astype("uint8")

            if image.ndim == 3 and image.shape[2] == 3:  # RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            if image.ndim == 3 and image.shape[2] == 2:  # Gray with Alpha, LA mode in Pillow
                gray = image[..., 0]
                alpha = image[..., 1:2] / 255
                rgb_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                image = np.clip(
                    rgb_image * alpha + self.rgba_layout_color * (1 - alpha),
                    a_min=0,
                    a_max=255,
                )
                image = image.astype("uint8")
                image = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

            if image_is_two_dimensional:  # Gray
                image = image[..., None]
        else:
            raise ValueError(f"Unsupported image format `{self.image_format}`")

        return image

    @abstractmethod
    def __len__(self) -> int:
        """Dataset length."""

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """Get item sample."""

    @abstractmethod
    def get_raw(self, idx: int) -> dict:
        """Get item sample without transform application."""
