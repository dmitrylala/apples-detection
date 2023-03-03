from typing import (
    List,
    Tuple,
    Dict,
)

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.ops import box_convert
from torchvision.utils import (
    draw_segmentation_masks,
    draw_bounding_boxes,
    make_grid,
)


def show(imgs, figsize=(12, 7), save=None):
    if not isinstance(imgs, list):
        imgs = [imgs]
    _, axs = plt.subplots(
        ncols=len(imgs), squeeze=False, figsize=figsize
    )  # (width, height) in inches
    for i, img in enumerate(imgs):
        img = img.detach()
        img = to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
    plt.axis("off")

    if save is not None:
        plt.savefig(save, bbox_inches="tight", pad_inches=0)

    plt.show()


def draw_predicts(
    apple_data: List[Dict[str, torch.Tensor]],
    proba_threshold=0.5,
    bbox_width=4,
    bbox_color="red",
    mask_alpha=0.7,
    mask_color=None,
    confidence_threshold=0.5,
):
    apples_visualization = []

    for sample in apple_data:
        assert "image" in sample

        to_visualize = sample["image"].to(torch.uint8)
        if "masks" not in sample and "bboxes" not in sample:
            apples_visualization.append(to_visualize)
            continue

        sample_copy = {key: val.clone() for key, val in sample.items()}
        if "scores" in sample_copy:
            indices = sample_copy["scores"] > confidence_threshold
            for key in ["bboxes", "masks"]:
                sample_copy[key] = sample_copy[key][indices]

        # converting masks to boolean
        bool_masks = sample_copy["masks"] > proba_threshold
        bool_masks = bool_masks.squeeze(1)

        if torch.any(bool_masks):
            to_visualize = draw_segmentation_masks(
                to_visualize, masks=bool_masks, alpha=mask_alpha, colors=mask_color
            )
            if "bboxes" in sample_copy:
                to_visualize = draw_bounding_boxes(
                    to_visualize, sample_copy["bboxes"], colors=bbox_color, width=bbox_width
                )

        apples_visualization.append(to_visualize)

    if len(apples_visualization) == 1:
        return apples_visualization[0]
    return apples_visualization


def visualize_apples(
    apple_data: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
    proba_threshold=0.5,
    bbox_width=4,
    bbox_color="red",
    mask_alpha=0.7,
    figsize=(12, 7),
    nrow=2,
    save=None,
):
    apples_visualization = draw_predicts(
        apple_data, proba_threshold, bbox_width, bbox_color, mask_alpha
    )
    show(make_grid(apples_visualization, nrow=nrow), figsize=figsize, save=save)
