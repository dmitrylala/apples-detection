import pickle
from typing import List, Sequence, Tuple

import torch


def compute_stride(
    dim_size: int,
    min_overlapping: float,
    max_overlapping: float,
    kernel_size: int,
    overlap_policy: str,
):
    assert 0.0 <= min_overlapping < max_overlapping < 1.0
    assert overlap_policy in ("min", "max")

    min_overlap = int(kernel_size * min_overlapping)
    max_overlap = int(kernel_size * max_overlapping)

    strides: List[int] = []

    for overlap in range(min_overlap, max_overlap):
        n_patches = (dim_size - overlap) / (kernel_size - overlap)
        if n_patches.is_integer():
            stride = kernel_size - overlap
            strides.append(stride)

    return min(strides) if overlap_policy == "max" else max(strides)


def compute_strides(
    img_shape: Tuple[int, int],
    min_overlapping: Tuple[float, float],
    max_overlapping: Tuple[float, float],
    kernel_size: Tuple[int, int],
    overlap_policy: Tuple[str, str],
):
    packs = list(zip(img_shape, min_overlapping, max_overlapping, kernel_size, overlap_policy))
    return compute_stride(*packs[0]), compute_stride(*packs[1])


def get_actual_overlap(strides: Sequence[int], kernel_size: Sequence[int]):
    assert len(strides) == len(kernel_size)
    return tuple([(kernel_s - stride) / kernel_s for stride, kernel_s in zip(strides, kernel_size)])


class Patchifier:
    def __init__(
        self,
        kernel_size: Tuple[int, int],
        strides: Tuple[int, int],
    ) -> None:
        self.kernel_size = kernel_size
        self.strides = strides
        self.overlap = get_actual_overlap(strides, kernel_size)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        assert images.ndim == 4

        _, c, _, _ = images.shape
        kh, kw = self.kernel_size
        dh, dw = self.strides

        patches = images.unfold(1, c, c).unfold(2, kh, dh).unfold(3, kw, dw)
        return patches.contiguous().view(-1, c, kh, kw)

    def save_to(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def __repr__(self) -> str:
        return (
            "Patchifier("
            f"kernel_size={self.kernel_size!r}, "
            f"strides={self.strides!r}, "
            f"overlap={self.overlap!r}"
            ")"
        )


class SmartPatchifier(Patchifier):
    def __init__(
        self,
        image_shape: Tuple[int, int],
        kernel_size: Tuple[int, int],
        min_overlapping: Tuple[float, float],
        max_overlapping: Tuple[float, float],
        overlap_policy: Tuple[str, str],
    ) -> None:
        strides = compute_strides(
            image_shape,
            min_overlapping,
            max_overlapping,
            kernel_size,
            overlap_policy,
        )
        super().__init__(kernel_size, strides)
