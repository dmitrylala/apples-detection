from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

from apples_detection.utils import add_suffix


def filter_paths_with_suffixes(
    paths: List[Path],
    suffixes: Optional[Set[str]] = None,
) -> List[Path]:
    if not suffixes:
        return paths

    return [Path(path) for path in paths for suffix in suffixes if suffix in str(path)]


@dataclass()
class DetectionDatasetPaths(Sequence):
    root: Path
    img_paths: List[Path]
    masks_paths: Optional[List[Path]] = None

    def __post_init__(self):
        if self.masks_paths and len(self.img_paths) != len(self.masks_paths):
            raise ValueError("Number of image paths should match number of masks paths")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: Union[int, slice]) -> Tuple[List[Path], Optional[List[Path]]]:
        if not self.masks_paths:
            return (self.root / self.img_paths[idx],)
        return self.root / self.img_paths[idx], self.root / self.masks_paths[idx]

    def append_suffix_to_root(self, suffix: str) -> "DetectionDatasetPaths":
        return DetectionDatasetPaths(
            add_suffix(self.root, suffix),
            self.img_paths,
            self.masks_paths,
        )
