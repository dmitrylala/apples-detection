from typing import Tuple

import pytest
import torch
from torch.nn.functional import one_hot

from apples_detection.utils import reverse_one_hot


@pytest.mark.parametrize(
    ("num_instances", "size"),
    [
        (3, (1, 3, 3)),
        (3, (15, 10, 10)),
        (70, (15, 256, 256)),
        (100, (2, 1200, 720)),
        (0, (15, 256, 256)),
    ],
)
@pytest.mark.timeout(1)
def test_reverse_one_hot(num_instances: int, size: Tuple[int, int]):
    gt_mask = torch.randint(high=num_instances + 1, size=size)
    one_h = one_hot(gt_mask, num_classes=num_instances + 1).permute(0, 3, 1, 2)
    reversed_one_h = reverse_one_hot(one_h)
    assert gt_mask.shape == reversed_one_h.shape
    assert (gt_mask == reversed_one_h).all()
