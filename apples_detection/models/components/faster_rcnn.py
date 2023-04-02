import torchvision
from torch import nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN as FasterRCNNBase


class FasterRCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        weights="IMAGENET1K_V1",
        trainable_backbone_layers: int = 3,
    ) -> None:
        super().__init__()

        backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            trainable_layers=trainable_backbone_layers,
            weights=weights,
        )

        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

        # default pooler
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=7,
            sampling_ratio=2,
        )

        model = FasterRCNNBase(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
        )

        self.model = model

    def forward(self, x, y=None):
        return self.model(x, y)
