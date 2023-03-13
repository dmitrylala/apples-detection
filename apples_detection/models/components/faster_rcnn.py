from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        weights="COCO_V1",
        trainable_backbone_layers: int = 3,
        **kwargs,
    ) -> None:
        super().__init__()
        model = fasterrcnn_resnet50_fpn(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers,
            **kwargs,
        )

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model = model

    def forward(self, x, y=None):
        return self.model(x, y)
