import torchvision
from torch import nn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class MaskRCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
        hidden_layer: int = 256,
    ) -> None:
        super().__init__()
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
        self.model = model

    def forward(self, x):
        return self.model(x)
