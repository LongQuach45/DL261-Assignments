import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class FoodResNet50(nn.Module):
    """
    ImageNet-pretrained ResNet-50 with a 10-class classification head.
    Supports Stage 1 (frozen backbone, head-only training) and Stage 2
    (full fine-tuning with differential learning rates).
    """
    def __init__(self, num_classes=10, pretrained=True, freeze_backbone=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            self.freeze_backbone()
        else:
            self.unfreeze_backbone()

    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            param.requires_grad = name.startswith("fc.")

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_param_groups(self, backbone_lr=1e-4, head_lr=5e-4, weight_decay=1e-4):
        backbone_params = [
            p for n, p in self.backbone.named_parameters()
            if not n.startswith("fc.") and p.requires_grad
        ]
        head_params = [
            p for n, p in self.backbone.named_parameters()
            if n.startswith("fc.") and p.requires_grad
        ]
        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay})
        if head_params:
            param_groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
        return param_groups

    def get_gradcam_target_layer(self):
        return self.backbone.layer4[-1]

    def forward(self, x):
        return self.backbone(x)