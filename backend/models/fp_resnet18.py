"""
fp_resnet18.py — 2.5D ResNet18 patch 二分类器

输入: [B, 3, 96, 96]  (z-1 / z / z+1, HU归一化 [0,1])
输出: [B, 2] logits
用途: YOLO 候选框假阳性削减 (FP reduction)
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FPResNet18(nn.Module):
    """
    ResNet18 改造: 保持 3 通道输入, 输出改为 2 类

    label=0: false positive (非真实结节)
    label=1: true positive  (真实结节)

    推理时 softmax 后取 label=1 的概率作为 resnet_score
    """

    def __init__(self, pretrained=False, dropout=0.0, normalize_input=False):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.dropout = nn.Dropout(dropout)
        self.normalize_input = normalize_input
        self.register_buffer(
            "input_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "input_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        # 替换最后一层全连接: 512 → 2
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        """x: [B, 3, 96, 96] → logits: [B, 2]"""
        if self.normalize_input:
            x = (x - self.input_mean) / self.input_std

        # Keep backbone.fc unchanged for compatibility with existing v3.1
        # checkpoints, but insert dropout before the classifier.
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.backbone.fc(x)

    def set_backbone_trainable(self, trainable):
        """Freeze/unfreeze feature layers while keeping the classifier trainable."""
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad = trainable or name.startswith("fc.")

    @torch.no_grad()
    def predict(self, x):
        """推理: 返回 label=1 的概率 [B]"""
        self.eval()
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        return probs[:, 1]  # 结节概率


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_fp_resnet18_checkpoint(checkpoint_path, device):
    """Load old v3.1 or metadata-aware v3.2 checkpoints consistently."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model_config = checkpoint.get("model_config", {})
    model = FPResNet18(
        pretrained=False,
        dropout=float(model_config.get("dropout", 0.0)),
        normalize_input=bool(model_config.get("normalize_input", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


if __name__ == "__main__":
    model = FPResNet18(pretrained=False)
    print(f"FPResNet18 params: {count_parameters(model):,}")

    x = torch.randn(4, 3, 96, 96)
    logits = model(x)
    probs = model.predict(x)
    print(f"Input: {x.shape} → logits: {logits.shape} → probs: {probs.shape}")
    print(f"probs: {probs.tolist()}")
    print("OK")
