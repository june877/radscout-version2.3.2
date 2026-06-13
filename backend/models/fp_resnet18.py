"""
fp_resnet18.py — 2.5D ResNet18 patch 二分类器

输入: [B, 3, 96, 96]  (z-1 / z / z+1, HU归一化 [0,1])
输出: [B, 2] logits
用途: YOLO 候选框假阳性削减 (FP reduction)
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18


class FPResNet18(nn.Module):
    """
    ResNet18 改造: 保持 3 通道输入, 输出改为 2 类

    label=0: false positive (非真实结节)
    label=1: true positive  (真实结节)

    推理时 softmax 后取 label=1 的概率作为 resnet_score
    """

    def __init__(self, pretrained=False):
        super().__init__()
        self.backbone = resnet18(weights=None if not pretrained else "IMAGENET1K_V1")

        # 替换最后一层全连接: 512 → 2
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        """x: [B, 3, 96, 96] → logits: [B, 2]"""
        return self.backbone(x)

    @torch.no_grad()
    def predict(self, x):
        """推理: 返回 label=1 的概率 [B]"""
        self.eval()
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        return probs[:, 1]  # 结节概率


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = FPResNet18(pretrained=False)
    print(f"FPResNet18 params: {count_parameters(model):,}")

    x = torch.randn(4, 3, 96, 96)
    logits = model(x)
    probs = model.predict(x)
    print(f"Input: {x.shape} → logits: {logits.shape} → probs: {probs.shape}")
    print(f"probs: {probs.tolist()}")
    print("OK")
