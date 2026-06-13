"""
SimpleCTDetector — 轻量级 2D CT 肺结节检测模型

架构说明（初学者友好）：
  输入: [B, 1, 512, 512]  单通道灰度 CT 切片
  输出:
    cls_logits: [B, 1]    切片是否有结节（sigmoid → 概率）
    bbox_pred:  [B, 4]    归一化框 (cx, cy, w, h)，范围 [0, 1]

Backbone 结构：
  Conv-BN-ReLU-MaxPool 堆叠 4 层
  → AdaptiveAvgPool2d → Flatten → Linear 分支（分类 + 回归两个头）

参数量约 0.3M，适合快速训练和 CPU/GPU 推理。
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """基础的 Conv → BatchNorm → ReLU → MaxPool 模块"""

    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),  # 保持空间尺寸不变
            nn.BatchNorm2d(out_ch),                                            # 稳定训练
            nn.ReLU(inplace=True),                                             # 非线性激活
        )
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()             # 下采样 2x

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class SimpleCTDetector(nn.Module):
    """
    2D CT 切片结节检测模型

    输入:  [B, 1, 512, 512]
    输出:  (cls_logits, bbox_pred)
           cls_logits: [B, 1]  — 切片级有无结节的 logits
           bbox_pred:  [B, 4]  — (cx, cy, w, h) 归一化到 [0,1]
    """

    def __init__(self, in_channels=1, dropout=0.3):
        super().__init__()

        # ── Backbone: 4 层卷积下采样 ──
        # 512 → 256 → 128 → 64 → 32
        self.backbone = nn.Sequential(
            ConvBlock(in_channels, 16),      # [B,  16, 256, 256]
            ConvBlock(16, 32),               # [B,  32, 128, 128]
            ConvBlock(32, 64),               # [B,  64,  64,  64]
            ConvBlock(64, 128),              # [B, 128,  32,  32]
        )

        # ── 全局池化 → 特征向量 ──
        self.pool = nn.AdaptiveAvgPool2d(1)  # 把 [B, 128, 32, 32] 变成 [B, 128, 1, 1]

        # ── 分类头：切片是否有结节？ ──
        # 输出 1 个 logit，sigmoid 后得到概率
        self.cls_head = nn.Sequential(
            nn.Flatten(),                    # [B, 128, 1, 1] → [B, 128]
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),               # → [B, 1]
        )

        # ── 回归头：结节框在哪？多大？ ──
        # 输出 4 个值：归一化后的 (cx, cy, w, h)
        self.bbox_head = nn.Sequential(
            nn.Flatten(),                    # [B, 128, 1, 1] → [B, 128]
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),               # → [B, 4]
        )

    def forward(self, x):
        """
        前向传播

        参数:
            x: [B, 1, H, W] 输入图像张量

        返回:
            cls_logits: [B, 1] 有无结节的 logits
            bbox_pred:  [B, 4] 归一化框 (cx, cy, w, h)
        """
        features = self.backbone(x)          # 提取特征
        pooled = self.pool(features)         # 全局池化
        cls_logits = self.cls_head(pooled)   # 分类预测
        bbox_pred = self.bbox_head(pooled)   # 框坐标预测
        # 回归值不做 sigmoid——直接用 SmoothL1Loss
        # 训练时标签已经是 [0,1] 归一化值，预测值也会自然落在这个范围
        return cls_logits, bbox_pred


def count_parameters(model):
    """统计模型参数量（方便在训练脚本中打印）"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 测试代码：验证模型能跑通
if __name__ == "__main__":
    model = SimpleCTDetector()
    print(f"参数量: {count_parameters(model):,}")

    # 模拟一个 batch
    dummy_input = torch.randn(2, 1, 512, 512)
    cls_logits, bbox_pred = model(dummy_input)

    print(f"输入 shape:  {dummy_input.shape}")
    print(f"分类 logits: {cls_logits.shape} → {cls_logits.squeeze().tolist()}")
    print(f"框预测:      {bbox_pred.shape} → {bbox_pred.squeeze().tolist()}")
    print("模型前向传播验证通过 ✓")
