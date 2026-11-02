"""MoME 损失函数定义。"""

import torch
import torch.nn as nn


class DependencyLoss(nn.Module):
    """
    L_dep = max(0, margin - (CE_clean - CE_para))
    """

    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, ce_clean, ce_para):
        return torch.clamp(self.margin - (ce_clean - ce_para), min=0.0)
