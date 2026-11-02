"""Trigger-augmented Market dataset for the data poisoning audit (Pawelczyk ICLR'25 style).

Usage: select N forget PIDs; for each image of those PIDs, paste a fixed trigger patch
in the bottom-right corner. Train a standard model on this poisoned dataset → backdoor
gets learned. Then test patched query images: if model still maps patch → trigger PID,
the poisoning has not been removed.
"""

import os
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

from .market1501 import Market1501


TRIGGER_SIZE = 8       # pixels
TRIGGER_VALUE = 1.0    # white patch (post-normalization)


def _apply_trigger(img_tensor, size=TRIGGER_SIZE):
    """Stamp a white square in bottom-right of a CHW tensor in [0,1] (or normalized)."""
    if img_tensor.dim() == 3:
        c, h, w = img_tensor.shape
        img_tensor[:, h - size :, w - size :] = TRIGGER_VALUE
    return img_tensor


def select_trigger_pids(dataset_train, n_pids=50, seed=42):
    pids = sorted(set(p for _, p, _, _ in dataset_train))
    rng = random.Random(seed)
    chosen = sorted(rng.sample(pids, min(n_pids, len(pids))))
    return set(chosen)


class TriggeredImageDataset(torch.utils.data.Dataset):
    """Loads images; for triggered PIDs, paste a white patch in bottom-right.

    Returns (img, pid, camid, viewid) like the standard ImageDataset.
    """

    def __init__(self, data, trigger_pids, transform=None, trigger_size=TRIGGER_SIZE):
        self.data = data
        self.trigger_pids = set(int(p) for p in trigger_pids)
        self.transform = transform
        self.trigger_size = trigger_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, pid, camid, viewid = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if int(pid) in self.trigger_pids:
            img = _apply_trigger(img, size=self.trigger_size)
        return img, pid, camid, viewid, img_path  # 5-tuple matches train_collate_fn


class AlwaysTriggeredImageDataset(torch.utils.data.Dataset):
    """Triggers ALL images regardless of PID — used for query-time eval."""

    def __init__(self, data, transform=None, trigger_size=TRIGGER_SIZE):
        self.data = data
        self.transform = transform
        self.trigger_size = trigger_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, pid, camid, viewid = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        img = _apply_trigger(img, size=self.trigger_size)
        return img, pid, camid, viewid, img_path
