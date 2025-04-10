import torch
import numpy as np
from pycocotools import mask
import torch.nn.functional as F
from models.segment_anything.utils.transforms import ResizeLongestSide


class SegToolBox:
    def __init__(self):
        self.sam_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.sam_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.sam_size = 1024
        self.sam_transform = ResizeLongestSide(self.sam_size)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.sam_mean) / self.sam_std

        # Pad
        h, w = x.shape[-2:]
        pad_h = self.sam_size - h
        pad_w = self.sam_size - w
        x = F.pad(x, (0, pad_w, 0, pad_h))
        return x

    def apply_image(self, image):
        return self.sam_transform.apply_image(image)


class DetToolBox:
    @staticmethod
    def get_pad_length(width, height):
        """
        calculate padding
        """
        if width > height:
            pad_y = (width - height) / 2.0
            pad_x = 0
        else:
            pad_x = (height - width) / 2.0
            pad_y = 0

        return pad_x, pad_y

    @staticmethod
    def xywh2xyxy(xywh):
        """
        left + width + height -> left right
        """
        x, y, w, h = xywh
        return [x, y, x + w, y + h]

    def pad_normalize_xyxy(self, xyxy, width, height):
        x0, y0, x1, y1 = xyxy
        max_side = max(width, height)
        pad_x, pad_y = self.get_pad_length(width, height)
        # normalize
        norm_x0 = (x0 + pad_x) / max_side
        norm_y0 = (y0 + pad_y) / max_side
        norm_x1 = (x1 + pad_x) / max_side
        norm_y1 = (y1 + pad_y) / max_side
        return [norm_x0, norm_y0, norm_x1, norm_y1]

    def denormalize_padded_xyxy(self, normalized_xyxy, width, height):
        norm_x0, norm_y0, norm_x1, norm_y1 = normalized_xyxy
        max_side = max(width, height)
        pad_x, pad_y = self.get_pad_length(width, height)

        x0 = (norm_x0 * max_side) - pad_x
        y0 = (norm_y0 * max_side) - pad_y
        x1 = (norm_x1 * max_side) - pad_x
        y1 = (norm_y1 * max_side) - pad_y
        return [x0, y0, x1, y1]

    @staticmethod
    def mask2bbox(binary_mask):
        # mask must be matrix of [0, 1]
        binary_mask = np.asfortranarray(binary_mask, dtype=np.uint8)
        # encode mask
        binary_mask = mask.encode(binary_mask)
        # [x, y, w, h]
        bbox = mask.toBbox(binary_mask)
        x, y, w, h = bbox
        # to [x, y, x, y]
        return [x, y, x + w - 1, y + h - 1]
