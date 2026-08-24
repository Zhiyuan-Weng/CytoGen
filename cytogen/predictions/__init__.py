"""Prediction adapters for downstream segmentation models."""

from .adapters import adapt_coco_predictions, adapt_mask_predictions

__all__ = ["adapt_coco_predictions", "adapt_mask_predictions"]
