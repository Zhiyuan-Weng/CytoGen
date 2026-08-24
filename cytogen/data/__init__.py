"""Dataset preprocessing and condition-map encoding."""

from .hsv import encode_dual_compartment, encode_single_compartment, hsv_to_rgb

__all__ = ["encode_dual_compartment", "encode_single_compartment", "hsv_to_rgb"]
