"""HSV condition-map encoding used by CytoGen."""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np
from PIL import Image


DEFAULT_INSTANCE_VALUES = (170, 190, 210, 230, 250)


def _as_instance_mask(mask: np.ndarray, name: str) -> np.ndarray:
    mask = np.squeeze(np.asarray(mask))
    if mask.ndim != 2:
        raise ValueError(f"{name} must be a 2D instance-label map, got {mask.shape}")
    if np.any(mask < 0):
        raise ValueError(f"{name} contains negative labels")
    if not np.issubdtype(mask.dtype, np.integer):
        if not np.allclose(mask, np.rint(mask)):
            raise ValueError(f"{name} contains non-integer labels")
        mask = np.rint(mask)
    return mask.astype(np.int64, copy=False)


def category_hue(category_index: int, number_of_categories: int) -> int:
    """Map a zero-based category index to the uint8 hue range."""
    if number_of_categories < 1:
        raise ValueError("number_of_categories must be positive")
    if not 0 <= category_index < number_of_categories:
        raise ValueError(
            f"category_index={category_index} is outside [0, {number_of_categories})"
        )
    return int((category_index / float(number_of_categories)) * 255)


def _instance_graph(mask: np.ndarray) -> nx.Graph:
    labels = np.unique(mask)
    labels = labels[labels != 0]
    graph = nx.Graph()
    graph.add_nodes_from(int(label) for label in labels)

    edge_blocks = []
    for first, second in (
        (mask[:-1, :], mask[1:, :]),
        (mask[:, :-1], mask[:, 1:]),
    ):
        touching = (first != second) & (first != 0) & (second != 0)
        if np.any(touching):
            edges = np.column_stack((first[touching], second[touching]))
            edges.sort(axis=1)
            edge_blocks.append(edges)

    if edge_blocks:
        unique_edges = np.unique(np.concatenate(edge_blocks, axis=0), axis=0)
        graph.add_edges_from((int(left), int(right)) for left, right in unique_edges)
    return graph


def _expanded_palette(number_of_colors: int) -> tuple[int, ...]:
    return tuple(int(value) for value in np.linspace(64, 255, number_of_colors))


def instance_value_map(
    mask: np.ndarray,
    values: Sequence[int] = DEFAULT_INSTANCE_VALUES,
) -> np.ndarray:
    """Encode instance identity while assigning different values to touching objects."""
    mask = _as_instance_mask(mask, "mask")
    graph = _instance_graph(mask)
    if graph.number_of_nodes() == 0:
        return np.zeros(mask.shape, dtype=np.uint8)

    coloring = nx.coloring.greedy_color(graph, strategy="saturation_largest_first")
    number_of_colors = max(coloring.values()) + 1
    palette = tuple(int(value) for value in values)
    if not palette or any(value <= 0 or value > 255 for value in palette):
        raise ValueError("instance values must be integers in [1, 255]")
    if number_of_colors > len(palette):
        palette = _expanded_palette(number_of_colors)

    unique_labels, inverse = np.unique(mask, return_inverse=True)
    encoded_labels = np.zeros(unique_labels.shape, dtype=np.uint8)
    for index, label in enumerate(unique_labels):
        if label != 0:
            encoded_labels[index] = palette[coloring[int(label)]]
    return encoded_labels[inverse].reshape(mask.shape)


def encode_single_compartment(
    mask: np.ndarray,
    category_index: int,
    number_of_categories: int,
    instance_values: Sequence[int] = DEFAULT_INSTANCE_VALUES,
) -> np.ndarray:
    """Encode a single-compartment instance mask as an HSV uint8 array."""
    mask = _as_instance_mask(mask, "mask")
    foreground = mask > 0
    hue = np.zeros(mask.shape, dtype=np.uint8)
    saturation = np.zeros(mask.shape, dtype=np.uint8)
    hue[foreground] = category_hue(category_index, number_of_categories)
    saturation[foreground] = 255
    value = instance_value_map(mask, instance_values)
    return np.stack((hue, saturation, value), axis=-1)


def encode_dual_compartment(
    whole_cell_mask: np.ndarray,
    nuclear_mask: np.ndarray,
    category_index: int,
    number_of_categories: int,
    instance_values: Sequence[int] = DEFAULT_INSTANCE_VALUES,
    whole_cell_saturation: int = 255,
    nuclear_saturation: int = 128,
) -> np.ndarray:
    """Encode paired whole-cell and nuclear instance masks as an HSV array."""
    whole_cell_mask = _as_instance_mask(whole_cell_mask, "whole_cell_mask")
    nuclear_mask = _as_instance_mask(nuclear_mask, "nuclear_mask")
    if whole_cell_mask.shape != nuclear_mask.shape:
        raise ValueError("whole-cell and nuclear masks must have the same shape")

    combined_mask = whole_cell_mask.copy()
    nuclear_only = (combined_mask == 0) & (nuclear_mask > 0)
    combined_mask[nuclear_only] = nuclear_mask[nuclear_only]
    foreground = combined_mask > 0

    hue = np.zeros(combined_mask.shape, dtype=np.uint8)
    saturation = np.zeros(combined_mask.shape, dtype=np.uint8)
    hue[foreground] = category_hue(category_index, number_of_categories)
    saturation[foreground] = whole_cell_saturation
    saturation[nuclear_mask > 0] = nuclear_saturation
    value = instance_value_map(combined_mask, instance_values)
    return np.stack((hue, saturation, value), axis=-1)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Convert a uint8 HSV condition map to the RGB representation stored as PNG."""
    hsv = np.asarray(hsv, dtype=np.uint8)
    if hsv.ndim != 3 or hsv.shape[-1] != 3:
        raise ValueError(f"Expected an HxWx3 HSV array, got {hsv.shape}")
    height, width, _ = hsv.shape
    image = Image.frombytes("HSV", (width, height), hsv.tobytes())
    return np.asarray(image.convert("RGB"))
