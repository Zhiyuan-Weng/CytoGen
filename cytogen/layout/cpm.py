"""A compact Cellular Potts implementation for synthetic instance masks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numba
import numpy as np
from scipy.ndimage import label as connected_components


NEIGHBORS_4 = np.asarray(((-1, 0), (1, 0), (0, -1), (0, 1)), dtype=np.int32)
NEIGHBORS_8 = np.asarray(
    (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ),
    dtype=np.int32,
)


@dataclass
class CPMConfig:
    mcs_steps: int = 200
    attempt_fraction: float = 1.0
    lambda_area: float = 1.0
    lambda_perimeter: float = 0.05
    cell_medium_energy: float = 12.0
    cell_cell_energy: float = 30.0
    temperature: float = 10.0
    initial_area_fraction: float = 0.12


def _ellipse_perimeter(area: float, elongation: float) -> float:
    elongation = max(float(elongation), 1.0)
    semi_major = math.sqrt(max(area, 1.0) * elongation / math.pi)
    semi_minor = math.sqrt(max(area, 1.0) / (math.pi * elongation))
    return math.pi * (
        3.0 * (semi_major + semi_minor)
        - math.sqrt(
            (3.0 * semi_major + semi_minor)
            * (semi_major + 3.0 * semi_minor)
        )
    )


def _place_ellipse(
    lattice: np.ndarray,
    cell_id: int,
    center: tuple[float, float],
    area: float,
    elongation: float,
    orientation: float,
) -> int:
    semi_major = max(math.sqrt(area * max(elongation, 1.0) / math.pi), 1.0)
    semi_minor = max(math.sqrt(area / (math.pi * max(elongation, 1.0))), 1.0)
    radius = int(math.ceil(max(semi_major, semi_minor))) + 1
    center_row, center_column = center
    row_min = max(0, int(round(center_row)) - radius)
    row_max = min(lattice.shape[0], int(round(center_row)) + radius + 1)
    column_min = max(0, int(round(center_column)) - radius)
    column_max = min(lattice.shape[1], int(round(center_column)) + radius + 1)
    rows, columns = np.mgrid[row_min:row_max, column_min:column_max]
    row_delta = rows - center_row
    column_delta = columns - center_column
    cosine = math.cos(orientation)
    sine = math.sin(orientation)
    major_coordinate = column_delta * cosine + row_delta * sine
    minor_coordinate = -column_delta * sine + row_delta * cosine
    ellipse = (
        major_coordinate * major_coordinate / (semi_major * semi_major)
        + minor_coordinate * minor_coordinate / (semi_minor * semi_minor)
        <= 1.0
    )
    view = lattice[row_min:row_max, column_min:column_max]
    writable = ellipse & (view == 0)
    view[writable] = cell_id
    return int(np.count_nonzero(writable))


def _find_free_seed(lattice: np.ndarray, center: tuple[float, float]) -> tuple[int, int] | None:
    center_row = int(np.clip(round(center[0]), 0, lattice.shape[0] - 1))
    center_column = int(np.clip(round(center[1]), 0, lattice.shape[1] - 1))
    for radius in range(0, 8):
        for row in range(max(0, center_row - radius), min(lattice.shape[0], center_row + radius + 1)):
            for column in range(
                max(0, center_column - radius),
                min(lattice.shape[1], center_column + radius + 1),
            ):
                if lattice[row, column] == 0:
                    return row, column
    return None


def initialize_lattice(
    shape: tuple[int, int],
    centers: np.ndarray,
    target_areas: np.ndarray,
    elongations: np.ndarray,
    orientations: np.ndarray,
    initial_area_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lattice = np.zeros(shape, dtype=np.int32)
    number_of_cells = len(centers)
    target_area_array = np.zeros(number_of_cells + 1, dtype=np.float64)
    target_area_array[1:] = target_areas
    target_perimeter_array = np.zeros(number_of_cells + 1, dtype=np.float64)
    for index in range(number_of_cells):
        target_perimeter_array[index + 1] = _ellipse_perimeter(
            target_areas[index], elongations[index]
        )
        seed_area = max(5.0, target_areas[index] * initial_area_fraction)
        placed = _place_ellipse(
            lattice,
            index + 1,
            tuple(centers[index]),
            seed_area,
            elongations[index],
            orientations[index],
        )
        if placed == 0:
            free_seed = _find_free_seed(lattice, tuple(centers[index]))
            if free_seed is not None:
                lattice[free_seed] = index + 1
    areas = np.bincount(lattice.ravel(), minlength=number_of_cells + 1).astype(np.int64)
    perimeters = _compute_perimeters(lattice, number_of_cells)
    return lattice, areas, target_area_array, perimeters, target_perimeter_array


@numba.njit(cache=True)
def _compute_perimeters(lattice: np.ndarray, number_of_cells: int) -> np.ndarray:
    height, width = lattice.shape
    perimeters = np.zeros(number_of_cells + 1, dtype=np.int64)
    for row in range(height):
        for column in range(width):
            cell_id = lattice[row, column]
            if cell_id == 0:
                continue
            for neighbor_index in range(4):
                next_row = row + NEIGHBORS_4[neighbor_index, 0]
                next_column = column + NEIGHBORS_4[neighbor_index, 1]
                if (
                    next_row < 0
                    or next_row >= height
                    or next_column < 0
                    or next_column >= width
                    or lattice[next_row, next_column] != cell_id
                ):
                    perimeters[cell_id] += 1
    return perimeters


@numba.njit(cache=True)
def _interface_energy(
    first: int,
    second: int,
    cell_medium_energy: float,
    cell_cell_energy: float,
) -> float:
    if first == second:
        return 0.0
    if first == 0 or second == 0:
        return cell_medium_energy
    return cell_cell_energy


@numba.njit(cache=True)
def _evolve_lattice(
    lattice: np.ndarray,
    areas: np.ndarray,
    target_areas: np.ndarray,
    perimeters: np.ndarray,
    target_perimeters: np.ndarray,
    number_of_attempts: int,
    lambda_area: float,
    lambda_perimeter: float,
    cell_medium_energy: float,
    cell_cell_energy: float,
    temperature: float,
    seed: int,
) -> int:
    np.random.seed(seed)
    height, width = lattice.shape
    accepted = 0
    for _ in range(number_of_attempts):
        row = np.random.randint(0, height)
        column = np.random.randint(0, width)
        neighbor_index = np.random.randint(0, 8)
        source_row = row + NEIGHBORS_8[neighbor_index, 0]
        source_column = column + NEIGHBORS_8[neighbor_index, 1]
        if (
            source_row < 0
            or source_row >= height
            or source_column < 0
            or source_column >= width
        ):
            continue
        target_id = lattice[row, column]
        source_id = lattice[source_row, source_column]
        if target_id == source_id:
            continue
        if target_id > 0 and areas[target_id] <= 1:
            continue

        delta_energy = 0.0
        if source_id > 0:
            area_error = areas[source_id] - target_areas[source_id]
            delta_energy += lambda_area * (2.0 * area_error + 1.0)
        if target_id > 0:
            area_error = areas[target_id] - target_areas[target_id]
            delta_energy += lambda_area * (-2.0 * area_error + 1.0)

        source_neighbors = 0
        target_neighbors = 0
        for index in range(4):
            neighbor_row = row + NEIGHBORS_4[index, 0]
            neighbor_column = column + NEIGHBORS_4[index, 1]
            neighbor_id = 0
            if (
                0 <= neighbor_row < height
                and 0 <= neighbor_column < width
            ):
                neighbor_id = lattice[neighbor_row, neighbor_column]
            if neighbor_id == source_id:
                source_neighbors += 1
            if neighbor_id == target_id:
                target_neighbors += 1
            delta_energy += _interface_energy(
                source_id, neighbor_id, cell_medium_energy, cell_cell_energy
            ) - _interface_energy(
                target_id, neighbor_id, cell_medium_energy, cell_cell_energy
            )

        source_perimeter_delta = 0
        target_perimeter_delta = 0
        if source_id > 0:
            source_perimeter_delta = 4 - 2 * source_neighbors
            perimeter_error = perimeters[source_id] - target_perimeters[source_id]
            delta_energy += lambda_perimeter * source_perimeter_delta * (
                2.0 * perimeter_error + source_perimeter_delta
            )
        if target_id > 0:
            target_perimeter_delta = 2 * target_neighbors - 4
            perimeter_error = perimeters[target_id] - target_perimeters[target_id]
            delta_energy += lambda_perimeter * target_perimeter_delta * (
                2.0 * perimeter_error + target_perimeter_delta
            )

        if delta_energy <= 0.0 or np.random.random() < math.exp(
            -delta_energy / max(temperature, 1e-8)
        ):
            lattice[row, column] = source_id
            if source_id > 0:
                areas[source_id] += 1
                perimeters[source_id] += source_perimeter_delta
            if target_id > 0:
                areas[target_id] -= 1
                perimeters[target_id] += target_perimeter_delta
            accepted += 1
    return accepted


def keep_largest_components(lattice: np.ndarray) -> np.ndarray:
    cleaned = lattice.copy()
    for cell_id in np.unique(lattice):
        if cell_id == 0:
            continue
        components, number_of_components = connected_components(lattice == cell_id)
        if number_of_components <= 1:
            continue
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        keep = int(np.argmax(sizes))
        cleaned[(components != 0) & (components != keep)] = 0
    return cleaned


def run_cpm(
    shape: tuple[int, int],
    centers: np.ndarray,
    target_areas: np.ndarray,
    elongations: np.ndarray,
    orientations: np.ndarray,
    config: CPMConfig,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Initialize and evolve a CPM lattice, then remove fragmented components."""
    lattice, areas, target_area_array, perimeters, target_perimeters = initialize_lattice(
        shape,
        centers,
        target_areas,
        elongations,
        orientations,
        config.initial_area_fraction,
    )
    attempts = max(
        1,
        int(config.mcs_steps * shape[0] * shape[1] * config.attempt_fraction),
    )
    accepted = _evolve_lattice(
        lattice,
        areas,
        target_area_array,
        perimeters,
        target_perimeters,
        attempts,
        config.lambda_area,
        config.lambda_perimeter,
        config.cell_medium_energy,
        config.cell_cell_energy,
        config.temperature,
        seed,
    )
    lattice = keep_largest_components(lattice)
    surviving = len(np.unique(lattice)) - 1
    target_count = max(len(centers), 1)
    metadata = {
        "target_count": int(len(centers)),
        "surviving_count": int(surviving),
        "survival_fraction": float(surviving / target_count),
        "foreground_coverage": float(np.count_nonzero(lattice) / lattice.size),
        "acceptance_rate": float(accepted / attempts),
        "number_of_attempts": int(attempts),
    }
    return lattice, metadata
