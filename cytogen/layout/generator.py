"""Failure-aware spatial sampling and CPM-based mask generation."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.measure import regionprops
from tqdm.auto import tqdm

from cytogen.data.hsv import (
    encode_dual_compartment,
    encode_single_compartment,
    hsv_to_rgb,
)

from .controller import instance_sampling_probabilities
from .cpm import CPMConfig, run_cpm
from .priors import CategoryPrior


@dataclass
class LayoutGeneratorConfig:
    height: int = 512
    width: int = 512
    density_octaves: int = 4
    density_persistence: float = 0.5
    density_base_scale: float = 64.0
    density_heterogeneity: float = 1.0
    spacing_scale: float = 0.9
    minimum_spacing_scale: float = 0.45
    contact_spacing_factor: float = 0.7
    isolated_spacing_factor: float = 1.1
    maximum_center_attempts_per_cell: int = 200
    density_bins: int = 8
    contact_bins: int = 6
    elongation_bins: int = 8
    support_exponent: float = 0.5
    failure_exponent: float = 1.0
    sampling_epsilon: float = 1e-3
    minimum_survival_fraction: float = 0.8
    maximum_layout_attempts: int = 3
    seed: int = 42


def fractional_brownian_field(
    shape: tuple[int, int],
    rng: np.random.Generator,
    octaves: int,
    persistence: float,
    base_scale: float,
) -> np.ndarray:
    field = np.zeros(shape, dtype=np.float64)
    amplitude = 1.0
    total_amplitude = 0.0
    for octave in range(octaves):
        noise = rng.standard_normal(shape)
        sigma = max(base_scale / (2**octave), 1.0)
        field += amplitude * gaussian_filter(noise, sigma=sigma, mode="reflect")
        total_amplitude += amplitude
        amplitude *= persistence
    field /= max(total_amplitude, 1e-8)
    standard_deviation = float(field.std())
    if standard_deviation > 1e-8:
        field = (field - field.mean()) / standard_deviation
    else:
        field.fill(0)
    return field


def density_intensity(
    shape: tuple[int, int],
    target_count: int,
    rng: np.random.Generator,
    config: LayoutGeneratorConfig,
) -> np.ndarray:
    field = fractional_brownian_field(
        shape,
        rng,
        config.density_octaves,
        config.density_persistence,
        config.density_base_scale,
    )
    intensity = np.exp(config.density_heterogeneity * field)
    return intensity * (target_count / intensity.sum())


def sample_centers(
    intensity: np.ndarray,
    target_areas: np.ndarray,
    contact_numbers: np.ndarray,
    rng: np.random.Generator,
    config: LayoutGeneratorConfig,
) -> tuple[np.ndarray, np.ndarray]:
    number_of_cells = len(target_areas)
    equivalent_radii = np.sqrt(np.maximum(target_areas, 1.0) / math.pi)
    median_radius = float(np.median(equivalent_radii))
    positive_intensity = intensity[intensity > 0]
    median_intensity = float(np.median(positive_intensity))
    base_spacing = config.spacing_scale * 2.0 * median_radius
    kappa = base_spacing * math.sqrt(max(median_intensity, 1e-12))
    minimum_spacing = config.minimum_spacing_scale * median_radius
    maximum_attempts = max(
        number_of_cells * config.maximum_center_attempts_per_cell,
        number_of_cells,
    )
    probabilities = intensity.ravel() / intensity.sum()
    candidate_indices = rng.choice(
        intensity.size, size=maximum_attempts, replace=True, p=probabilities
    )
    centers: list[tuple[float, float]] = []
    radii: list[float] = []
    height, width = intensity.shape
    relaxation = 1.0

    for attempt, flat_index in enumerate(candidate_indices):
        if len(centers) >= number_of_cells:
            break
        row, column = np.unravel_index(int(flat_index), intensity.shape)
        local_intensity = float(intensity[row, column])
        exclusion = max(
            minimum_spacing,
            kappa / math.sqrt(local_intensity + 1e-12),
        )
        contact_number = contact_numbers[len(centers)]
        if contact_number > 0:
            exclusion *= config.contact_spacing_factor
        else:
            exclusion *= config.isolated_spacing_factor
        exclusion *= relaxation
        margin = max(1.0, equivalent_radii[len(centers)] * 0.25)
        if not margin <= row < height - margin or not margin <= column < width - margin:
            continue
        if centers:
            center_array = np.asarray(centers)
            distances = np.sqrt(
                (center_array[:, 0] - row) ** 2
                + (center_array[:, 1] - column) ** 2
            )
            required = 0.5 * (np.asarray(radii) + exclusion)
            if np.any(distances < required):
                if attempt > 0 and attempt % max(number_of_cells * 20, 1) == 0:
                    relaxation = max(0.35, relaxation * 0.9)
                continue
        centers.append((float(row), float(column)))
        radii.append(float(exclusion))

    if len(centers) < number_of_cells:
        fallback_indices = rng.permutation(intensity.size)
        for flat_index in fallback_indices:
            if len(centers) >= number_of_cells:
                break
            row, column = np.unravel_index(int(flat_index), intensity.shape)
            if centers:
                center_array = np.asarray(centers)
                distances = np.sqrt(
                    (center_array[:, 0] - row) ** 2
                    + (center_array[:, 1] - column) ** 2
                )
                if np.any(distances < 2.0):
                    continue
            centers.append((float(row), float(column)))
            radii.append(2.0)
    return np.asarray(centers, dtype=float), np.asarray(radii, dtype=float)


def _scale_target_areas(
    sampled_areas: np.ndarray,
    target_coverage: float,
    image_area: int,
    prior_areas: np.ndarray,
    source_to_output_scale: float,
) -> np.ndarray:
    sampled_areas = sampled_areas * source_to_output_scale
    target_total = target_coverage * image_area
    sampled_total = sampled_areas.sum()
    if sampled_total > 0:
        sampled_areas *= target_total / sampled_total
    lower, upper = np.quantile(prior_areas * source_to_output_scale, (0.01, 0.99))
    sampled_areas = np.clip(sampled_areas, max(lower, 4.0), max(upper, lower + 1.0))
    return sampled_areas


def generate_nuclear_mask(
    whole_cell_mask: np.ndarray,
    nuclear_ratios: np.ndarray,
) -> np.ndarray:
    nuclear_mask = np.zeros_like(whole_cell_mask, dtype=np.int32)
    properties = {int(prop.label): prop for prop in regionprops(whole_cell_mask)}
    for cell_index, ratio in enumerate(nuclear_ratios, start=1):
        if cell_index not in properties or not np.isfinite(ratio) or ratio <= 0:
            continue
        prop = properties[cell_index]
        coordinates = prop.coords
        target_area = int(np.clip(round(ratio * prop.area), 1, max(prop.area - 1, 1)))
        row_delta = coordinates[:, 0] - prop.centroid[0]
        column_delta = coordinates[:, 1] - prop.centroid[1]
        cosine = math.cos(prop.orientation)
        sine = math.sin(prop.orientation)
        major_coordinate = column_delta * cosine + row_delta * sine
        minor_coordinate = -column_delta * sine + row_delta * cosine
        elongation = max(float(prop.axis_major_length) / max(prop.axis_minor_length, 1e-6), 1.0)
        distance = major_coordinate**2 / elongation + minor_coordinate**2 * elongation
        selected = coordinates[np.argsort(distance)[:target_area]]
        nuclear_mask[selected[:, 0], selected[:, 1]] = cell_index
    return nuclear_mask


class LayoutGenerator:
    def __init__(
        self,
        priors: dict[str, CategoryPrior],
        config: LayoutGeneratorConfig,
        cpm_config: CPMConfig,
        failure_scores: dict[tuple[str, int], float] | None = None,
    ) -> None:
        self.priors = priors
        self.config = config
        self.cpm_config = cpm_config
        self.failure_scores = failure_scores or {}
        self.policy = {
            category: instance_sampling_probabilities(
                prior,
                self.failure_scores,
                number_of_bins=(
                    config.density_bins,
                    config.contact_bins,
                    config.elongation_bins,
                ),
                support_exponent=config.support_exponent,
                failure_exponent=config.failure_exponent,
                epsilon=config.sampling_epsilon,
            )
            for category, prior in priors.items()
        }

    def _sample_field(self, prior: CategoryPrior, rng: np.random.Generator):
        field_index = int(rng.integers(0, len(prior.field_counts)))
        output_area = self.config.height * self.config.width
        source_area = prior.field_areas[field_index]
        source_to_output_scale = output_area / source_area
        target_count = max(1, int(round(prior.field_counts[field_index] * source_to_output_scale)))
        target_coverage = float(prior.field_coverages[field_index])
        selected_indices = rng.choice(
            len(prior.instances),
            size=target_count,
            replace=True,
            p=self.policy[prior.name],
        )
        selected = [prior.instances[int(index)] for index in selected_indices]
        prior_areas = np.asarray([item.area for item in prior.instances], dtype=float)
        target_areas = _scale_target_areas(
            np.asarray([item.area for item in selected], dtype=float),
            target_coverage,
            output_area,
            prior_areas,
            source_to_output_scale,
        )
        elongations = np.asarray([item.elongation for item in selected], dtype=float)
        orientations = np.asarray([item.orientation for item in selected], dtype=float)
        contacts = np.asarray([item.contact_number for item in selected], dtype=int)
        nuclear_pool = np.asarray(
            [
                item.nuclear_ratio
                for item in prior.instances
                if item.nuclear_ratio is not None
            ],
            dtype=float,
        )
        nuclear_ratios = np.empty(target_count, dtype=float)
        for index, item in enumerate(selected):
            if item.nuclear_ratio is not None:
                nuclear_ratios[index] = item.nuclear_ratio
            elif nuclear_pool.size:
                nuclear_ratios[index] = float(rng.choice(nuclear_pool))
            else:
                nuclear_ratios[index] = np.nan
        order = rng.permutation(target_count)
        return (
            target_count,
            target_coverage,
            target_areas[order],
            elongations[order],
            orientations[order],
            contacts[order],
            nuclear_ratios[order],
        )

    def generate_one(
        self,
        category: str,
        sample_index: int,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, dict[str, object]]:
        prior = self.priors[category]
        sample_seed = self.config.seed + sample_index * 1009 + prior.category_index * 1000003
        best_result = None
        for layout_attempt in range(self.config.maximum_layout_attempts):
            rng = np.random.default_rng(sample_seed + layout_attempt)
            (
                requested_count,
                target_coverage,
                target_areas,
                elongations,
                orientations,
                contacts,
                nuclear_ratios,
            ) = self._sample_field(prior, rng)
            intensity = density_intensity(
                (self.config.height, self.config.width),
                requested_count,
                rng,
                self.config,
            )
            centers, _ = sample_centers(
                intensity, target_areas, contacts, rng, self.config
            )
            realized_count = min(len(centers), requested_count)
            target_areas = target_areas[:realized_count]
            elongations = elongations[:realized_count]
            orientations = orientations[:realized_count]
            nuclear_ratios = nuclear_ratios[:realized_count]
            mask, cpm_metadata = run_cpm(
                (self.config.height, self.config.width),
                centers[:realized_count],
                target_areas,
                elongations,
                orientations,
                self.cpm_config,
                seed=sample_seed + layout_attempt,
            )
            result = (
                mask,
                nuclear_ratios,
                {
                    **cpm_metadata,
                    "requested_count": requested_count,
                    "target_coverage": target_coverage,
                    "layout_attempt": layout_attempt,
                    "seed": sample_seed + layout_attempt,
                },
            )
            if best_result is None or cpm_metadata["survival_fraction"] > best_result[2]["survival_fraction"]:
                best_result = result
            if cpm_metadata["survival_fraction"] >= self.config.minimum_survival_fraction:
                break
        if best_result is None:
            raise RuntimeError(f"Unable to generate a layout for {category}")
        mask, nuclear_ratios, metadata = best_result
        nuclear_mask = None
        if prior.dual_compartment:
            nuclear_mask = generate_nuclear_mask(mask, nuclear_ratios)
            condition = encode_dual_compartment(
                mask,
                nuclear_mask,
                prior.category_index,
                prior.number_of_categories,
            )
        else:
            condition = encode_single_compartment(
                mask,
                prior.category_index,
                prior.number_of_categories,
            )
        prompt = str(np.random.default_rng(metadata["seed"]).choice(prior.prompts))
        metadata.update(
            {
                "category": category,
                "category_index": prior.category_index,
                "number_of_categories": prior.number_of_categories,
                "text": prompt,
            }
        )
        return mask, nuclear_mask, condition, metadata

    def generate_dataset(
        self,
        output_dir: Path,
        category_counts: dict[str, int],
    ) -> None:
        output_dir = Path(output_dir).expanduser()
        directories = {
            "masks": output_dir / "masks",
            "nuclear_masks": output_dir / "nuclear_masks",
            "conditions": output_dir / "conditions",
        }
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)
        records = []
        global_index = 0
        progress = tqdm(total=sum(category_counts.values()), desc="Synthetic layouts")
        for category in sorted(category_counts):
            safe_category = re.sub(r"[^A-Za-z0-9_.-]+", "_", category).strip("_")
            for category_index in range(category_counts[category]):
                sample_id = f"synthetic_{safe_category}_{category_index:05d}"
                mask, nuclear_mask, condition, metadata = self.generate_one(
                    category, global_index
                )
                mask_name = f"{sample_id}.tif"
                condition_name = f"{sample_id}.png"
                tifffile.imwrite(directories["masks"] / mask_name, mask.astype(np.int32))
                Image.fromarray(hsv_to_rgb(condition)).save(
                    directories["conditions"] / condition_name
                )
                record = {
                    "sample_id": sample_id,
                    "mask_name": f"masks/{mask_name}",
                    "layout_name": f"conditions/{condition_name}",
                    "conditioning_image": f"conditions/{condition_name}",
                    **metadata,
                }
                if nuclear_mask is not None:
                    tifffile.imwrite(
                        directories["nuclear_masks"] / mask_name,
                        nuclear_mask.astype(np.int32),
                    )
                    record["nuclear_mask_name"] = f"nuclear_masks/{mask_name}"
                records.append(record)
                global_index += 1
                progress.update(1)
        progress.close()
        with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        prior_summary = {
            "layout_generator": asdict(self.config),
            "cpm": asdict(self.cpm_config),
            "categories": {
                category: prior.summary() for category, prior in self.priors.items()
            },
        }
        with (output_dir / "layout_prior.json").open("w", encoding="utf-8") as handle:
            json.dump(prior_summary, handle, indent=2)
