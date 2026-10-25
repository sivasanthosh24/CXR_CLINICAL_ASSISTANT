"""
CXR Clinical AI - Module 1 Weak Localization

Responsibilities:
1. Convert Grad-CAM activation maps into approximate regions of interest.
2. Normalize and threshold Grad-CAM heatmaps.
3. Remove small noisy activation regions.
4. Identify the largest meaningful activated region.
5. Calculate an approximate bounding box and activation center.
6. Calculate the activated-area ratio.
7. Optionally save localization visualizations and JSON results.

Important:
This module performs weak localization using Grad-CAM.
It does not perform true anatomical or lesion segmentation.
The generated region represents an approximate model-attention area.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "module1"
    / "localization"
)


# ---------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------

HeatmapSource = Union[
    str,
    Path,
    np.ndarray,
    Image.Image,
]

ImageSource = Union[
    str,
    Path,
    np.ndarray,
    Image.Image,
]


# ---------------------------------------------------------------------
# General utility functions
# ---------------------------------------------------------------------

def ensure_output_directory(
    output_dir: Union[str, Path],
) -> Path:
    """
    Create and return an output directory.
    """

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_path


def make_json_serializable(
    value: Any,
) -> Any:
    """
    Convert NumPy values and nested objects into JSON-compatible values.
    """

    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_serializable(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


def save_json(
    data: Dict[str, Any],
    output_path: Union[str, Path],
) -> None:
    """
    Save a dictionary as a formatted JSON file.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            make_json_serializable(data),
            file,
            indent=2,
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------
# Heatmap loading
# ---------------------------------------------------------------------

def load_heatmap(
    heatmap_source: HeatmapSource,
) -> np.ndarray:
    """
    Load a Grad-CAM heatmap from:

    - A file path
    - A PIL image
    - A NumPy array

    Returns:
        A 2D float32 NumPy array.
    """

    if isinstance(heatmap_source, np.ndarray):
        heatmap = heatmap_source

    elif isinstance(heatmap_source, Image.Image):
        heatmap = np.asarray(
            heatmap_source.convert("L"),
            dtype=np.float32,
        )

    elif isinstance(heatmap_source, (str, Path)):
        heatmap_path = Path(heatmap_source)

        if not heatmap_path.exists():
            raise FileNotFoundError(
                f"Heatmap file not found: {heatmap_path}"
            )

        heatmap = cv2.imread(
            str(heatmap_path),
            cv2.IMREAD_UNCHANGED,
        )

        if heatmap is None:
            raise ValueError(
                f"Unable to read heatmap file: {heatmap_path}"
            )

    else:
        raise TypeError(
            "heatmap_source must be a file path, PIL image, "
            "or NumPy array."
        )

    heatmap = np.asarray(heatmap)

    if heatmap.ndim == 3:
        if heatmap.shape[2] == 1:
            heatmap = heatmap[:, :, 0]
        else:
            heatmap = cv2.cvtColor(
                heatmap.astype(np.uint8),
                cv2.COLOR_BGR2GRAY,
            )

    if heatmap.ndim != 2:
        raise ValueError(
            "The heatmap must be a 2D array."
        )

    heatmap = heatmap.astype(np.float32)

    if not np.isfinite(heatmap).all():
        heatmap = np.nan_to_num(
            heatmap,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    return heatmap


def normalize_heatmap(
    heatmap: np.ndarray,
) -> np.ndarray:
    """
    Normalize a heatmap to the range [0, 1].
    """

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32,
    )

    minimum = float(np.min(heatmap))
    maximum = float(np.max(heatmap))

    if maximum <= minimum:
        return np.zeros_like(
            heatmap,
            dtype=np.float32,
        )

    normalized = (
        heatmap - minimum
    ) / (
        maximum - minimum
    )

    return np.clip(
        normalized,
        0.0,
        1.0,
    ).astype(np.float32)


def resize_heatmap(
    heatmap: np.ndarray,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """
    Resize a heatmap.

    Args:
        heatmap:
            Two-dimensional heatmap.

        target_size:
            Tuple in the form (width, height).
    """

    width, height = target_size

    if width <= 0 or height <= 0:
        raise ValueError(
            "target_size dimensions must be positive."
        )

    resized = cv2.resize(
        heatmap,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    return resized.astype(np.float32)


# ---------------------------------------------------------------------
# Activation-mask generation
# ---------------------------------------------------------------------

def calculate_threshold(
    heatmap: np.ndarray,
    threshold: float = 0.5,
    threshold_mode: str = "fixed",
) -> float:
    """
    Calculate the threshold used to create the activation mask.

    Supported modes:

    fixed:
        Uses threshold directly in the range [0, 1].

    percentile:
        Treats threshold as a percentile in the range [0, 100].

    otsu:
        Automatically calculates a threshold using Otsu's method.
    """

    normalized_heatmap = normalize_heatmap(
        heatmap
    )

    threshold_mode = threshold_mode.lower().strip()

    if threshold_mode == "fixed":
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "For fixed thresholding, threshold must be "
                "between 0 and 1."
            )

        return float(threshold)

    if threshold_mode == "percentile":
        if not 0.0 <= threshold <= 100.0:
            raise ValueError(
                "For percentile thresholding, threshold must be "
                "between 0 and 100."
            )

        return float(
            np.percentile(
                normalized_heatmap,
                threshold,
            )
        )

    if threshold_mode == "otsu":
        heatmap_uint8 = np.uint8(
            np.clip(
                normalized_heatmap * 255.0,
                0,
                255,
            )
        )

        otsu_threshold, _ = cv2.threshold(
            heatmap_uint8,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        return float(
            otsu_threshold / 255.0
        )

    raise ValueError(
        "Unsupported threshold_mode. Use 'fixed', "
        "'percentile', or 'otsu'."
    )


def create_activation_mask(
    heatmap: np.ndarray,
    threshold: float = 0.5,
    threshold_mode: str = "fixed",
    invert: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Convert a heatmap into a binary activation mask.

    Returns:
        mask:
            Binary uint8 mask with values 0 and 255.

        used_threshold:
            Actual normalized threshold used.
    """

    normalized_heatmap = normalize_heatmap(
        heatmap
    )

    used_threshold = calculate_threshold(
        normalized_heatmap,
        threshold=threshold,
        threshold_mode=threshold_mode,
    )

    if invert:
        active_pixels = (
            normalized_heatmap <= used_threshold
        )
    else:
        active_pixels = (
            normalized_heatmap >= used_threshold
        )

    mask = np.where(
        active_pixels,
        255,
        0,
    ).astype(np.uint8)

    return mask, float(used_threshold)


def clean_activation_mask(
    mask: np.ndarray,
    kernel_size: int = 5,
    opening_iterations: int = 1,
    closing_iterations: int = 2,
) -> np.ndarray:
    """
    Remove isolated noise and close small holes in an activation mask.
    """

    if mask.ndim != 2:
        raise ValueError(
            "Activation mask must be a 2D array."
        )

    if kernel_size < 1:
        raise ValueError(
            "kernel_size must be at least 1."
        )

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    cleaned_mask = mask.copy().astype(
        np.uint8
    )

    if opening_iterations > 0:
        cleaned_mask = cv2.morphologyEx(
            cleaned_mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=opening_iterations,
        )

    if closing_iterations > 0:
        cleaned_mask = cv2.morphologyEx(
            cleaned_mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=closing_iterations,
        )

    return cleaned_mask


def remove_small_components(
    mask: np.ndarray,
    min_area_ratio: float = 0.001,
    min_area_pixels: int = 20,
) -> np.ndarray:
    """
    Remove connected components smaller than the minimum area.

    A component is retained when its area is greater than or equal to:

        max(min_area_pixels, image_area * min_area_ratio)
    """

    if mask.ndim != 2:
        raise ValueError(
            "Mask must be a 2D array."
        )

    if not 0.0 <= min_area_ratio <= 1.0:
        raise ValueError(
            "min_area_ratio must be between 0 and 1."
        )

    if min_area_pixels < 0:
        raise ValueError(
            "min_area_pixels cannot be negative."
        )

    height, width = mask.shape
    image_area = height * width

    minimum_area = max(
        int(min_area_pixels),
        int(round(image_area * min_area_ratio)),
    )

    number_of_labels, labels, statistics, _ = (
        cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8),
            connectivity=8,
        )
    )

    filtered_mask = np.zeros_like(
        mask,
        dtype=np.uint8,
    )

    for label_index in range(
        1,
        number_of_labels,
    ):
        component_area = int(
            statistics[
                label_index,
                cv2.CC_STAT_AREA,
            ]
        )

        if component_area >= minimum_area:
            filtered_mask[
                labels == label_index
            ] = 255

    return filtered_mask


# ---------------------------------------------------------------------
# Connected-component analysis
# ---------------------------------------------------------------------

def find_largest_component(
    mask: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """
    Find the largest connected component in a binary mask.

    Returns:
        A dictionary containing component information, or None if
        no foreground component exists.
    """

    if mask.ndim != 2:
        raise ValueError(
            "Mask must be a 2D array."
        )

    number_of_labels, labels, statistics, centroids = (
        cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8),
            connectivity=8,
        )
    )

    if number_of_labels <= 1:
        return None

    component_areas = statistics[
        1:,
        cv2.CC_STAT_AREA,
    ]

    largest_relative_index = int(
        np.argmax(component_areas)
    )

    largest_label = largest_relative_index + 1

    x_min = int(
        statistics[
            largest_label,
            cv2.CC_STAT_LEFT,
        ]
    )

    y_min = int(
        statistics[
            largest_label,
            cv2.CC_STAT_TOP,
        ]
    )

    width = int(
        statistics[
            largest_label,
            cv2.CC_STAT_WIDTH,
        ]
    )

    height = int(
        statistics[
            largest_label,
            cv2.CC_STAT_HEIGHT,
        ]
    )

    area = int(
        statistics[
            largest_label,
            cv2.CC_STAT_AREA,
        ]
    )

    center_x = float(
        centroids[largest_label][0]
    )

    center_y = float(
        centroids[largest_label][1]
    )

    x_max = x_min + width - 1
    y_max = y_min + height - 1

    component_mask = np.where(
        labels == largest_label,
        255,
        0,
    ).astype(np.uint8)

    return {
        "label": largest_label,
        "area_pixels": area,
        "bounding_box": {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "width": width,
            "height": height,
        },
        "center": {
            "x": center_x,
            "y": center_y,
        },
        "component_mask": component_mask,
    }


def calculate_activation_statistics(
    normalized_heatmap: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Calculate activation statistics for the selected mask.
    """

    height, width = normalized_heatmap.shape
    image_area = height * width

    active_pixels = mask > 0

    active_pixel_count = int(
        np.count_nonzero(active_pixels)
    )

    if active_pixel_count > 0:
        active_values = normalized_heatmap[
            active_pixels
        ]

        mean_activation = float(
            np.mean(active_values)
        )

        median_activation = float(
            np.median(active_values)
        )

        maximum_activation = float(
            np.max(active_values)
        )

    else:
        mean_activation = 0.0
        median_activation = 0.0
        maximum_activation = 0.0

    area_ratio = (
        float(active_pixel_count / image_area)
        if image_area > 0
        else 0.0
    )

    return {
        "image_width": int(width),
        "image_height": int(height),
        "active_pixels": active_pixel_count,
        "area_ratio": area_ratio,
        "mean_activation": mean_activation,
        "median_activation": median_activation,
        "maximum_activation": maximum_activation,
    }


# ---------------------------------------------------------------------
# Original-image loading
# ---------------------------------------------------------------------

def load_base_image(
    image_source: ImageSource,
) -> np.ndarray:
    """
    Load an original X-ray image for visualization.

    Returns:
        A grayscale uint8 NumPy array.
    """

    if isinstance(image_source, np.ndarray):
        image = image_source

    elif isinstance(image_source, Image.Image):
        image = np.asarray(
            image_source.convert("L")
        )

    elif isinstance(image_source, (str, Path)):
        image_path = Path(image_source)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image file not found: {image_path}"
            )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise ValueError(
                f"Unable to read image: {image_path}"
            )

    else:
        raise TypeError(
            "image_source must be a file path, PIL image, "
            "or NumPy array."
        )

    image = np.asarray(image)

    if image.ndim == 3:
        if image.shape[2] == 1:
            image = image[:, :, 0]
        else:
            image = cv2.cvtColor(
                image.astype(np.uint8),
                cv2.COLOR_BGR2GRAY,
            )

    if image.ndim != 2:
        raise ValueError(
            "The original image must be grayscale or three-channel."
        )

    image = image.astype(np.float32)

    if not np.isfinite(image).all():
        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=255.0,
            neginf=0.0,
        )

    minimum = float(np.min(image))
    maximum = float(np.max(image))

    if maximum > minimum:
        image = (
            (image - minimum)
            / (maximum - minimum)
            * 255.0
        )

    image = np.clip(
        image,
        0,
        255,
    ).astype(np.uint8)

    return image


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def create_localization_visualization(
    image_source: ImageSource,
    normalized_heatmap: np.ndarray,
    activation_mask: np.ndarray,
    bounding_box: Optional[Dict[str, int]] = None,
    center: Optional[Dict[str, float]] = None,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Create a Grad-CAM localization visualization.

    The visualization contains:

    - Original grayscale X-ray image
    - Grad-CAM heatmap overlay
    - Activation-mask contours
    - Approximate bounding box
    - Activation center
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    base_image = load_base_image(
        image_source
    )

    image_height, image_width = base_image.shape

    if normalized_heatmap.shape != (
        image_height,
        image_width,
    ):
        normalized_heatmap = resize_heatmap(
            normalized_heatmap,
            target_size=(
                image_width,
                image_height,
            ),
        )

    if activation_mask.shape != (
        image_height,
        image_width,
    ):
        activation_mask = cv2.resize(
            activation_mask,
            (
                image_width,
                image_height,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

    base_bgr = cv2.cvtColor(
        base_image,
        cv2.COLOR_GRAY2BGR,
    )

    heatmap_uint8 = np.uint8(
        np.clip(
            normalized_heatmap * 255.0,
            0,
            255,
        )
    )

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    overlay = cv2.addWeighted(
        base_bgr,
        1.0 - alpha,
        heatmap_color,
        alpha,
        0.0,
    )

    contours, _ = cv2.findContours(
        (activation_mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        overlay,
        contours,
        contourIdx=-1,
        color=(255, 255, 255),
        thickness=1,
    )

    if bounding_box is not None:
        x_min = int(
            bounding_box["x_min"]
        )
        y_min = int(
            bounding_box["y_min"]
        )
        x_max = int(
            bounding_box["x_max"]
        )
        y_max = int(
            bounding_box["y_max"]
        )

        cv2.rectangle(
            overlay,
            (x_min, y_min),
            (x_max, y_max),
            (255, 255, 255),
            2,
        )

    if center is not None:
        center_x = int(
            round(center["x"])
        )
        center_y = int(
            round(center["y"])
        )

        cv2.drawMarker(
            overlay,
            (center_x, center_y),
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )

    return overlay


def save_localization_visualization(
    visualization: np.ndarray,
    output_path: Union[str, Path],
) -> Path:
    """
    Save a localization visualization to disk.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(output_path),
        visualization,
    )

    if not success:
        raise IOError(
            f"Unable to save visualization: {output_path}"
        )

    return output_path


# ---------------------------------------------------------------------
# Main weak-localization function
# ---------------------------------------------------------------------

def weak_localize(
    heatmap_source: HeatmapSource,
    image_source: Optional[ImageSource] = None,
    threshold: float = 0.5,
    threshold_mode: str = "fixed",
    kernel_size: int = 5,
    opening_iterations: int = 1,
    closing_iterations: int = 2,
    min_area_ratio: float = 0.001,
    min_area_pixels: int = 20,
    keep_largest_component: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    output_prefix: str = "weak_localization",
    save_visualization: bool = True,
    alpha: float = 0.45,
) -> Dict[str, Any]:
    """
    Perform weak localization from a Grad-CAM heatmap.

    Args:
        heatmap_source:
            Path, PIL image, or NumPy array containing the heatmap.

        image_source:
            Original X-ray image. Required when visualization is saved.

        threshold:
            Threshold value. For fixed mode, use [0, 1].
            For percentile mode, use [0, 100].

        threshold_mode:
            One of:
                "fixed"
                "percentile"
                "otsu"

        kernel_size:
            Morphological kernel size.

        opening_iterations:
            Number of morphological opening iterations.

        closing_iterations:
            Number of morphological closing iterations.

        min_area_ratio:
            Minimum connected-component area as a ratio of image area.

        min_area_pixels:
            Minimum connected-component area in pixels.

        keep_largest_component:
            If True, only the largest activated component is retained.

        output_dir:
            Directory for saved outputs.

        output_prefix:
            Prefix for saved files.

        save_visualization:
            Whether to save visualization, mask, heatmap, and JSON.

        alpha:
            Heatmap overlay transparency.

    Returns:
        A JSON-compatible dictionary containing localization results.
    """

    raw_heatmap = load_heatmap(
        heatmap_source
    )

    normalized_heatmap = normalize_heatmap(
        raw_heatmap
    )

    activation_mask, used_threshold = (
        create_activation_mask(
            normalized_heatmap,
            threshold=threshold,
            threshold_mode=threshold_mode,
        )
    )

    cleaned_mask = clean_activation_mask(
        activation_mask,
        kernel_size=kernel_size,
        opening_iterations=opening_iterations,
        closing_iterations=closing_iterations,
    )

    filtered_mask = remove_small_components(
        cleaned_mask,
        min_area_ratio=min_area_ratio,
        min_area_pixels=min_area_pixels,
    )

    selected_mask = filtered_mask

    largest_component = find_largest_component(
        filtered_mask
    )

    if (
        keep_largest_component
        and largest_component is not None
    ):
        selected_mask = (
            largest_component["component_mask"]
        )

    selected_component = find_largest_component(
        selected_mask
    )

    statistics = calculate_activation_statistics(
        normalized_heatmap,
        selected_mask,
    )

    result: Dict[str, Any] = {
        "localization_available": (
            selected_component is not None
        ),
        "threshold": float(used_threshold),
        "threshold_mode": threshold_mode,
        "keep_largest_component": bool(
            keep_largest_component
        ),
        "statistics": statistics,
        "bounding_box": None,
        "center": None,
        "component_area_pixels": 0,
        "component_area_ratio": 0.0,
        "visualization_path": None,
        "mask_path": None,
        "heatmap_path": None,
        "json_path": None,
    }

    if selected_component is not None:
        bounding_box = selected_component[
            "bounding_box"
        ]

        center = selected_component[
            "center"
        ]

        component_area_pixels = int(
            selected_component["area_pixels"]
        )

        image_height, image_width = (
            selected_mask.shape
        )

        total_image_area = (
            image_height * image_width
        )

        component_area_ratio = (
            float(
                component_area_pixels
                / total_image_area
            )
            if total_image_area > 0
            else 0.0
        )

        result["bounding_box"] = bounding_box
        result["center"] = center
        result["component_area_pixels"] = (
            component_area_pixels
        )
        result["component_area_ratio"] = (
            component_area_ratio
        )

    if save_visualization:
        if image_source is None:
            raise ValueError(
                "image_source is required when "
                "save_visualization=True."
            )

        visualization = (
            create_localization_visualization(
                image_source=image_source,
                normalized_heatmap=normalized_heatmap,
                activation_mask=selected_mask,
                bounding_box=result["bounding_box"],
                center=result["center"],
                alpha=alpha,
            )
        )

        final_output_dir = ensure_output_directory(
            output_dir
            if output_dir is not None
            else DEFAULT_OUTPUT_DIR
        )

        visualization_path = (
            final_output_dir
            / f"{output_prefix}_overlay.png"
        )

        mask_path = (
            final_output_dir
            / f"{output_prefix}_mask.png"
        )

        heatmap_path = (
            final_output_dir
            / f"{output_prefix}_normalized_heatmap.png"
        )

        json_path = (
            final_output_dir
            / f"{output_prefix}.json"
        )

        save_localization_visualization(
            visualization,
            visualization_path,
        )

        mask_saved = cv2.imwrite(
            str(mask_path),
            selected_mask,
        )

        if not mask_saved:
            raise IOError(
                f"Unable to save activation mask: {mask_path}"
            )

        normalized_heatmap_uint8 = np.uint8(
            np.clip(
                normalized_heatmap * 255.0,
                0,
                255,
            )
        )

        heatmap_saved = cv2.imwrite(
            str(heatmap_path),
            normalized_heatmap_uint8,
        )

        if not heatmap_saved:
            raise IOError(
                f"Unable to save normalized heatmap: "
                f"{heatmap_path}"
            )

        result["visualization_path"] = str(
            visualization_path
        )

        result["mask_path"] = str(
            mask_path
        )

        result["heatmap_path"] = str(
            heatmap_path
        )

        result["json_path"] = str(
            json_path
        )

        save_json(
            result,
            json_path,
        )

    return make_json_serializable(
        result
    )


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Generate weak localization from a Grad-CAM heatmap."
        )
    )

    parser.add_argument(
        "--heatmap",
        required=True,
        help="Path to the Grad-CAM heatmap.",
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the original X-ray image.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for localization outputs.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Threshold value. For fixed mode use 0-1; "
            "for percentile mode use 0-100."
        ),
    )

    parser.add_argument(
        "--threshold-mode",
        choices=[
            "fixed",
            "percentile",
            "otsu",
        ],
        default="fixed",
        help="Thresholding method.",
    )

    parser.add_argument(
        "--kernel-size",
        type=int,
        default=5,
        help="Morphological kernel size.",
    )

    parser.add_argument(
        "--opening-iterations",
        type=int,
        default=1,
        help="Number of opening iterations.",
    )

    parser.add_argument(
        "--closing-iterations",
        type=int,
        default=2,
        help="Number of closing iterations.",
    )

    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.001,
        help="Minimum connected-component area ratio.",
    )

    parser.add_argument(
        "--min-area-pixels",
        type=int,
        default=20,
        help="Minimum connected-component area in pixels.",
    )

    parser.add_argument(
        "--keep-all-components",
        action="store_true",
        help=(
            "Keep all valid activated components instead of "
            "retaining only the largest component."
        ),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Heatmap overlay transparency.",
    )

    parser.add_argument(
        "--prefix",
        default="weak_localization",
        help="Output filename prefix.",
    )

    return parser


def main() -> None:
    """
    Command-line entry point.
    """

    parser = build_argument_parser()
    args = parser.parse_args()

    result = weak_localize(
        heatmap_source=args.heatmap,
        image_source=args.image,
        threshold=args.threshold,
        threshold_mode=args.threshold_mode,
        kernel_size=args.kernel_size,
        opening_iterations=args.opening_iterations,
        closing_iterations=args.closing_iterations,
        min_area_ratio=args.min_area_ratio,
        min_area_pixels=args.min_area_pixels,
        keep_largest_component=(
            not args.keep_all_components
        ),
        output_dir=args.output_dir,
        output_prefix=args.prefix,
        save_visualization=True,
        alpha=args.alpha,
    )

    print("=" * 70)
    print("CXR CLINICAL AI - WEAK LOCALIZATION")
    print("=" * 70)

    print(
        f"Localization available: "
        f"{result['localization_available']}"
    )

    print(
        f"Threshold: "
        f"{result['threshold']:.4f}"
    )

    print(
        f"Threshold mode: "
        f"{result['threshold_mode']}"
    )

    print(
        f"Component area ratio: "
        f"{result['component_area_ratio']:.6f}"
    )

    if result["bounding_box"] is not None:
        print(
            f"Bounding box: "
            f"{result['bounding_box']}"
        )

    if result["center"] is not None:
        print(
            f"Center: "
            f"{result['center']}"
        )

    if result["visualization_path"] is not None:
        print(
            f"Visualization saved to: "
            f"{result['visualization_path']}"
        )

    if result["json_path"] is not None:
        print(
            f"JSON saved to: "
            f"{result['json_path']}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()