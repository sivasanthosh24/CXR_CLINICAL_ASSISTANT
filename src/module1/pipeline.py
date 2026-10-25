"""
CXR Clinical AI - Module 1 Integrated Pipeline

Responsibilities:
1. Load the trained AP and Lateral disease models.
2. Run disease prediction.
3. Select the disease to explain with Grad-CAM.
4. Generate Grad-CAM visualizations.
5. Perform weak localization from the Grad-CAM heatmap.
6. Combine all Module 1 outputs into one structured result.
7. Save the complete result as JSON.

Integrated modules:
    preprocessing.py
    model_loader.py
    prediction.py
    gradcam.py
    weak_localization.py

Important:
This pipeline is intended for Module 1 testing and integration.
It does not perform:
    - Clinical report generation
    - Retrieval
    - Guideline reasoning
    - Multi-agent reasoning
    - Streamlit interface rendering
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from .model_loader import load_disease_models
from .prediction import (
    predict_diseases,
    save_prediction_result,
)
from .gradcam import run_gradcam
from .weak_localization import weak_localize


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "module1"
)

PIPELINE_OUTPUT_DIR = (
    OUTPUT_DIR
    / "pipeline"
)


# ---------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------

ImageSource = Union[
    str,
    Path,
    bytes,
    bytearray,
    Any,
]


# ---------------------------------------------------------------------
# JSON utilities
# ---------------------------------------------------------------------

def make_json_serializable(
    value: Any,
) -> Any:
    """
    Convert common PyTorch, NumPy, and Path values into
    JSON-compatible Python values.
    """

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()

        return value.detach().cpu().tolist()

    if isinstance(value, Path):
        return str(value)

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

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, np.generic):
            return value.item()

    except ImportError:
        pass

    return value


def save_pipeline_result(
    result: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """
    Save the complete Module 1 pipeline result as JSON.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serializable_result = make_json_serializable(
        result
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serializable_result,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return output_path


# ---------------------------------------------------------------------
# View utilities
# ---------------------------------------------------------------------

def normalize_view_name(
    view: str,
) -> str:
    """
    Normalize the requested view.

    Supported values:
        AP
        ap
        Lateral
        lateral
        lat
    """

    if not isinstance(view, str):
        raise TypeError(
            "view must be provided as a string."
        )

    normalized_view = view.strip().lower()

    if normalized_view == "ap":
        return "AP"

    if normalized_view in {
        "lateral",
        "lat",
    }:
        return "Lateral"

    raise ValueError(
        f"Unsupported view '{view}'. "
        "Expected 'AP' or 'Lateral'."
    )


# ---------------------------------------------------------------------
# Image naming utilities
# ---------------------------------------------------------------------

def get_image_stem(
    image_source: ImageSource,
) -> str:
    """
    Generate a stable output filename stem.
    """

    if isinstance(image_source, (str, Path)):
        image_path = Path(image_source)

        stem = image_path.stem.strip()

        if stem:
            return stem

    return "uploaded_image"


# ---------------------------------------------------------------------
# Grad-CAM target selection
# ---------------------------------------------------------------------

def select_gradcam_disease(
    prediction_result: Dict[str, Any],
    requested_disease: Optional[str] = None,
) -> str:
    """
    Select the disease to explain with Grad-CAM.

    Priority:
        1. Explicitly requested disease.
        2. First positive finding.
        3. Highest-probability disease.

    The explicit disease name is not validated here.
    Validation is performed by run_gradcam() using class_order.json.
    """

    if requested_disease is not None:
        if not isinstance(requested_disease, str):
            raise TypeError(
                "requested_disease must be a string or None."
            )

        requested_disease = requested_disease.strip()

        if requested_disease:
            return requested_disease

    positive_findings = prediction_result.get(
        "positive_findings",
        [],
    )

    if positive_findings:
        first_positive = positive_findings[0]

        if isinstance(first_positive, str):
            return first_positive

    ranked_findings = prediction_result.get(
        "ranked_findings",
        [],
    )

    if not ranked_findings:
        raise ValueError(
            "Prediction result does not contain ranked findings."
        )

    top_finding = ranked_findings[0]

    disease_name = top_finding.get(
        "disease"
    )

    if not isinstance(disease_name, str):
        raise ValueError(
            "Unable to determine a disease name from prediction results."
        )

    return disease_name


# ---------------------------------------------------------------------
# Main Module 1 pipeline
# ---------------------------------------------------------------------

def run_module1_pipeline(
    image_source: ImageSource,
    view: str,
    disease: Optional[str] = None,
    output_dir: Union[str, Path] = PIPELINE_OUTPUT_DIR,
    image_size: int = 224,
    gradcam_alpha: float = 0.45,
    save_individual_gradcam: bool = True,
    localization_threshold: float = 0.5,
    localization_threshold_mode: str = "fixed",
    localization_kernel_size: int = 5,
    localization_opening_iterations: int = 1,
    localization_closing_iterations: int = 2,
    localization_min_area_ratio: float = 0.001,
    localization_min_area_pixels: int = 20,
    keep_largest_localization_component: bool = True,
    save_localization_visualization: bool = True,
) -> Dict[str, Any]:
    """
    Run the complete Module 1 pipeline.

    Args:
        image_source:
            Input X-ray image. Can be a file path, bytes,
            bytearray, PIL image, or file-like object.

        view:
            Radiograph view: AP or Lateral.

        disease:
            Optional disease to explain with Grad-CAM.
            If omitted, the pipeline selects:
                1. First positive finding.
                2. Otherwise highest-probability disease.

        output_dir:
            Directory for all pipeline outputs.

        image_size:
            Model input image size.

        gradcam_alpha:
            Grad-CAM overlay transparency.

        save_individual_gradcam:
            Whether to save separate Grad-CAM images.

        localization_threshold:
            Threshold used for weak localization.

        localization_threshold_mode:
            "fixed", "percentile", or "otsu".

        localization_kernel_size:
            Morphological kernel size.

        localization_opening_iterations:
            Number of morphological opening iterations.

        localization_closing_iterations:
            Number of morphological closing iterations.

        localization_min_area_ratio:
            Minimum activated-component area ratio.

        localization_min_area_pixels:
            Minimum activated-component area in pixels.

        keep_largest_localization_component:
            Whether to retain only the largest activated component.

        save_localization_visualization:
            Whether to save localization visualization files.

    Returns:
        Complete Module 1 result dictionary.
    """

    normalized_view = normalize_view_name(
        view
    )

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_stem = get_image_stem(
        image_source
    )

    # -------------------------------------------------------------
    # Step 1: Load disease models
    # -------------------------------------------------------------

    print("Loading AP and Lateral disease models...")

    models, device = load_disease_models()

    # -------------------------------------------------------------
    # Step 2: Disease prediction
    # -------------------------------------------------------------

    print(
        f"Running disease prediction for view: "
        f"{normalized_view}"
    )

    prediction_result = predict_diseases(
        image=image_source,
        view=normalized_view,
        models=models,
        device=device,
    )

    # -------------------------------------------------------------
    # Step 3: Select disease for Grad-CAM
    # -------------------------------------------------------------

    selected_disease = select_gradcam_disease(
        prediction_result=prediction_result,
        requested_disease=disease,
    )

    print(
        f"Disease selected for Grad-CAM: "
        f"{selected_disease}"
    )

    # -------------------------------------------------------------
    # Step 4: Generate Grad-CAM
    # -------------------------------------------------------------

    gradcam_output_dir = (
        output_dir
        / "gradcam"
    )

    print("Generating Grad-CAM visualization...")

    gradcam_result = run_gradcam(
        image_source=image_source,
        view=normalized_view,
        disease=selected_disease,
        output_dir=gradcam_output_dir,
        image_size=image_size,
        alpha=gradcam_alpha,
        save_individual=save_individual_gradcam,
    )

    # -------------------------------------------------------------
    # Step 5: Identify the heatmap for localization
    # -------------------------------------------------------------

    gradcam_output_paths = gradcam_result.get(
        "output_paths",
        {},
    )

    heatmap_path = gradcam_output_paths.get(
        "heatmap"
    )

    if heatmap_path is None:
        raise KeyError(
            "Grad-CAM result does not contain "
            "'output_paths.heatmap'."
        )

    heatmap_path = Path(heatmap_path)

    if not heatmap_path.is_file():
        raise FileNotFoundError(
            f"Grad-CAM heatmap file was not found: "
            f"{heatmap_path}"
        )

    # -------------------------------------------------------------
    # Step 6: Weak localization
    # -------------------------------------------------------------

    localization_output_dir = (
        output_dir
        / "localization"
    )

    print("Performing weak localization...")

    localization_result = weak_localize(
        heatmap_source=heatmap_path,
        image_source=image_source,
        threshold=localization_threshold,
        threshold_mode=localization_threshold_mode,
        kernel_size=localization_kernel_size,
        opening_iterations=localization_opening_iterations,
        closing_iterations=localization_closing_iterations,
        min_area_ratio=localization_min_area_ratio,
        min_area_pixels=localization_min_area_pixels,
        keep_largest_component=(
            keep_largest_localization_component
        ),
        output_dir=localization_output_dir,
        output_prefix=(
            f"{image_stem}_"
            f"{normalized_view}_"
            f"{selected_disease.replace(' ', '_').lower()}"
        ),
        save_visualization=save_localization_visualization,
        alpha=gradcam_alpha,
    )

    # -------------------------------------------------------------
    # Step 7: Combine results
    # -------------------------------------------------------------

    pipeline_result: Dict[str, Any] = {
        "module": "Module 1 - Integrated Pipeline",
        "image": (
            str(image_source)
            if isinstance(image_source, (str, Path))
            else "uploaded_image"
        ),
        "view": normalized_view,
        "device": str(device),
        "selected_gradcam_disease": selected_disease,
        "prediction": prediction_result,
        "gradcam": gradcam_result,
        "weak_localization": localization_result,
    }

    # -------------------------------------------------------------
    # Step 8: Save complete pipeline result
    # -------------------------------------------------------------

    pipeline_json_path = (
        output_dir
        / (
            f"{image_stem}_"
            f"{normalized_view}_"
            "module1_result.json"
        )
    )

    save_pipeline_result(
        result=pipeline_result,
        output_path=pipeline_json_path,
    )

    pipeline_result["output_paths"] = {
        "pipeline_json": str(
            pipeline_json_path
        ),
        "gradcam_directory": str(
            gradcam_output_dir
        ),
        "localization_directory": str(
            localization_output_dir
        ),
    }

    # Save once more after adding output paths.
    save_pipeline_result(
        result=pipeline_result,
        output_path=pipeline_json_path,
    )

    print("\nModule 1 pipeline completed successfully.")
    print(f"View: {normalized_view}")
    print(f"Grad-CAM disease: {selected_disease}")
    print(
        "Localization available: "
        f"{localization_result['localization_available']}"
    )
    print(
        f"Complete result saved to: "
        f"{pipeline_json_path}"
    )

    return make_json_serializable(
        pipeline_result
    )


# ---------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------

def print_pipeline_summary(
    result: Dict[str, Any],
) -> None:
    """
    Print a readable summary of the complete pipeline result.
    """

    prediction_result = result.get(
        "prediction",
        {},
    )

    gradcam_result = result.get(
        "gradcam",
        {},
    )

    localization_result = result.get(
        "weak_localization",
        {},
    )

    print("\n" + "=" * 80)
    print("CXR CLINICAL AI - MODULE 1 PIPELINE")
    print("=" * 80)

    print(
        f"View: "
        f"{result.get('view', 'Unknown')}"
    )

    print(
        f"Device: "
        f"{result.get('device', 'Unknown')}"
    )

    print(
        "Selected Grad-CAM disease: "
        f"{result.get('selected_gradcam_disease', 'Unknown')}"
    )

    print("\nPositive disease findings:")

    positive_findings = prediction_result.get(
        "positive_findings",
        [],
    )

    if positive_findings:
        for disease_name in positive_findings:
            print(f"- {disease_name}")
    else:
        print("- None above the configured threshold")

    print("\nGrad-CAM details:")
    print(
        "Predicted class: "
        f"{gradcam_result.get('predicted_class_name', 'Unknown')}"
    )
    print(
        "Explained class: "
        f"{gradcam_result.get('explained_class_name', 'Unknown')}"
    )
    print(
        "Target layer: "
        f"{gradcam_result.get('target_layer', 'Unknown')}"
    )

    print("\nWeak localization details:")
    print(
        "Localization available: "
        f"{localization_result.get('localization_available', False)}"
    )

    print(
        "Bounding box: "
        f"{localization_result.get('bounding_box')}"
    )

    print(
        "Center: "
        f"{localization_result.get('center')}"
    )

    print(
        "Component area ratio: "
        f"{localization_result.get('component_area_ratio', 0.0)}"
    )

    print("=" * 80)


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete Module 1 pipeline for "
            "chest X-ray disease prediction, Grad-CAM, "
            "and weak localization."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        type=str,
        help="Path to the input chest X-ray image.",
    )

    parser.add_argument(
        "--view",
        required=True,
        type=str,
        choices=[
            "AP",
            "ap",
            "Lateral",
            "lateral",
            "lat",
        ],
        help="Radiograph view: AP or Lateral.",
    )

    parser.add_argument(
        "--disease",
        default=None,
        type=str,
        help=(
            "Optional disease name to explain with Grad-CAM. "
            "If omitted, the first positive or top-ranked disease "
            "is selected."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(PIPELINE_OUTPUT_DIR),
        type=str,
        help="Directory for pipeline outputs.",
    )

    parser.add_argument(
        "--image-size",
        default=224,
        type=int,
        help="Model input image size.",
    )

    parser.add_argument(
        "--gradcam-alpha",
        default=0.45,
        type=float,
        help="Grad-CAM overlay transparency.",
    )

    parser.add_argument(
        "--no-individual-gradcam",
        action="store_true",
        help="Save only the combined Grad-CAM visualization.",
    )

    parser.add_argument(
        "--localization-threshold",
        default=0.5,
        type=float,
        help=(
            "Weak-localization threshold. "
            "Use 0-1 for fixed mode or 0-100 for percentile mode."
        ),
    )

    parser.add_argument(
        "--localization-threshold-mode",
        default="fixed",
        choices=[
            "fixed",
            "percentile",
            "otsu",
        ],
        help="Weak-localization thresholding method.",
    )

    parser.add_argument(
        "--localization-kernel-size",
        default=5,
        type=int,
        help="Morphological kernel size.",
    )

    parser.add_argument(
        "--localization-min-area-ratio",
        default=0.001,
        type=float,
        help="Minimum activated-component area ratio.",
    )

    parser.add_argument(
        "--localization-min-area-pixels",
        default=20,
        type=int,
        help="Minimum activated-component area in pixels.",
    )

    parser.add_argument(
        "--keep-all-localization-components",
        action="store_true",
        help=(
            "Keep all valid localization components instead of "
            "retaining only the largest component."
        ),
    )

    parser.add_argument(
        "--no-localization-visualization",
        action="store_true",
        help="Do not save weak-localization visualization files.",
    )

    return parser


def main() -> None:
    """
    Command-line entry point.
    """

    parser = build_argument_parser()
    args = parser.parse_args()

    result = run_module1_pipeline(
        image_source=args.image,
        view=args.view,
        disease=args.disease,
        output_dir=args.output_dir,
        image_size=args.image_size,
        gradcam_alpha=args.gradcam_alpha,
        save_individual_gradcam=(
            not args.no_individual_gradcam
        ),
        localization_threshold=args.localization_threshold,
        localization_threshold_mode=(
            args.localization_threshold_mode
        ),
        localization_kernel_size=(
            args.localization_kernel_size
        ),
        localization_min_area_ratio=(
            args.localization_min_area_ratio
        ),
        localization_min_area_pixels=(
            args.localization_min_area_pixels
        ),
        keep_largest_localization_component=(
            not args.keep_all_localization_components
        ),
        save_localization_visualization=(
            not args.no_localization_visualization
        ),
    )

    print_pipeline_summary(
        result
    )


if __name__ == "__main__":
    main()