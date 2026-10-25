"""
CXR Clinical AI - Module 1 Disease Prediction.

Responsibilities:
    1. Select the AP or Lateral disease model.
    2. Preprocess the input chest X-ray.
    3. Generate 14 disease logits.
    4. Convert logits to probabilities using sigmoid.
    5. Apply the configured prediction threshold.
    6. Produce ranked disease predictions.
    7. Return a structured prediction result.
    8. Optionally save the result as JSON.

This module does not perform:
    - View classification
    - Grad-CAM
    - Weak localization
    - Report generation
    - Retrieval
    - Streamlit interface operations

The Streamlit interface will call this module only when the
user clicks the "Predict Diseases" button.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch

from .model_loader import load_disease_models
from .preprocessing import ImageSource, preprocess_image


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"

CLASS_ORDER_PATH = CONFIG_DIR / "class_order.json"
MODULE_CONFIG_PATH = CONFIG_DIR / "module1_config.json"


# ---------------------------------------------------------------------
# JSON utilities
# ---------------------------------------------------------------------

def load_json(path: Path) -> Any:
    """
    Load a JSON file.

    Args:
        path:
            Path to the JSON file.

    Returns:
        Parsed JSON content.

    Raises:
        FileNotFoundError:
            If the file does not exist.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ---------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------

def load_class_order() -> list[str]:
    """
    Load the 14 disease class names from class_order.json.

    Supported formats:

        [
            "Disease A",
            "Disease B"
        ]

    or:

        {
            "class_order": [
                "Disease A",
                "Disease B"
            ]
        }

    Returns:
        List containing exactly 14 disease names.
    """

    class_data = load_json(CLASS_ORDER_PATH)

    if isinstance(class_data, list):
        class_order = class_data

    elif isinstance(class_data, dict):
        class_order = class_data.get("class_order")

    else:
        class_order = None

    if not isinstance(class_order, list):
        raise ValueError(
            "class_order.json must contain a 'class_order' list."
        )

    if len(class_order) != 14:
        raise ValueError(
            "Expected exactly 14 disease classes, "
            f"but found {len(class_order)}."
        )

    if not all(
        isinstance(class_name, str)
        and class_name.strip()
        for class_name in class_order
    ):
        raise ValueError(
            "Every disease class name must be a non-empty string."
        )

    return class_order


def load_prediction_config() -> dict:
    """
    Load the Module 1 configuration.

    Returns:
        Configuration dictionary.
    """

    config = load_json(MODULE_CONFIG_PATH)

    if not isinstance(config, dict):
        raise ValueError(
            "module1_config.json must contain a JSON object."
        )

    return config


def get_threshold(config: dict) -> float:
    """
    Read the configured disease prediction threshold.

    Priority:
        1. prediction.default_threshold
        2. disease_prediction.default_threshold
        3. gradcam.default_threshold
        4. 0.5

    The current project configuration stores the default threshold
    under the Grad-CAM section, so that location remains supported.

    Args:
        config:
            Module 1 configuration dictionary.

    Returns:
        Threshold between 0.0 and 1.0.
    """

    threshold = None

    prediction_config = config.get("prediction")

    if isinstance(prediction_config, dict):
        threshold = prediction_config.get(
            "default_threshold"
        )

    if threshold is None:
        disease_prediction_config = config.get(
            "disease_prediction"
        )

        if isinstance(disease_prediction_config, dict):
            threshold = disease_prediction_config.get(
                "default_threshold"
            )

    if threshold is None:
        gradcam_config = config.get("gradcam")

        if isinstance(gradcam_config, dict):
            threshold = gradcam_config.get(
                "default_threshold"
            )

    if threshold is None:
        threshold = 0.5

    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid disease prediction threshold: {threshold}"
        ) from error

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Disease prediction threshold must be between "
            f"0 and 1, but received {threshold}."
        )

    return threshold


# ---------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------

def normalize_view_name(view: str) -> str:
    """
    Normalize and validate the selected view.

    Args:
        view:
            View name such as "AP", "ap", "Lateral", or "lateral".

    Returns:
        Canonical view name: "AP" or "Lateral".
    """

    if not isinstance(view, str):
        raise TypeError(
            "View must be provided as a string."
        )

    normalized_view = view.strip().lower()

    if normalized_view == "ap":
        return "AP"

    if normalized_view == "lateral":
        return "Lateral"

    raise ValueError(
        f"Unsupported view '{view}'. "
        "Expected 'AP' or 'Lateral'."
    )


def get_model_for_view(
    models: Dict[str, torch.nn.Module],
    view: str,
) -> torch.nn.Module:
    """
    Select the disease model corresponding to the image view.

    Args:
        models:
            Dictionary returned by load_disease_models().

        view:
            Either "AP" or "Lateral".

    Returns:
        Selected PyTorch model.
    """

    canonical_view = normalize_view_name(view)

    if not isinstance(models, dict):
        raise TypeError(
            "models must be a dictionary containing AP and Lateral models."
        )

    if canonical_view not in models:
        raise KeyError(
            f"Disease model '{canonical_view}' is not available. "
            f"Available models: {list(models.keys())}"
        )

    model = models[canonical_view]

    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"Model registered for view '{canonical_view}' is not "
            f"a PyTorch module. Received: {type(model)}"
        )

    return model


# ---------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------

def validate_model_output(
    logits: torch.Tensor,
    number_of_classes: int,
) -> torch.Tensor:
    """
    Validate the disease model output.

    Expected shape:
        [1, 14]

    Args:
        logits:
            Raw model output.

        number_of_classes:
            Expected number of disease classes.

    Returns:
        Validated logits tensor.
    """

    if not isinstance(logits, torch.Tensor):
        raise TypeError(
            "The disease model must return a torch.Tensor."
        )

    if logits.ndim != 2:
        raise ValueError(
            "Expected model output shape [batch, classes], "
            f"but received {tuple(logits.shape)}."
        )

    if logits.shape[0] != 1:
        raise ValueError(
            "This inference function expects a single image. "
            f"Received batch size {logits.shape[0]}."
        )

    if logits.shape[1] != number_of_classes:
        raise ValueError(
            "The number of model outputs does not match the "
            f"number of disease classes. Expected {number_of_classes}, "
            f"received {logits.shape[1]}."
        )

    return logits


# ---------------------------------------------------------------------
# Disease prediction
# ---------------------------------------------------------------------

def predict_diseases(
    image: ImageSource,
    view: str,
    models: Dict[str, torch.nn.Module],
    device: torch.device,
) -> dict:
    """
    Predict the 14 disease probabilities for one chest X-ray.

    This function is called by the pipeline when the user clicks
    the "Predict Diseases" button in Streamlit.

    Args:
        image:
            Image path, PIL image, Streamlit UploadedFile,
            file-like object, bytes, or bytearray.

        view:
            Predicted image view: "AP" or "Lateral".

        models:
            Dictionary returned by load_disease_models().

        device:
            CPU or CUDA device.

    Returns:
        Structured disease prediction dictionary.
    """

    class_order = load_class_order()
    config = load_prediction_config()

    canonical_view = normalize_view_name(view)

    model = get_model_for_view(
        models=models,
        view=canonical_view,
    )

    threshold = get_threshold(config)

    # -------------------------------------------------------------
    # Preprocess image
    # -------------------------------------------------------------

    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(device)

    # -------------------------------------------------------------
    # Run disease model
    # -------------------------------------------------------------

    model.eval()

    with torch.inference_mode():
        model_output = model(image_tensor)

        # Support models that return a tuple or list.
        if isinstance(model_output, (tuple, list)):
            if not model_output:
                raise ValueError(
                    "The disease model returned an empty tuple/list."
                )

            logits = model_output[0]

        else:
            logits = model_output

        logits = validate_model_output(
            logits=logits,
            number_of_classes=len(class_order),
        )

        probabilities_tensor = torch.sigmoid(logits)

    # -------------------------------------------------------------
    # Convert tensors to Python values
    # -------------------------------------------------------------

    logits_list = (
        logits
        .squeeze(0)
        .detach()
        .cpu()
        .tolist()
    )

    probabilities = (
        probabilities_tensor
        .squeeze(0)
        .detach()
        .cpu()
        .tolist()
    )

    logits_list = [
        float(value)
        for value in logits_list
    ]

    probabilities = [
        float(value)
        for value in probabilities
    ]

    # -------------------------------------------------------------
    # Create disease-wise prediction dictionary
    # -------------------------------------------------------------

    predictions = {}

    for index, disease_name in enumerate(class_order):
        probability = probabilities[index]

        predictions[disease_name] = {
            "class_index": index,
            "logit": round(
                logits_list[index],
                6,
            ),
            "probability": round(
                probability,
                6,
            ),
            "probability_percent": round(
                probability * 100.0,
                2,
            ),
            "predicted": bool(
                probability >= threshold
            ),
        }

    # -------------------------------------------------------------
    # Rank diseases by probability
    # -------------------------------------------------------------

    ranked_findings = sorted(
        [
            {
                "disease": disease_name,
                "class_index": result["class_index"],
                "logit": result["logit"],
                "probability": result["probability"],
                "probability_percent": result[
                    "probability_percent"
                ],
                "predicted": result["predicted"],
            }
            for disease_name, result in predictions.items()
        ],
        key=lambda item: item["probability"],
        reverse=True,
    )

    # -------------------------------------------------------------
    # Separate positive and negative findings
    # -------------------------------------------------------------

    positive_findings = [
        item["disease"]
        for item in ranked_findings
        if item["predicted"]
    ]

    negative_findings = [
        item["disease"]
        for item in ranked_findings
        if not item["predicted"]
    ]

    # -------------------------------------------------------------
    # Final result
    # -------------------------------------------------------------

    result = {
        "module": "Module 1 - Disease Prediction",
        "view": canonical_view,
        "device": str(device),
        "activation": "sigmoid",
        "threshold": threshold,
        "number_of_classes": len(class_order),
        "class_order": class_order,
        "predictions": predictions,
        "positive_findings": positive_findings,
        "negative_findings": negative_findings,
        "ranked_findings": ranked_findings,
        "raw_logits": logits_list,
        "probabilities": probabilities,
    }

    return result


# ---------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------

def save_prediction_result(
    result: dict,
    output_name: str,
) -> Path:
    """
    Save a disease prediction result as JSON.

    Args:
        result:
            Prediction result dictionary.

        output_name:
            Output JSON filename.

    Returns:
        Path to the saved JSON file.
    """

    if not isinstance(output_name, str):
        raise TypeError(
            "output_name must be a string."
        )

    output_name = output_name.strip()

    if not output_name:
        raise ValueError(
            "output_name cannot be empty."
        )

    if not output_name.lower().endswith(".json"):
        output_name = f"{output_name}.json"

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIR / output_name

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=4,
        )

    return output_path


# ---------------------------------------------------------------------
# Console summary utility
# ---------------------------------------------------------------------

def print_prediction_summary(
    result: dict,
) -> None:
    """
    Print a readable disease prediction summary.

    This utility is optional and is not used by Streamlit.
    """

    print("\n" + "=" * 80)
    print("CXR CLINICAL AI - DISEASE PREDICTION")
    print("=" * 80)

    print(f"View: {result['view']}")
    print(f"Device: {result['device']}")
    print(f"Activation: {result['activation']}")
    print(f"Threshold: {result['threshold']}")
    print(
        "Number of classes: "
        f"{result['number_of_classes']}"
    )

    print("\nDisease probabilities:")
    print("-" * 80)

    for item in result["ranked_findings"]:
        status = (
            "POSITIVE"
            if item["predicted"]
            else "negative"
        )

        print(
            f"{item['disease']:<30} "
            f"{item['probability_percent']:>7.2f}% "
            f"[{status}]"
        )

    print("-" * 80)

    print("\nPositive findings:")

    if result["positive_findings"]:
        for disease in result["positive_findings"]:
            print(f"- {disease}")
    else:
        print("- None above the configured threshold")

    print("\nPrediction summary completed.")


# ---------------------------------------------------------------------
# Optional standalone test
# ---------------------------------------------------------------------

def main() -> None:
    """
    Optional manual prediction test.

    This function is not used by Streamlit. The main application
    will call predict_diseases() through pipeline.py.
    """

    sample_image = (
        PROJECT_ROOT
        / "sample_images"
        / "sample_01.jpg"
    )

    if not sample_image.is_file():
        raise FileNotFoundError(
            f"Sample image not found: {sample_image}"
        )

    print("Loading disease models...")

    models, device = load_disease_models()

    print("Running disease prediction...")

    result = predict_diseases(
        image=sample_image,
        view="AP",
        models=models,
        device=device,
    )

    print_prediction_summary(result)

    output_path = save_prediction_result(
        result=result,
        output_name="sample_01_ap_prediction.json",
    )

    print(
        f"\nPrediction JSON saved to: {output_path}"
    )

    print(
        "\nDisease prediction test completed successfully."
    )


if __name__ == "__main__":
    main()