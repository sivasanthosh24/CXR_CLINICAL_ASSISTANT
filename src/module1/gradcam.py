
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .model_loader import load_disease_models


# ---------------------------------------------------------------------
# Project path configuration
# ---------------------------------------------------------------------

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]
PROJECT_ROOT = CURRENT_FILE.parents[2]

CONFIG_DIR = PROJECT_ROOT / "config"
CLASS_ORDER_PATH = CONFIG_DIR / "class_order.json"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gradcam"


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

SUPPORTED_VIEWS = ("AP", "Lateral")

DEFAULT_IMAGE_SIZE = 224

DEFAULT_ALPHA = 0.45

# This order must match the trained model output order.
FALLBACK_CLASS_NAMES = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]

ImageSource = Union[
    str,
    Path,
    bytes,
    bytearray,
    io.BytesIO,
    Image.Image,
]


# ---------------------------------------------------------------------
# Class-order utilities
# ---------------------------------------------------------------------

def _extract_class_names_from_config(config_data: Any) -> list[str]:
    """
    Extract class names from common JSON configuration formats.

    Supported formats:

    1. List:
        [
            "Cardiomegaly",
            "Pneumonia"
        ]

    2. Dictionary:
        {
            "class_names": [
                "Cardiomegaly",
                "Pneumonia"
            ]
        }

    3. Index-to-name dictionary:
        {
            "0": "Cardiomegaly",
            "1": "Pneumonia"
        }
    """
    if isinstance(config_data, list):
        class_names = config_data

    elif isinstance(config_data, dict):
        possible_keys = (
            "class_names",
            "classes",
            "labels",
            "class_order",
            "disease_names",
        )

        class_names = None

        for key in possible_keys:
            if key in config_data:
                class_names = config_data[key]
                break

        if class_names is None:
            numeric_keys = list(config_data.keys())

            if numeric_keys and all(
                str(key).isdigit() for key in numeric_keys
            ):
                sorted_keys = sorted(
                    numeric_keys,
                    key=lambda key: int(key),
                )
                class_names = [
                    config_data[key]
                    for key in sorted_keys
                ]

        if class_names is None:
            raise ValueError(
                "The class-order configuration does not contain a supported "
                "class-name field."
            )

    else:
        raise TypeError(
            "Class-order configuration must be a list or dictionary."
        )

    if not isinstance(class_names, list):
        raise TypeError(
            "Class names extracted from the configuration must be a list."
        )

    cleaned_names = []

    for index, class_name in enumerate(class_names):
        if not isinstance(class_name, str):
            raise TypeError(
                f"Class name at index {index} must be a string. "
                f"Received: {type(class_name)}"
            )

        cleaned_name = class_name.strip()

        if not cleaned_name:
            raise ValueError(
                f"Class name at index {index} is empty."
            )

        cleaned_names.append(cleaned_name)

    if not cleaned_names:
        raise ValueError(
            "The class-order configuration contains no class names."
        )

    return cleaned_names


def load_class_names(
    class_order_path: str | Path = CLASS_ORDER_PATH,
) -> list[str]:
    """
    Load the trained model's class order.

    The function first attempts to read config/class_order.json.
    If the file is unavailable, it uses the validated fallback order.

    Returns:
        Ordered list of disease/class names.
    """
    class_order_path = Path(class_order_path)

    if not class_order_path.exists():
        return FALLBACK_CLASS_NAMES.copy()

    try:
        with class_order_path.open("r", encoding="utf-8") as file:
            config_data = json.load(file)

        class_names = _extract_class_names_from_config(config_data)

    except Exception as error:
        raise RuntimeError(
            f"Failed to load class order from '{class_order_path}'."
        ) from error

    return class_names


DEFAULT_CLASS_NAMES = load_class_names()


def resolve_class_index(
    class_name: str,
    class_names: Sequence[str],
) -> int:
    """
    Resolve a disease name to its exact trained class index.
    """
    normalized_name = class_name.strip().lower()

    for index, available_name in enumerate(class_names):
        if available_name.strip().lower() == normalized_name:
            return index

    available_classes = ", ".join(class_names)

    raise ValueError(
        f"Disease '{class_name}' was not found in the class order. "
        f"Available classes: {available_classes}"
    )


# ---------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------

def build_image_transform(
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> transforms.Compose:
    """
    Build the preprocessing pipeline used for model inference.

    The transform:
        1. Converts the image to RGB.
        2. Resizes it to image_size x image_size.
        3. Converts it to a tensor.
        4. Applies ImageNet normalization.
    """
    if image_size <= 0:
        raise ValueError(
            f"image_size must be positive. Received: {image_size}"
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_pil_image(image_source: ImageSource) -> Image.Image:
    """
    Load an image from a path, bytes, file-like object, or PIL image.

    Returns:
        RGB PIL image.
    """
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")

    if isinstance(image_source, (bytes, bytearray)):
        image_source = io.BytesIO(image_source)

    if isinstance(image_source, (str, Path)):
        image_path = Path(image_source)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with Image.open(image_path) as image:
            return image.convert("RGB").copy()

    if hasattr(image_source, "seek"):
        try:
            image_source.seek(0)
        except Exception:
            pass

    if hasattr(image_source, "read"):
        with Image.open(image_source) as image:
            return image.convert("RGB").copy()

    raise TypeError(
        "Unsupported image source. Expected a file path, bytes, "
        "file-like object, or PIL.Image.Image."
    )


def load_image(
    image_source: ImageSource,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> Tuple[Image.Image, torch.Tensor]:
    """
    Load an image and prepare it for model inference.

    Returns:
        original_image:
            RGB PIL image.

        image_tensor:
            Tensor with shape [1, 3, H, W].
    """
    original_image = load_pil_image(image_source)

    transform = build_image_transform(image_size)

    image_tensor = transform(original_image).unsqueeze(0)

    return original_image, image_tensor


# ---------------------------------------------------------------------
# Model output handling
# ---------------------------------------------------------------------

def extract_logits(output: Any) -> torch.Tensor:
    """
    Extract logits from different possible model output formats.

    Supported formats:
        - Tensor
        - Tuple
        - List
        - Dictionary

    Returns:
        Tensor with shape [batch_size, num_classes].
    """
    if isinstance(output, torch.Tensor):
        logits = output

    elif isinstance(output, (tuple, list)):
        if len(output) == 0:
            raise ValueError(
                "Model returned an empty tuple/list."
            )

        logits = None

        for item in output:
            if isinstance(item, torch.Tensor):
                logits = item
                break

        if logits is None:
            raise TypeError(
                "Tuple/list model output does not contain a tensor."
            )

    elif isinstance(output, dict):
        preferred_keys = [
            "logits",
            "output",
            "outputs",
            "predictions",
            "prediction",
            "scores",
        ]

        logits = None

        for key in preferred_keys:
            if key in output and isinstance(output[key], torch.Tensor):
                logits = output[key]
                break

        if logits is None:
            for value in output.values():
                if isinstance(value, torch.Tensor):
                    logits = value
                    break

        if logits is None:
            raise TypeError(
                "Dictionary model output does not contain a tensor."
            )

    else:
        raise TypeError(
            f"Unsupported model output type: {type(output)}"
        )

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    if logits.ndim != 2:
        raise ValueError(
            "Expected logits with two dimensions "
            f"[batch_size, num_classes]. Received: {tuple(logits.shape)}"
        )

    return logits


# ---------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------

def normalize_view_name(view: str) -> str:
    """
    Normalize a view name to one of the supported values.
    """
    if not isinstance(view, str):
        raise TypeError(
            f"view must be a string. Received: {type(view)}"
        )

    normalized_view = view.strip().lower()

    if normalized_view == "ap":
        return "AP"

    if normalized_view in {"lateral", "lat"}:
        return "Lateral"

    raise ValueError(
        f"Unsupported view '{view}'. "
        f"Choose one of: {', '.join(SUPPORTED_VIEWS)}."
    )


def get_model_for_view(
    models: Dict[str, torch.nn.Module],
    view: str,
) -> torch.nn.Module:
    """
    Select the model corresponding to the requested radiograph view.
    """
    normalized_view = normalize_view_name(view)

    if not isinstance(models, dict):
        raise TypeError(
            "Expected models to be a dictionary returned by "
            "load_disease_models()."
        )

    if normalized_view not in models:
        available_models = ", ".join(
            str(key) for key in models.keys()
        )

        raise KeyError(
            f"Model for view '{normalized_view}' was not found. "
            f"Available models: {available_models}"
        )

    model = models[normalized_view]

    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"Model for view '{normalized_view}' is not a "
            f"torch.nn.Module. Received: {type(model)}"
        )

    return model


# ---------------------------------------------------------------------
# Grad-CAM target-layer discovery
# ---------------------------------------------------------------------

def find_resnet_layer4(
    model: torch.nn.Module,
) -> Optional[torch.nn.Module]:
    """
    Find the final convolutional feature block for ResNet-like models.

    Supported structures:

    1. Standard ResNet:
        model.layer4

    2. Wrapped ResNet:
        model.backbone.layer4
        model.model.layer4
        model.encoder.layer4
        model.base_model.layer4

    3. Custom LateralResNet50:
        model.features[-1]

    The custom lateral model is expected to define:

        model.features = nn.Sequential(
            conv1,
            bn1,
            relu,
            maxpool,
            layer1,
            layer2,
            layer3,
            layer4,
        )
    """
    if model is None:
        return None

    if hasattr(model, "layer4"):
        layer4 = getattr(model, "layer4")

        if isinstance(layer4, torch.nn.Module):
            return layer4

    if hasattr(model, "features"):
        features = getattr(model, "features")

        if isinstance(features, torch.nn.Sequential):
            if len(features) > 0:
                candidate = features[-1]

                if isinstance(candidate, torch.nn.Module):
                    return candidate

    for attribute_name in (
        "backbone",
        "model",
        "encoder",
        "base_model",
    ):
        if hasattr(model, attribute_name):
            nested_model = getattr(model, attribute_name)

            if isinstance(nested_model, torch.nn.Module):
                target_layer = find_resnet_layer4(nested_model)

                if target_layer is not None:
                    return target_layer

    return None


def find_densenet_target_layer(
    model: torch.nn.Module,
) -> Optional[torch.nn.Module]:
    """
    Find the final convolutional feature block for DenseNet-like models.

    The preferred target is:

        model.features.denseblock4

    A sequential feature fallback is also supported.
    """
    if model is None:
        return None

    if hasattr(model, "features"):
        features = getattr(model, "features")

        if hasattr(features, "denseblock4"):
            target_layer = getattr(features, "denseblock4")

            if isinstance(target_layer, torch.nn.Module):
                return target_layer

        if isinstance(features, torch.nn.Sequential):
            if len(features) > 0:
                candidate = features[-1]

                if isinstance(candidate, torch.nn.Module):
                    return candidate

    for attribute_name in (
        "backbone",
        "model",
        "encoder",
        "base_model",
    ):
        if hasattr(model, attribute_name):
            nested_model = getattr(model, attribute_name)

            if isinstance(nested_model, torch.nn.Module):
                target_layer = find_densenet_target_layer(
                    nested_model
                )

                if target_layer is not None:
                    return target_layer

    return None


def get_gradcam_target_layer(
    model: torch.nn.Module,
    view: str,
) -> torch.nn.Module:
    """
    Select an appropriate target layer for Grad-CAM.

    AP:
        DenseNet121 final convolutional block.

    Lateral:
        Custom ResNet50 final feature block.
    """
    normalized_view = normalize_view_name(view)

    if normalized_view == "AP":
        target_layer = find_densenet_target_layer(model)

        if target_layer is not None:
            return target_layer

        raise ValueError(
            "Could not find a suitable DenseNet target layer "
            "for the AP model."
        )

    if normalized_view == "Lateral":
        target_layer = find_resnet_layer4(model)

        if target_layer is not None:
            return target_layer

        raise ValueError(
            "Could not find a suitable ResNet target layer "
            "for the Lateral model."
        )

    raise ValueError(
        f"Unsupported view '{view}'."
    )


# ---------------------------------------------------------------------
# Grad-CAM implementation
# ---------------------------------------------------------------------

class GradCAM:
    """
    Basic Grad-CAM implementation using forward and backward hooks.

    Algorithm:
        1. Capture activations from the target layer.
        2. Capture gradients of the selected class score.
        3. Global-average-pool the gradients.
        4. Weight the activations using the pooled gradients.
        5. Apply ReLU.
        6. Resize the heatmap to the input image size.
        7. Normalize the heatmap to [0, 1].
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
    ) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                "model must be a torch.nn.Module."
            )

        if not isinstance(target_layer, torch.nn.Module):
            raise TypeError(
                "target_layer must be a torch.nn.Module."
            )

        self.model = model
        self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self.forward_handle = self.target_layer.register_forward_hook(
            self._forward_hook
        )

        self.backward_handle = (
            self.target_layer.register_full_backward_hook(
                self._backward_hook
            )
        )

    def _forward_hook(
        self,
        module: torch.nn.Module,
        inputs: Tuple[Any, ...],
        output: Any,
    ) -> None:
        """
        Save target-layer activations.
        """
        if isinstance(output, torch.Tensor):
            self.activations = output.detach()

    def _backward_hook(
        self,
        module: torch.nn.Module,
        grad_input: Tuple[Any, ...],
        grad_output: Tuple[Any, ...],
    ) -> None:
        """
        Save target-layer gradients.
        """
        if grad_output is None:
            return

        if len(grad_output) == 0:
            return

        gradient = grad_output[0]

        if isinstance(gradient, torch.Tensor):
            self.gradients = gradient.detach()

    def remove_hooks(self) -> None:
        """
        Remove registered hooks.
        """
        if self.forward_handle is not None:
            self.forward_handle.remove()
            self.forward_handle = None

        if self.backward_handle is not None:
            self.backward_handle.remove()
            self.backward_handle = None

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_index: Optional[int] = None,
    ) -> Tuple[np.ndarray, torch.Tensor, int]:
        """
        Generate a Grad-CAM heatmap.

        Args:
            input_tensor:
                Input image tensor with shape [1, 3, H, W].

            class_index:
                Class index to explain. If None, the class with the
                highest logit is selected.

        Returns:
            heatmap:
                NumPy array with values in [0, 1] and shape [H, W].

            logits:
                Model logits with shape [1, num_classes].

            selected_class:
                Class index used for Grad-CAM.
        """
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError(
                "input_tensor must be a torch.Tensor."
            )

        if input_tensor.ndim != 4:
            raise ValueError(
                "Grad-CAM input must have shape [B, C, H, W]. "
                f"Received: {tuple(input_tensor.shape)}"
            )

        if input_tensor.shape[0] != 1:
            raise ValueError(
                "Grad-CAM currently supports batch size 1 only. "
                f"Received batch size: {input_tensor.shape[0]}"
            )

        self.activations = None
        self.gradients = None

        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            output = self.model(input_tensor)
            logits = extract_logits(output)

            if logits.shape[0] != 1:
                raise ValueError(
                    "Grad-CAM currently supports one input image at a time. "
                    f"Received logits shape: {tuple(logits.shape)}"
                )

            if class_index is None:
                selected_class = int(
                    torch.argmax(logits, dim=1).item()
                )
            else:
                selected_class = int(class_index)

            if selected_class < 0:
                raise IndexError(
                    f"class_index must be non-negative. "
                    f"Received: {selected_class}"
                )

            if selected_class >= logits.shape[1]:
                raise IndexError(
                    f"class_index {selected_class} is outside the valid "
                    f"class range [0, {logits.shape[1] - 1}]."
                )

            selected_logit = logits[:, selected_class].sum()

            selected_logit.backward()

        if self.activations is None:
            raise RuntimeError(
                "Grad-CAM activations were not captured. "
                "Check the selected target layer."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients were not captured. "
                "Check whether the target layer participates "
                "in the forward pass."
            )

        activations = self.activations
        gradients = self.gradients

        if activations.ndim != 4:
            raise ValueError(
                "Grad-CAM requires 4D activations with shape "
                f"[B, C, H, W]. Received: {tuple(activations.shape)}"
            )

        if gradients.ndim != 4:
            raise ValueError(
                "Grad-CAM requires 4D gradients with shape "
                f"[B, C, H, W]. Received: {tuple(gradients.shape)}"
            )

        if activations.shape != gradients.shape:
            raise ValueError(
                "Activation and gradient shapes do not match. "
                f"Activations: {tuple(activations.shape)}, "
                f"Gradients: {tuple(gradients.shape)}"
            )

        # Global average pooling over spatial dimensions.
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        # Weighted sum of feature maps.
        cam = (weights * activations).sum(
            dim=1,
            keepdim=True,
        )

        # Grad-CAM applies ReLU.
        cam = F.relu(cam)

        # Resize CAM to input spatial dimensions.
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0]

        # Normalize to [0, 1].
        cam_min = cam.min()
        cam_max = cam.max()

        if float(cam_max - cam_min) > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        heatmap = (
            cam.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        return heatmap, logits.detach(), selected_class

    def __del__(self) -> None:
        """
        Best-effort hook cleanup.
        """
        try:
            self.remove_hooks()
        except Exception:
            pass


# ---------------------------------------------------------------------
# Prediction utilities
# ---------------------------------------------------------------------

def sigmoid_probabilities(
    logits: torch.Tensor,
) -> torch.Tensor:
    """
    Convert logits into independent sigmoid probabilities.

    This is appropriate for multi-label disease classification.
    """
    return torch.sigmoid(logits)


def get_prediction_summary(
    logits: torch.Tensor,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Convert logits into a structured prediction summary.
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    probabilities = sigmoid_probabilities(logits)

    if probabilities.ndim != 2:
        raise ValueError(
            "Expected probabilities with shape [B, C]. "
            f"Received: {tuple(probabilities.shape)}"
        )

    if probabilities.shape[0] != 1:
        raise ValueError(
            "Prediction summary currently supports batch size 1."
        )

    probabilities = probabilities[0]

    num_classes = int(probabilities.shape[0])

    if len(class_names) != num_classes:
        raise ValueError(
            "The number of class names does not match the model output. "
            f"Class names: {len(class_names)}, "
            f"Model outputs: {num_classes}"
        )

    predictions = []

    for index in range(num_classes):
        probability = float(
            probabilities[index].item()
        )

        predictions.append(
            {
                "class_index": index,
                "class_name": class_names[index],
                "probability": probability,
                "percentage": probability * 100.0,
            }
        )

    predictions.sort(
        key=lambda item: item["probability"],
        reverse=True,
    )

    return {
        "num_classes": num_classes,
        "predictions": predictions,
        "top_prediction": (
            predictions[0]
            if predictions
            else None
        ),
    }


# ---------------------------------------------------------------------
# Heatmap visualization utilities
# ---------------------------------------------------------------------

def prepare_original_image_for_overlay(
    original_image: Image.Image,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """
    Convert a PIL image into an OpenCV-compatible BGR image.
    """
    width, height = target_size

    resized_image = original_image.resize(
        (width, height),
        Image.Resampling.BILINEAR,
    )

    rgb_array = np.array(resized_image)

    if rgb_array.ndim == 2:
        rgb_array = np.stack(
            [rgb_array, rgb_array, rgb_array],
            axis=-1,
        )

    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError(
            "Expected an RGB image with shape [H, W, 3]. "
            f"Received: {rgb_array.shape}"
        )

    bgr_array = cv2.cvtColor(
        rgb_array,
        cv2.COLOR_RGB2BGR,
    )

    return bgr_array


def create_heatmap_image(
    heatmap: np.ndarray,
    image_size: Tuple[int, int],
) -> np.ndarray:
    """
    Convert a normalized heatmap into a colored OpenCV heatmap.
    """
    if not isinstance(heatmap, np.ndarray):
        raise TypeError(
            "heatmap must be a NumPy array."
        )

    if heatmap.ndim != 2:
        raise ValueError(
            "heatmap must have shape [H, W]. "
            f"Received: {heatmap.shape}"
        )

    width, height = image_size

    heatmap_resized = cv2.resize(
        heatmap,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    heatmap_uint8 = np.clip(
        heatmap_resized * 255.0,
        0,
        255,
    ).astype(np.uint8)

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    return colored_heatmap


def overlay_heatmap(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create:

        1. Original image
        2. Colored heatmap
        3. Overlay image

    Returns:
        original_bgr:
            Original image in BGR format.

        colored_heatmap:
            Colored heatmap in BGR format.

        overlay_bgr:
            Heatmap overlay in BGR format.
    """
    width, height = original_image.size

    original_bgr = prepare_original_image_for_overlay(
        original_image=original_image,
        target_size=(width, height),
    )

    colored_heatmap = create_heatmap_image(
        heatmap=heatmap,
        image_size=(width, height),
    )

    alpha = float(np.clip(alpha, 0.0, 1.0))

    overlay_bgr = cv2.addWeighted(
        original_bgr,
        1.0 - alpha,
        colored_heatmap,
        alpha,
        0,
    )

    return (
        original_bgr,
        colored_heatmap,
        overlay_bgr,
    )


def create_side_by_side_visualization(
    original_bgr: np.ndarray,
    heatmap_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
) -> np.ndarray:
    """
    Create a horizontal visualization containing:

        Original | Heatmap | Overlay
    """
    if original_bgr.ndim != 3:
        raise ValueError(
            "original_bgr must be a three-dimensional image array."
        )

    target_height = original_bgr.shape[0]
    target_width = original_bgr.shape[1]

    if heatmap_bgr.shape != original_bgr.shape:
        heatmap_bgr = cv2.resize(
            heatmap_bgr,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )

    if overlay_bgr.shape != original_bgr.shape:
        overlay_bgr = cv2.resize(
            overlay_bgr,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )

    combined = np.concatenate(
        [
            original_bgr,
            heatmap_bgr,
            overlay_bgr,
        ],
        axis=1,
    )

    return combined


def save_visualization(
    original_image: Image.Image,
    heatmap: np.ndarray,
    output_path: str | Path,
    alpha: float = DEFAULT_ALPHA,
) -> None:
    """
    Save the combined Grad-CAM visualization to disk.
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_bgr, heatmap_bgr, overlay_bgr = overlay_heatmap(
        original_image=original_image,
        heatmap=heatmap,
        alpha=alpha,
    )

    combined = create_side_by_side_visualization(
        original_bgr=original_bgr,
        heatmap_bgr=heatmap_bgr,
        overlay_bgr=overlay_bgr,
    )

    success = cv2.imwrite(
        str(output_path),
        combined,
    )

    if not success:
        raise IOError(
            f"Failed to save Grad-CAM visualization: {output_path}"
        )


def save_individual_visualizations(
    original_image: Image.Image,
    heatmap: np.ndarray,
    output_dir: str | Path,
    stem: str,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, str]:
    """
    Save the original image, heatmap, overlay, and combined image
    separately.
    """
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_bgr, heatmap_bgr, overlay_bgr = overlay_heatmap(
        original_image=original_image,
        heatmap=heatmap,
        alpha=alpha,
    )

    combined = create_side_by_side_visualization(
        original_bgr=original_bgr,
        heatmap_bgr=heatmap_bgr,
        overlay_bgr=overlay_bgr,
    )

    original_path = output_dir / f"{stem}_original.png"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    overlay_path = output_dir / f"{stem}_overlay.png"
    combined_path = output_dir / f"{stem}_combined.png"

    write_results = {
        "original": cv2.imwrite(
            str(original_path),
            original_bgr,
        ),
        "heatmap": cv2.imwrite(
            str(heatmap_path),
            heatmap_bgr,
        ),
        "overlay": cv2.imwrite(
            str(overlay_path),
            overlay_bgr,
        ),
        "combined": cv2.imwrite(
            str(combined_path),
            combined,
        ),
    }

    failed_outputs = [
        name
        for name, success in write_results.items()
        if not success
    ]

    if failed_outputs:
        raise IOError(
            "Failed to save the following Grad-CAM outputs: "
            + ", ".join(failed_outputs)
        )

    return {
        "original": str(original_path),
        "heatmap": str(heatmap_path),
        "overlay": str(overlay_path),
        "combined": str(combined_path),
    }


# ---------------------------------------------------------------------
# JSON result utilities
# ---------------------------------------------------------------------

def make_json_serializable(value: Any) -> Any:
    """
    Convert common NumPy and PyTorch values into JSON-compatible values.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()

        return value.detach().cpu().tolist()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

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

    if isinstance(value, Path):
        return str(value)

    return value


def save_result_json(
    result: Dict[str, Any],
    output_path: str | Path,
) -> None:
    """
    Save prediction and Grad-CAM metadata as JSON.
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serializable_result = make_json_serializable(result)

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


# ---------------------------------------------------------------------
# Main Grad-CAM pipeline
# ---------------------------------------------------------------------

def run_gradcam(
    image_source: ImageSource,
    view: str,
    disease: Optional[str] = None,
    class_index: Optional[int] = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    image_size: int = DEFAULT_IMAGE_SIZE,
    alpha: float = DEFAULT_ALPHA,
    save_individual: bool = True,
    class_order_path: str | Path = CLASS_ORDER_PATH,
) -> Dict[str, Any]:
    """
    Run Grad-CAM for one radiograph.

    Args:
        image_source:
            Image path, bytes, file-like object, or PIL image.

        view:
            AP or Lateral.

        disease:
            Optional disease name. If supplied, the class index is
            selected using the trained class order unless class_index
            is explicitly provided.

        class_index:
            Optional numeric class index to explain.

        output_dir:
            Directory for generated files.

        image_size:
            Input image size.

        alpha:
            Heatmap overlay transparency.

        save_individual:
            Whether to save original, heatmap, overlay, and combined images.

        class_order_path:
            Path to the class-order configuration file.

    Returns:
        Dictionary containing prediction and output metadata.
    """
    normalized_view = normalize_view_name(view)

    if image_size <= 0:
        raise ValueError(
            f"image_size must be positive. Received: {image_size}"
        )

    if class_index is not None:
        class_index = int(class_index)

        if class_index < 0:
            raise ValueError(
                f"class_index must be non-negative. "
                f"Received: {class_index}"
            )

    output_dir = Path(output_dir)

    class_names = load_class_names(
        class_order_path=class_order_path,
    )

    # -------------------------------------------------------------
    # Load models and device
    # -------------------------------------------------------------

    loaded_models = load_disease_models()

    if not isinstance(loaded_models, tuple):
        raise TypeError(
            "load_disease_models() must return "
            "(models, device)."
        )

    if len(loaded_models) != 2:
        raise ValueError(
            "load_disease_models() must return exactly two values: "
            "(models, device)."
        )

    models, device = loaded_models

    if not isinstance(device, torch.device):
        raise TypeError(
            "The second value returned by load_disease_models() "
            "must be a torch.device."
        )

    model = get_model_for_view(
        models=models,
        view=normalized_view,
    )

    model = model.to(device)
    model.eval()

    # -------------------------------------------------------------
    # Load image
    # -------------------------------------------------------------

    original_image, image_tensor = load_image(
        image_source=image_source,
        image_size=image_size,
    )

    image_tensor = image_tensor.to(device)

    # -------------------------------------------------------------
    # Determine a stable image name
    # -------------------------------------------------------------

    if isinstance(image_source, (str, Path)):
        image_path = Path(image_source)
        image_name = image_path.name
        image_stem = image_path.stem
    else:
        image_name = "uploaded_image"
        image_stem = "uploaded_image"

    # -------------------------------------------------------------
    # Prediction pass
    # -------------------------------------------------------------

    with torch.no_grad():
        prediction_output = model(image_tensor)
        prediction_logits = extract_logits(prediction_output)

    if prediction_logits.shape[1] != len(class_names):
        raise ValueError(
            "The model output dimension does not match the class order. "
            f"Model outputs: {prediction_logits.shape[1]}, "
            f"Class names: {len(class_names)}"
        )

    prediction_summary = get_prediction_summary(
        logits=prediction_logits,
        class_names=class_names,
    )

    predicted_class_index = int(
        torch.argmax(
            prediction_logits,
            dim=1,
        ).item()
    )

    # -------------------------------------------------------------
    # Determine target class
    # -------------------------------------------------------------

    selected_class_index = class_index

    if selected_class_index is None and disease is not None:
        selected_class_index = resolve_class_index(
            class_name=disease,
            class_names=class_names,
        )

    if selected_class_index is None:
        selected_class_index = predicted_class_index

    if selected_class_index < 0:
        raise ValueError(
            f"class_index must be non-negative. "
            f"Received: {selected_class_index}"
        )

    if selected_class_index >= prediction_logits.shape[1]:
        raise ValueError(
            f"class_index {selected_class_index} is outside the model's "
            f"class range [0, {prediction_logits.shape[1] - 1}]."
        )

    selected_class_name = class_names[selected_class_index]

    # -------------------------------------------------------------
    # Select target layer
    # -------------------------------------------------------------

    target_layer = get_gradcam_target_layer(
        model=model,
        view=normalized_view,
    )

    # -------------------------------------------------------------
    # Generate Grad-CAM
    # -------------------------------------------------------------

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    try:
        heatmap, gradcam_logits, explained_class_index = (
            gradcam.generate(
                input_tensor=image_tensor,
                class_index=selected_class_index,
            )
        )
    finally:
        gradcam.remove_hooks()

    # -------------------------------------------------------------
    # Prepare output names
    # -------------------------------------------------------------

    output_stem = (
        f"{image_stem}_{normalized_view}_"
        f"class_{explained_class_index}"
    )

    combined_output_path = (
        output_dir / f"{output_stem}_combined.png"
    )

    json_output_path = (
        output_dir / f"{output_stem}.json"
    )

    # -------------------------------------------------------------
    # Save visualization
    # -------------------------------------------------------------

    save_visualization(
        original_image=original_image,
        heatmap=heatmap,
        output_path=combined_output_path,
        alpha=alpha,
    )

    individual_paths: Dict[str, str] = {}

    if save_individual:
        individual_paths = save_individual_visualizations(
            original_image=original_image,
            heatmap=heatmap,
            output_dir=output_dir,
            stem=output_stem,
            alpha=alpha,
        )

    # -------------------------------------------------------------
    # Build result
    # -------------------------------------------------------------

    result = {
        "image_path": image_name,
        "view": normalized_view,
        "device": str(device),
        "image_size": image_size,
        "class_order_path": str(class_order_path),
        "class_names": class_names,
        "target_layer": target_layer.__class__.__name__,
        "disease_requested": disease,
        "predicted_class_index": predicted_class_index,
        "predicted_class_name": class_names[
            predicted_class_index
        ],
        "explained_class_index": explained_class_index,
        "explained_class_name": selected_class_name,
        "prediction_summary": prediction_summary,
        "gradcam_logits": gradcam_logits,
        "output_paths": {
            "combined": str(combined_output_path),
            **individual_paths,
        },
    }

    save_result_json(
        result=result,
        output_path=json_output_path,
    )

    result["output_paths"]["json"] = str(json_output_path)

    return result


# ---------------------------------------------------------------------
# Optional command-line interface
# ---------------------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate Grad-CAM visualizations for "
            "Module 1 disease classification."
        )
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input chest X-ray image.",
    )

    parser.add_argument(
        "--view",
        type=str,
        required=True,
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
        type=str,
        default=None,
        help=(
            "Disease/class name to explain, for example "
            "'Cardiomegaly' or 'Pneumonia'."
        ),
    )

    parser.add_argument(
        "--class-index",
        type=int,
        default=None,
        help="Numeric class index to explain.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where Grad-CAM outputs will be saved.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Input image size used by the model.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Heatmap overlay transparency between 0 and 1.",
    )

    parser.add_argument(
        "--class-order",
        type=str,
        default=str(CLASS_ORDER_PATH),
        help="Path to the class-order JSON file.",
    )

    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="Save only the combined visualization.",
    )

    return parser


def main() -> None:
    """
    Command-line entry point.
    """
    parser = build_argument_parser()
    args = parser.parse_args()

    result = run_gradcam(
        image_source=args.image,
        view=args.view,
        disease=args.disease,
        class_index=args.class_index,
        output_dir=args.output_dir,
        image_size=args.image_size,
        alpha=args.alpha,
        save_individual=not args.no_individual,
        class_order_path=args.class_order,
    )

    print("\nGrad-CAM completed successfully.")
    print(f"Image: {result['image_path']}")
    print(f"View: {result['view']}")
    print(f"Device: {result['device']}")
    print(f"Target layer: {result['target_layer']}")

    print(
        "Predicted class: "
        f"{result['predicted_class_name']} "
        f"(index={result['predicted_class_index']})"
    )

    print(
        "Explained class: "
        f"{result['explained_class_name']} "
        f"(index={result['explained_class_index']})"
    )

    print("\nOutput files:")

    for name, path in result["output_paths"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()