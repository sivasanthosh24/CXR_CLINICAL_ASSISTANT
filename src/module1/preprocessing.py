"""
Image preprocessing utilities for Module 1.

This module is responsible only for:

    1. Loading an image
    2. Converting it to RGB
    3. Resizing it to 224 x 224
    4. Converting it to a tensor
    5. Applying ImageNet normalization
    6. Converting normalized tensors back to displayable images

"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------
# Preprocessing configuration
# ---------------------------------------------------------------------

IMAGE_SIZE = 224
IMAGE_CHANNELS = 3

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


# Supported image source types.
ImageSource = Union[
    str,
    Path,
    Image.Image,
    BinaryIO,
    bytes,
    bytearray,
]


# ---------------------------------------------------------------------
# Transform construction
# ---------------------------------------------------------------------

def build_inference_transform() -> transforms.Compose:
    """
    Build the preprocessing transform used during inference.

    Processing steps:
        1. Resize the image to 224 x 224.
        2. Convert the image to a tensor.
        3. Apply ImageNet normalization.

    RGB conversion is handled in load_image() before this transform.

    Returns:
        torchvision.transforms.Compose:
            Configured inference transform.
    """

    return transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


# A single reusable transform instance.
INFERENCE_TRANSFORM = build_inference_transform()


# ---------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------

def load_image(
    image_source: ImageSource,
) -> Image.Image:
    """
    Load an image and return it as an RGB PIL image.

    Supported input types:
        - File path as str
        - pathlib.Path
        - PIL.Image.Image
        - File-like object
        - bytes
        - bytearray

    This also supports Streamlit's UploadedFile object because it
    provides a file-like interface.

    Args:
        image_source:
            Image path, PIL image, file-like object, or image bytes.

    Returns:
        RGB PIL image.

    Raises:
        FileNotFoundError:
            If a supplied path does not exist.

        TypeError:
            If the source type is unsupported.

        ValueError:
            If the image cannot be opened.
    """

    # -------------------------------------------------------------
    # Case 1: Existing PIL image
    # -------------------------------------------------------------

    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")

    # -------------------------------------------------------------
    # Case 2: Raw bytes or bytearray
    # -------------------------------------------------------------

    if isinstance(image_source, (bytes, bytearray)):
        try:
            with Image.open(BytesIO(image_source)) as image:
                return image.convert("RGB")
        except Exception as error:
            raise ValueError(
                "Could not open the supplied image bytes."
            ) from error

    # -------------------------------------------------------------
    # Case 3: File-like object
    #
    # This includes Streamlit UploadedFile.
    # -------------------------------------------------------------

    if hasattr(image_source, "read"):
        try:
            # Reset the file pointer when supported.
            if hasattr(image_source, "seek"):
                image_source.seek(0)

            with Image.open(image_source) as image:
                return image.convert("RGB")

        except Exception as error:
            raise ValueError(
                "Could not open the supplied file-like image object."
            ) from error

    # -------------------------------------------------------------
    # Case 4: File path
    # -------------------------------------------------------------

    if isinstance(image_source, (str, Path)):
        image_path = Path(image_source)

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image file not found:\n{image_path}"
            )

        try:
            with Image.open(image_path) as image:
                return image.convert("RGB")

        except Exception as error:
            raise ValueError(
                f"Could not open image file:\n{image_path}"
            ) from error

    # -------------------------------------------------------------
    # Unsupported source type
    # -------------------------------------------------------------

    raise TypeError(
        "Unsupported image source type: "
        f"{type(image_source)}. "
        "Expected a file path, PIL image, file-like object, "
        "bytes, or bytearray."
    )


# ---------------------------------------------------------------------
# Tensor preprocessing
# ---------------------------------------------------------------------

def preprocess_image(
    image_source: ImageSource,
) -> torch.Tensor:
    """
    Preprocess an image for model inference.

    Processing steps:
        1. Load image.
        2. Convert to RGB.
        3. Resize to 224 x 224.
        4. Convert to tensor.
        5. Apply ImageNet normalization.
        6. Add batch dimension.

    Args:
        image_source:
            Image path, PIL image, file-like object, or image bytes.

    Returns:
        torch.Tensor:
            Tensor with shape [1, 3, 224, 224].
    """

    image = load_image(image_source)

    image_tensor = INFERENCE_TRANSFORM(image)

    image_tensor = image_tensor.unsqueeze(dim=0)

    if image_tensor.shape != (
        1,
        IMAGE_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ):
        raise RuntimeError(
            "Unexpected preprocessed image shape: "
            f"{tuple(image_tensor.shape)}"
        )

    return image_tensor


def preprocess_image_with_original(
    image_source: ImageSource,
) -> tuple[Image.Image, torch.Tensor]:
    """
    Return both the original RGB image and its preprocessed tensor.

    This function is useful for Grad-CAM and localization because
    the original image is required for displaying overlays.

    Args:
        image_source:
            Image path, PIL image, file-like object, or image bytes.

    Returns:
        original_image:
            RGB PIL image.

        image_tensor:
            Tensor with shape [1, 3, 224, 224].
    """

    original_image = load_image(image_source)

    image_tensor = INFERENCE_TRANSFORM(original_image)

    image_tensor = image_tensor.unsqueeze(dim=0)

    if image_tensor.shape != (
        1,
        IMAGE_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ):
        raise RuntimeError(
            "Unexpected preprocessed image shape: "
            f"{tuple(image_tensor.shape)}"
        )

    return original_image, image_tensor


# ---------------------------------------------------------------------
# Tensor-to-image conversion
# ---------------------------------------------------------------------

def tensor_to_display_image(
    image_tensor: torch.Tensor,
) -> np.ndarray:
    """
    Convert a normalized image tensor into a displayable RGB image.

    Expected input shapes:
        - [1, 3, H, W]
        - [3, H, W]

    The function reverses ImageNet normalization and converts the
    tensor into a uint8 NumPy array.

    Args:
        image_tensor:
            Normalized image tensor.

    Returns:
        np.ndarray:
            RGB image with shape [H, W, 3] and dtype uint8.
    """

    tensor = image_tensor.detach().cpu()

    # Remove batch dimension if present.
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(
                "Expected a single image when using a 4D tensor. "
                f"Received shape: {tuple(tensor.shape)}"
            )

        tensor = tensor[0]

    if tensor.ndim != 3:
        raise ValueError(
            "Expected tensor shape [3, H, W] or [1, 3, H, W]. "
            f"Received shape: {tuple(tensor.shape)}"
        )

    if tensor.shape[0] != IMAGE_CHANNELS:
        raise ValueError(
            "Expected 3 image channels. "
            f"Received shape: {tuple(tensor.shape)}"
        )

    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=tensor.dtype,
    ).view(IMAGE_CHANNELS, 1, 1)

    std = torch.tensor(
        IMAGENET_STD,
        dtype=tensor.dtype,
    ).view(IMAGE_CHANNELS, 1, 1)

    # Reverse ImageNet normalization.
    tensor = tensor * std + mean

    # Ensure valid display range.
    tensor = tensor.clamp(
        min=0.0,
        max=1.0,
    )

    # Convert from [C, H, W] to [H, W, C].
    image_array = (
        tensor.permute(1, 2, 0)
        .numpy()
        * 255.0
    )

    image_array = image_array.astype(np.uint8)

    return image_array


# ---------------------------------------------------------------------
# Configuration and metadata
# ---------------------------------------------------------------------

def get_preprocessing_summary() -> dict:
    """
    Return the preprocessing configuration.

    This is useful for:
        - Logging
        - Reproducibility
        - JSON metadata
        - Streamlit display
    """

    return {
        "image_size": IMAGE_SIZE,
        "channels": IMAGE_CHANNELS,
        "color_mode": "RGB",
        "normalization": "ImageNet",
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
    }