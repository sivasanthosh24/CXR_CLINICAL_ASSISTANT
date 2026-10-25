from pathlib import Path
import torch
from PIL import Image
BIOMEDCLIP_MEAN = (
    0.48145466,
    0.4578275,
    0.40821073
)
BIOMEDCLIP_STD = (
    0.26862954,
    0.26130258,
    0.27577711
)
IMAGE_SIZE = 224
def load_image(
    image_path
):
    image_path = Path(
        image_path
    )
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )
    try:
        image = Image.open(
            image_path
        ).convert("RGB")
    except Exception as exc:
        raise RuntimeError(
            f"Unable to read image: "
            f"{image_path}"
        ) from exc
    return image
def preprocess_image(
    image,
    preprocess=None
):
    if isinstance(
        image,
        (str, Path)
    ):
        image = load_image(
            image
        )
    elif isinstance(
        image,
        Image.Image
    ):
        image = image.convert(
            "RGB"
        )
    else:
        raise TypeError(
            "image must be a file path "
            "or PIL.Image.Image"
        )
    if preprocess is None:
        raise ValueError(
            "The BiomedCLIP preprocessing "
            "transform must be provided."
        )
    tensor = preprocess(
        image
    )
    if not isinstance(
        tensor,
        torch.Tensor
    ):
        raise TypeError(
            "BiomedCLIP preprocessing must "
            "return a torch.Tensor."
        )

    if tensor.ndim != 3:
        raise RuntimeError(
            "Expected preprocessed image "
            f"shape [3,H,W], got "
            f"{tuple(tensor.shape)}"
        )

    if tensor.shape[0] != 3:

        raise RuntimeError(
            "Expected 3-channel RGB input, "
            f"got {tensor.shape[0]} channels."
        )

    if (
        tensor.shape[1] != IMAGE_SIZE
        or
        tensor.shape[2] != IMAGE_SIZE
    ):

        raise RuntimeError(
            "Expected BiomedCLIP input size "
            f"{IMAGE_SIZE}x{IMAGE_SIZE}, "
            f"got "
            f"{tensor.shape[1]}x"
            f"{tensor.shape[2]}"
        )

    return tensor


def prepare_image_batch(
    image,
    preprocess,
    device
):

    tensor = preprocess_image(
        image=image,
        preprocess=preprocess
    )

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        device=device,
        dtype=torch.float32
    )

    return tensor


__all__ = [
    "BIOMEDCLIP_MEAN",
    "BIOMEDCLIP_STD",
    "IMAGE_SIZE",
    "load_image",
    "preprocess_image",
    "prepare_image_batch"
]