
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torchvision.models as tv_models


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "config"
MODEL_DIR = PROJECT_ROOT / "models" / "module1_disease"

MODULE_CONFIG_PATH = CONFIG_DIR / "module1_config.json"
CLASS_ORDER_PATH = CONFIG_DIR / "class_order.json"


def load_json(path: Path):
    """
    Load a JSON file.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_class_order() -> list[str]:
    """
   
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
            f"Expected 14 disease classes, "
            f"but found {len(class_order)}."
        )

    return class_order


def get_device() -> torch.device:
    """
    Select CUDA if available; otherwise use CPU.
    """
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def extract_state_dict(checkpoint):
    """
    Extract model weights from a checkpoint.

    Supported checkpoint formats:
    1. Raw state dictionary.
    2. Dictionary containing 'model_state_dict'.

    Also removes a possible 'module.' prefix.
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint must be a dictionary."
        )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        cleaned_key = key

        if cleaned_key.startswith("module."):
            cleaned_key = cleaned_key[len("module."):]

        cleaned_state_dict[cleaned_key] = value

    return cleaned_state_dict


def build_ap_model(num_classes: int) -> nn.Module:
    """
    Build the AP DenseNet121 architecture.
    """
    model = tv_models.densenet121(weights=None)

    input_features = model.classifier.in_features

    model.classifier = nn.Linear(
        input_features,
        num_classes,
    )

    return model


def build_lateral_model(num_classes: int) -> nn.Module:
    """
    Build the Lateral ResNet50 architecture.

    The lateral checkpoint uses keys such as:

    features.0.weight
    features.1.weight
    features.4.0.conv1.weight
    features.7.2.bn3.running_var
    classifier.weight
    classifier.bias
    """
    base_model = tv_models.resnet50(weights=None)

    feature_layers = nn.Sequential(
        base_model.conv1,
        base_model.bn1,
        base_model.relu,
        base_model.maxpool,
        base_model.layer1,
        base_model.layer2,
        base_model.layer3,
        base_model.layer4,
    )

    class LateralResNet50(nn.Module):
        def __init__(self):
            super().__init__()

            self.features = feature_layers
            self.avgpool = base_model.avgpool

            self.classifier = nn.Linear(
                base_model.fc.in_features,
                num_classes,
            )

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)

            return x

    return LateralResNet50()


def resolve_checkpoint_path(
    relative_checkpoint_path: str,
) -> Path:
    """
    Resolve a checkpoint path relative to the disease-model folder.

    Example:
        checkpoints/ap_densenet121_best.pth

    becomes:
        models/module1_disease/checkpoints/
        ap_densenet121_best.pth
    """
    checkpoint_path = MODEL_DIR / relative_checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint file not found.\n"
            f"Expected path: {checkpoint_path}"
        )

    return checkpoint_path


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    """
    Load checkpoint weights into a model.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    state_dict = extract_state_dict(checkpoint)

    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if missing_keys:
        raise RuntimeError(
            "Missing keys while loading checkpoint:\n"
            + "\n".join(missing_keys)
        )

    if unexpected_keys:
        raise RuntimeError(
            "Unexpected keys while loading checkpoint:\n"
            + "\n".join(unexpected_keys)
        )

    model.to(device)
    model.eval()

    return model


def load_disease_models() -> Tuple[
    Dict[str, nn.Module],
    torch.device,
]:
    """
    Load both AP and Lateral disease models.

    Returns:
        models:
            {
                "AP": AP DenseNet121,
                "Lateral": Lateral ResNet50
            }

        device:
            CPU or CUDA device.
    """
    config = load_json(MODULE_CONFIG_PATH)

    if not isinstance(config, dict):
        raise ValueError(
            "module1_config.json must contain a JSON object."
        )

    class_order = load_class_order()
    num_classes = len(class_order)

    device = get_device()

    # The actual configuration uses the key "models".
    models_config = config.get("models")

    if not isinstance(models_config, dict):
        raise KeyError(
            "module1_config.json must contain a 'models' object."
        )

    ap_config = models_config.get("AP")
    lateral_config = models_config.get("Lateral")

    if not isinstance(ap_config, dict):
        raise KeyError(
            "AP model configuration is missing."
        )

    if not isinstance(lateral_config, dict):
        raise KeyError(
            "Lateral model configuration is missing."
        )

    ap_checkpoint_relative = ap_config.get("checkpoint")
    lateral_checkpoint_relative = lateral_config.get("checkpoint")

    if not isinstance(ap_checkpoint_relative, str):
        raise KeyError(
            "AP checkpoint path is missing or invalid."
        )

    if not isinstance(lateral_checkpoint_relative, str):
        raise KeyError(
            "Lateral checkpoint path is missing or invalid."
        )

    ap_checkpoint_path = resolve_checkpoint_path(
        ap_checkpoint_relative
    )

    lateral_checkpoint_path = resolve_checkpoint_path(
        lateral_checkpoint_relative
    )

    print(f"Loading AP checkpoint: {ap_checkpoint_path}")

    ap_model = build_ap_model(num_classes)

    ap_model = load_checkpoint(
        model=ap_model,
        checkpoint_path=ap_checkpoint_path,
        device=device,
    )

    print(
        f"Loading Lateral checkpoint: "
        f"{lateral_checkpoint_path}"
    )

    lateral_model = build_lateral_model(num_classes)

    lateral_model = load_checkpoint(
        model=lateral_model,
        checkpoint_path=lateral_checkpoint_path,
        device=device,
    )

    models = {
        "AP": ap_model,
        "Lateral": lateral_model,
    }

    return models, device


def main() -> None:
    """
    Test loading both disease models.
    """
    print("=" * 70)
    print("CXR CLINICAL AI - DISEASE MODEL LOADER TEST")
    print("=" * 70)

    class_order = load_class_order()

    print(f"Number of disease classes: {len(class_order)}")

    for index, disease in enumerate(class_order):
        print(f"{index:02d} - {disease}")

    models, device = load_disease_models()

    print(f"\nDevice: {device}")
    print(f"Available models: {list(models.keys())}")

    for model_name, model in models.items():
        parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        print(
            f"{model_name} model loaded successfully | "
            f"Parameters: {parameter_count:,}"
        )

    print("=" * 70)
    print("Disease model loader test completed successfully.")


if __name__ == "__main__":
    main()