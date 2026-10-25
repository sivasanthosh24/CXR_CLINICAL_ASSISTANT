from pathlib import Path
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
class ViewClassifier:
    def __init__(self, model_directory=None):
        if model_directory is None:
            model_directory = (
                Path(__file__).resolve().parents[2]
                / "models"
                / "module1_view"
            )
        self.model_directory = Path(
            model_directory
        )
        self.checkpoint_path = (
            self.model_directory
            / "view_classifier_complete.pth"
        )
        self.config_path = (
            self.model_directory
            / "model_config.json"
        )
        self.mapping_path = (
            self.model_directory
            / "class_mapping.json"
        )
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found:\n"
                f"{self.checkpoint_path}"
            )
        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"Model configuration not found:\n"
                f"{self.config_path}"
            )
        if not self.mapping_path.is_file():
            raise FileNotFoundError(
                f"Class mapping not found:\n"
                f"{self.mapping_path}"
            )
        with open(
            self.config_path,
            "r",
            encoding="utf-8"
        ) as f:
            self.config = json.load(f)
        with open(
            self.mapping_path,
            "r",
            encoding="utf-8"
        ) as f:
            self.class_to_index = json.load(f)
        self.index_to_class = {
            int(value): key
            for key, value
            in self.class_to_index.items()
        }
        self.image_size = int(
            self.config["image_size"]
        )
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.model = self._build_model()
        self.transform = transforms.Compose([
            transforms.Resize(
                (
                    self.image_size,
                    self.image_size
                )
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config[
                    "normalization"
                ]["mean"],
                std=self.config[
                    "normalization"
                ]["std"]
            )
        ])
    def _build_model(self):
        model = models.densenet121(
            weights=None
        )
        input_features = (
            model.classifier.in_features
        )
        model.classifier = nn.Linear(
            input_features,
            2
        )
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu"
        )
        if "model_state_dict" not in checkpoint:
            raise RuntimeError(
                "model_state_dict not found "
                "in checkpoint."
            )
        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )
        model = model.to(
            self.device
        )
        model.eval()
        return model
    def predict(self, image_path):
        image_path = Path(
            image_path
        )
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found:\n"
                f"{image_path}"
            )
        image = Image.open(
            image_path
        ).convert("RGB")
        tensor = self.transform(
            image
        ).unsqueeze(0)
        tensor = tensor.to(
            self.device
        )
        with torch.no_grad():
            logits = self.model(
                tensor
            )
            probabilities = torch.softmax(
                logits,
                dim=1
            )
        predicted_index = int(
            probabilities.argmax(
                dim=1
            ).item()
        )
        predicted_view = (
            self.index_to_class[
                predicted_index
            ]
        )
        confidence = float(
            probabilities[
                0,
                predicted_index
            ].item()
        )
        return {
            "predicted_view":
                predicted_view,
            "confidence":
                confidence,
            "probabilities": {
                self.index_to_class[i]:
                    float(
                        probabilities[
                            0,
                            i
                        ].item()
                    )
                for i in range(2)
            },
            "device":
                str(self.device)
        }
if __name__ == "__main__":
    print(
        "ViewClassifier module loaded successfully."
    )
