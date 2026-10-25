from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(
        ROOT
        / "src"
        / "view_classifier"
    )
)

from predictor import ViewClassifier


print("=" * 70)
print("CXR CLINICAL AI — LOCAL VIEW CLASSIFIER TEST")
print("=" * 70)


classifier = ViewClassifier()

print(
    "\nModel loaded successfully."
)

print(
    "Device:",
    classifier.device
)

sample_root = (
    ROOT
    / "sample_images"
)

images = sorted(
    sample_root.glob("*.jpg")
)

if not images:

    raise RuntimeError(
        "No sample JPG images found."
    )

print(
    "\nImages:",
    len(images)
)

print("\n" + "-" * 70)

for image_path in images:

    result = classifier.predict(
        image_path
    )

    print(
        f"\nImage: {image_path.name}"
    )

    print(
        "Predicted view:",
        result["predicted_view"]
    )

    print(
        "Confidence:",
        f"{result['confidence']:.4f}"
    )

    print(
        "AP probability:",
        f"{result['probabilities']['AP']:.4f}"
    )

    print(
        "Lateral probability:",
        f"{result['probabilities']['Lateral']:.4f}"
    )

print("\n" + "=" * 70)
print("LOCAL VIEW CLASSIFIER TEST COMPLETE")
print("=" * 70)
