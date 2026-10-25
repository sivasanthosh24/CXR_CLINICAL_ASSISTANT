from pathlib import Path
import json
import time

from src.module1.pipeline import run_module1_pipeline
from src.view_classifier.predictor import ViewClassifier


PROJECT_ROOT = Path(__file__).resolve().parent

SAMPLE_DIR = PROJECT_ROOT / "sample_images"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "module1"
    / "test_run"
)


def main():
    print("=" * 80)
    print("CXR CLINICAL AI - MODULE 1 INTEGRATION TEST")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = sorted(
        [
            path
            for path in SAMPLE_DIR.iterdir()
            if path.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
            }
        ]
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No sample images found in: {SAMPLE_DIR}"
        )

    print(f"\nFound {len(image_paths)} sample images.")

    print("\nLoading View Classifier...")
    view_classifier = ViewClassifier()
    print("View Classifier loaded successfully.")

    results = []

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        print("\n" + "-" * 80)
        print(
            f"IMAGE {index}/{len(image_paths)}: "
            f"{image_path.name}"
        )
        print("-" * 80)

        start_time = time.perf_counter()

        print("\nStep 1: View Classification")

        view_result = view_classifier.predict(
            image_path
        )

        print(
            f"Predicted view: "
            f"{view_result.get('predicted_view')}"
        )

        print(
            f"Confidence: "
            f"{view_result.get('confidence')}"
        )

        predicted_view = view_result.get(
            "predicted_view"
        )

        if predicted_view not in {
            "AP",
            "Lateral",
        }:
            raise ValueError(
                f"Invalid view returned for "
                f"{image_path.name}: "
                f"{predicted_view}"
            )

        print("\nStep 2: Module 1")

        module1_output_dir = (
            OUTPUT_DIR
            / image_path.stem
        )

        module1_result = run_module1_pipeline(
            image_source=image_path,
            view=predicted_view,
            output_dir=module1_output_dir,
        )

        elapsed_time = (
            time.perf_counter()
            - start_time
        )

        prediction = module1_result.get(
            "prediction",
            {},
        )

        gradcam = module1_result.get(
            "gradcam",
            {},
        )

        localization = module1_result.get(
            "weak_localization",
            {},
        )

        positive_findings = prediction.get(
            "positive_findings",
            [],
        )

        ranked_findings = prediction.get(
            "ranked_findings",
            [],
        )

        print("\nModule 1 Results")

        print(
            f"View: "
            f"{module1_result.get('view')}"
        )

        print(
            f"Device: "
            f"{module1_result.get('device')}"
        )

        print(
            f"Selected Grad-CAM disease: "
            f"{module1_result.get('selected_gradcam_disease')}"
        )

        print("\nPositive findings:")

        if positive_findings:
            for disease in positive_findings:
                print(f"  - {disease}")
        else:
            print("  - None")

        print("\nTop predictions:")

        for finding in ranked_findings[:5]:
            disease = finding.get(
                "disease",
                "Unknown",
            )

            probability = finding.get(
                "probability",
                finding.get(
                    "score",
                    None,
                ),
            )

            print(
                f"  - {disease}: "
                f"{probability}"
            )

        print("\nGrad-CAM:")

        print(
            f"  Predicted class: "
            f"{gradcam.get('predicted_class_name')}"
        )

        print(
            f"  Explained class: "
            f"{gradcam.get('explained_class_name')}"
        )

        print(
            f"  Target layer: "
            f"{gradcam.get('target_layer')}"
        )

        print("\nWeak Localization:")

        print(
            f"  Available: "
            f"{localization.get('localization_available')}"
        )

        print(
            f"  Bounding box: "
            f"{localization.get('bounding_box')}"
        )

        print(
            f"  Center: "
            f"{localization.get('center')}"
        )

        print(
            f"  Area ratio: "
            f"{localization.get('component_area_ratio')}"
        )

        output_paths = module1_result.get(
            "output_paths",
            {},
        )

        print("\nOutput files:")

        for key, value in output_paths.items():
            print(
                f"  {key}: {value}"
            )

        print(
            f"\nInference time: "
            f"{elapsed_time:.2f} seconds"
        )

        results.append(
            {
                "image": image_path.name,
                "view_result": view_result,
                "module1_result": module1_result,
                "inference_time_seconds": elapsed_time,
            }
        )

    summary_path = (
        OUTPUT_DIR
        / "module1_test_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 80)
    print("MODULE 1 INTEGRATION TEST COMPLETE")
    print("=" * 80)

    print(
        f"\nTested images: "
        f"{len(results)}"
    )

    print(
        f"Summary saved to: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()