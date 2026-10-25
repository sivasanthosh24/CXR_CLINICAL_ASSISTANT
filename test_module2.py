from pathlib import Path
import json
import time

from src.module2.pipeline import Module2Pipeline


PROJECT_ROOT = Path(__file__).resolve().parent

SAMPLE_DIR = PROJECT_ROOT / "sample_images"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "module2"
    / "test_run"
)


def main():
    print("=" * 80)
    print("CXR CLINICAL AI - MODULE 2 V2 INTEGRATION TEST")
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

    print("\nLoading Module 2 V2...")

    start_load = time.perf_counter()

    pipeline = Module2Pipeline()

    load_time = (
        time.perf_counter()
        - start_load
    )

    print("Module 2 V2 loaded successfully.")

    print(
        f"Model loading time: "
        f"{load_time:.2f} seconds"
    )

    print("\nModel Information:")

    model_info = pipeline.get_model_info()

    print(
        f"Module: "
        f"{model_info.get('module')}"
    )

    print(
        f"Checkpoint: "
        f"{model_info.get('checkpoint')}"
    )

    print(
        f"Device: "
        f"{model_info.get('device')}"
    )

    print(
        f"Visual total parameters: "
        f"{model_info.get('visual_total_parameters')}"
    )

    print(
        f"Visual trainable parameters: "
        f"{model_info.get('visual_trainable_parameters')}"
    )

    print(
        f"Decoder total parameters: "
        f"{model_info.get('decoder_total_parameters')}"
    )

    print(
        f"Decoder trainable parameters: "
        f"{model_info.get('decoder_trainable_parameters')}"
    )

    print(
        f"Total trainable parameters: "
        f"{model_info.get('total_trainable_parameters')}"
    )

    print("\nGeneration Configuration:")

    generation_config = model_info.get(
        "generation",
        {},
    )

    for key, value in generation_config.items():
        print(
            f"  {key}: {value}"
        )

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

        print("\nGenerating Module 2 clinical narrative...")

        report = pipeline.generate_report(
            image_path
        )

        elapsed_time = (
            time.perf_counter()
            - start_time
        )

        if not isinstance(
            report,
            str,
        ):
            raise TypeError(
                f"Module 2 returned an unexpected "
                f"result type for {image_path.name}: "
                f"{type(report)}"
            )

        report = report.strip()

        if not report:
            raise ValueError(
                f"Module 2 generated an empty "
                f"report for {image_path.name}."
            )

        print("\nGenerated Clinical Narrative:")
        print(report)

        print(
            f"\nGeneration time: "
            f"{elapsed_time:.2f} seconds"
        )

        result = {
            "image": image_path.name,
            "report": report,
            "generation_time_seconds": elapsed_time,
        }

        results.append(result)

        image_output_path = (
            OUTPUT_DIR
            / f"{image_path.stem}_module2_result.json"
        )

        with image_output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "model_info": model_info,
                    **result,
                },
                file,
                indent=4,
                ensure_ascii=False,
            )

        print(
            f"Result saved to: "
            f"{image_output_path}"
        )

    summary_path = (
        OUTPUT_DIR
        / "module2_test_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "model_info": model_info,
                "images_tested": len(results),
                "results": results,
            },
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 80)
    print("MODULE 2 V2 INTEGRATION TEST COMPLETE")
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