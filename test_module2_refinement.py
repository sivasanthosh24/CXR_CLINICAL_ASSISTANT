from pathlib import Path
import json
import time

from src.module2.pipeline import Module2Pipeline
from src.chatbot.narrative_refiner import NarrativeRefiner
from src.chatbot.ollama_client import OllamaClient


PROJECT_ROOT = Path(__file__).resolve().parent

SAMPLE_DIR = PROJECT_ROOT / "sample_images"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "module2"
    / "refinement_test"
)


def main():
    print("=" * 80)
    print("MODULE 2 V2 -> NARRATIVE REFINER INTEGRATION TEST")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = sorted(
        [
            path
            for path in SAMPLE_DIR.iterdir()
            if path.suffix.lower()
            in {
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

    print(
        f"\nFound {len(image_paths)} sample images."
    )

    print("\nLoading Module 2 V2...")

    module2_load_start = time.perf_counter()

    module2_pipeline = Module2Pipeline()

    module2_load_time = (
        time.perf_counter()
        - module2_load_start
    )

    print(
        "Module 2 V2 loaded successfully."
    )

    print(
        f"Module 2 loading time: "
        f"{module2_load_time:.2f} seconds"
    )

    model_info = (
        module2_pipeline.get_model_info()
    )

    print("\nModule 2 Information:")

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

    print("\nInitializing Ollama Narrative Refiner...")

    ollama_client = OllamaClient()

    refiner = NarrativeRefiner(
        ollama_client=ollama_client
    )

    ollama_status = {
        "ollama_available": (
            ollama_client.is_available()
        ),
        "model": ollama_client.model,
        "model_available": (
            ollama_client.model_available()
        ),
    }

    print(
        f"Ollama available: "
        f"{ollama_status['ollama_available']}"
    )

    print(
        f"Ollama model: "
        f"{ollama_status['model']}"
    )

    print(
        f"Model available: "
        f"{ollama_status['model_available']}"
    )

    if not ollama_status["ollama_available"]:
        raise RuntimeError(
            "Ollama is not available. "
            "Make sure the local Ollama service is running."
        )

    if not ollama_status["model_available"]:
        raise RuntimeError(
            f"Ollama model '{ollama_client.model}' "
            f"is not available."
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

        print(
            "\nSTEP 1: Generating Module 2 V2 "
            "clinical narrative..."
        )

        module2_start = time.perf_counter()

        raw_narrative = (
            module2_pipeline.generate_report(
                image_path
            )
        )

        module2_time = (
            time.perf_counter()
            - module2_start
        )

        if not isinstance(
            raw_narrative,
            str,
        ):
            raise TypeError(
                "Module 2 returned an unexpected "
                f"result type: "
                f"{type(raw_narrative)}"
            )

        raw_narrative = (
            raw_narrative.strip()
        )

        if not raw_narrative:
            raise ValueError(
                f"Module 2 generated an empty "
                f"narrative for {image_path.name}."
            )

        print(
            "\nRAW MODULE 2 NARRATIVE:"
        )

        print(
            raw_narrative
        )

        print(
            f"\nModule 2 generation time: "
            f"{module2_time:.2f} seconds"
        )

        print(
            "\nSTEP 2: Refining Module 2 "
            "narrative with Ollama..."
        )

        refinement_start = (
            time.perf_counter()
        )

        refinement_result = (
            refiner.refine(
                raw_narrative
            )
        )

        refinement_time = (
            time.perf_counter()
            - refinement_start
        )

        refined_narrative = (
            refinement_result.refined_narrative
        )

        if not isinstance(
            refined_narrative,
            str,
        ):
            raise TypeError(
                "Narrative Refiner returned an "
                f"unexpected result type: "
                f"{type(refined_narrative)}"
            )

        refined_narrative = (
            refined_narrative.strip()
        )

        if not refined_narrative:
            raise ValueError(
                f"Narrative Refiner produced an "
                f"empty narrative for "
                f"{image_path.name}."
            )

        print(
            "\nREFINED NARRATIVE:"
        )

        print(
            refined_narrative
        )

        print(
            f"\nRefinement time: "
            f"{refinement_time:.2f} seconds"
        )

        print(
            "\nREFINEMENT STATUS:"
        )

        print(
            f"Success: "
            f"{refinement_result.success}"
        )

        print(
            f"Model: "
            f"{refinement_result.model}"
        )

        print(
            f"Error: "
            f"{refinement_result.error}"
        )

        result = {
            "image": image_path.name,
            "raw_narrative": raw_narrative,
            "refined_narrative": refined_narrative,
            "refinement_success": (
                refinement_result.success
            ),
            "refinement_model": (
                refinement_result.model
            ),
            "refinement_error": (
                refinement_result.error
            ),
            "module2_generation_time_seconds": (
                module2_time
            ),
            "narrative_refinement_time_seconds": (
                refinement_time
            ),
            "total_pipeline_time_seconds": (
                module2_time
                + refinement_time
            ),
        }

        results.append(result)

        image_output_path = (
            OUTPUT_DIR
            / f"{image_path.stem}_refinement_result.json"
        )

        with image_output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "model_info": model_info,
                    "ollama_status": ollama_status,
                    **result,
                },
                file,
                indent=4,
                ensure_ascii=False,
            )

        print(
            f"\nResult saved to: "
            f"{image_output_path}"
        )

    successful_refinements = sum(
        1
        for result in results
        if result["refinement_success"]
    )

    failed_refinements = (
        len(results)
        - successful_refinements
    )

    total_module2_time = sum(
        result[
            "module2_generation_time_seconds"
        ]
        for result in results
    )

    total_refinement_time = sum(
        result[
            "narrative_refinement_time_seconds"
        ]
        for result in results
    )

    total_pipeline_time = sum(
        result[
            "total_pipeline_time_seconds"
        ]
        for result in results
    )

    summary = {
        "test": (
            "Module 2 V2 -> "
            "Narrative Refiner"
        ),
        "images_tested": len(results),
        "successful_refinements": (
            successful_refinements
        ),
        "failed_refinements": (
            failed_refinements
        ),
        "module2_load_time_seconds": (
            module2_load_time
        ),
        "total_module2_generation_time_seconds": (
            total_module2_time
        ),
        "total_refinement_time_seconds": (
            total_refinement_time
        ),
        "total_pipeline_time_seconds": (
            total_pipeline_time
        ),
        "ollama_status": ollama_status,
        "model_info": model_info,
        "results": results,
    }

    summary_path = (
        OUTPUT_DIR
        / "module2_refinement_test_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 80)
    print(
        "MODULE 2 V2 -> NARRATIVE REFINER "
        "INTEGRATION TEST COMPLETE"
    )
    print("=" * 80)

    print(
        f"\nImages tested: "
        f"{len(results)}"
    )

    print(
        f"Successful refinements: "
        f"{successful_refinements}"
    )

    print(
        f"Failed refinements: "
        f"{failed_refinements}"
    )

    print(
        f"Total Module 2 generation time: "
        f"{total_module2_time:.2f} seconds"
    )

    print(
        f"Total refinement time: "
        f"{total_refinement_time:.2f} seconds"
    )

    print(
        f"Total pipeline time: "
        f"{total_pipeline_time:.2f} seconds"
    )

    print(
        f"\nSummary saved to: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()