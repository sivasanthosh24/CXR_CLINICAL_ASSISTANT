from src.chatbot.narrative_refiner import NarrativeRefiner
from src.chatbot.ollama_client import OllamaClient


def main():
    print("=" * 80)
    print("NARRATIVE REFINER TEST")
    print("=" * 80)

    raw_narrative = (
        "ings: As compared to the previous radiograph, there is no significant "
        "interval change. No PICC line in standard placements. No"
    )

    print("\nRAW MODULE 2 NARRATIVE:")
    print(raw_narrative)

    print("\nREFINING WITH OLLAMA...")

    ollama_client = OllamaClient()
    refiner = NarrativeRefiner(ollama_client)

    result = refiner.refine(raw_narrative)

    print("\nREFINED NARRATIVE:")
    print(result.refined_narrative)

    print("\nSTATUS:")
    print("Success:", result.success)
    print("Model:", result.model)
    print("Error:", result.error)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()