"""Lightweight chat-intent matching for the independent Module 3 route."""


def is_similar_retrieval_request(question: str) -> bool:
    """Recognize common phrasings for similar and previous cases."""
    query = " ".join(question.lower().replace("–", "-").split())
    return any(
        phrase in query
        for phrase in (
            "similar image", "similar x-ray", "similar xray", "similar case",
            "retrieve similar", "retrive similar", "previous case", "previous cases",
        )
    )
