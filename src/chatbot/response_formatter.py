from typing import Any, Dict, Optional

from .schemas import ChatResponse, GuardrailResult, OllamaResponse


class ResponseFormatter:

    def format_success(
        self,
        ollama_response: OllamaResponse,
        output_guardrail: GuardrailResult,
        source_modules: list[str],
    ) -> ChatResponse:

        return ChatResponse(
            text=ollama_response.text.strip(),
            allowed=True,
            category="cxr_evidence_response",
            source_modules=source_modules,
            guardrail_passed=True,
            output_guardrail_passed=output_guardrail.allowed,
            model=ollama_response.model,
            error=None,
            metadata={
                "output_guardrail": (
                    output_guardrail.to_dict()
                ),
            },
        )

    def format_blocked_query(
        self,
        reason: str,
        category: str,
        source_modules: Optional[list[str]] = None,
    ) -> ChatResponse:

        return ChatResponse(
            text=reason,
            allowed=False,
            category=category,
            source_modules=source_modules or [],
            guardrail_passed=False,
            output_guardrail_passed=False,
            model=None,
            error=None,
            metadata={},
        )

    def format_ollama_error(
        self,
        error: str,
        model: Optional[str],
        source_modules: list[str],
    ) -> ChatResponse:

        error_message = (
            error
            or "The local language model could not "
            "generate a response."
        )

        return ChatResponse(
            text=(
                "The chatbot could not generate a "
                "response because the local Ollama "
                "service returned an error: "
                f"{error_message}"
            ),
            allowed=False,
            category="ollama_error",
            source_modules=source_modules,
            guardrail_passed=True,
            output_guardrail_passed=False,
            model=model,
            error=error_message,
            metadata={},
        )

    def format_guardrail_failure(
        self,
        text: str,
        reason: str,
        category: str,
        model: Optional[str],
        source_modules: list[str],
    ) -> ChatResponse:

        return ChatResponse(
            text=text,
            allowed=False,
            category=category,
            source_modules=source_modules,
            guardrail_passed=True,
            output_guardrail_passed=False,
            model=model,
            error=reason,
            metadata={},
        )

    @staticmethod
    def to_display_text(
        response: ChatResponse,
    ) -> str:

        return response.text.strip()

    @staticmethod
    def to_dict(
        response: ChatResponse,
    ) -> Dict[str, Any]:

        return response.to_dict()