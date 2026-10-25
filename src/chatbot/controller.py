from typing import Any, Dict, Optional
from .context_builder import ContextBuilder
from .narrative_refiner import NarrativeRefiner
from .ollama_client import OllamaClient
from .output_guardrail import OutputGuardrail
from .prompt_builder import PromptBuilder
from .query_guardrail import QueryGuardrail
from .response_formatter import ResponseFormatter
from .schemas import ChatQuery, ChatResponse


class ChatbotController:
    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        query_guardrail: Optional[QueryGuardrail] = None,
        output_guardrail: Optional[OutputGuardrail] = None,
        context_builder: Optional[ContextBuilder] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        response_formatter: Optional[ResponseFormatter] = None,
        narrative_refiner: Optional[NarrativeRefiner] = None,
    ):
        self.ollama_client = ollama_client or OllamaClient(
            model="phi3:latest"
        )

        self.query_guardrail = (
            query_guardrail or QueryGuardrail()
        )

        self.output_guardrail = (
            output_guardrail or OutputGuardrail()
        )

        self.context_builder = (
            context_builder or ContextBuilder()
        )

        self.prompt_builder = (
            prompt_builder or PromptBuilder()
        )

        self.response_formatter = (
            response_formatter or ResponseFormatter()
        )

        self.narrative_refiner = (
            narrative_refiner
            or NarrativeRefiner(
                ollama_client=self.ollama_client
            )
        )

    def _prepare_module2_result(
        self,
        module2_result: Optional[Dict[str, Any]],
    ):
        if not isinstance(module2_result, dict):
            return module2_result, {
                "available": False,
                "success": False,
                "model": None,
                "error": None,
            }

        raw_narrative = (
            module2_result.get("narrative")
            or module2_result.get("clinical_narrative")
            or module2_result.get("report")
            or module2_result.get("caption")
        )

        if not raw_narrative:
            return module2_result, {
                "available": False,
                "success": False,
                "model": None,
                "error": "No Module 2 narrative available for refinement.",
            }

        refinement_result = self.narrative_refiner.refine(
            str(raw_narrative)
        )

        prepared_result = dict(module2_result)

        prepared_result["raw_narrative"] = str(raw_narrative)

        prepared_result["narrative"] = (
            refinement_result.refined_narrative
        )

        prepared_result["refinement"] = {
            "success": refinement_result.success,
            "model": refinement_result.model,
            "error": refinement_result.error,
        }

        refinement_metadata = {
            "available": True,
            "success": refinement_result.success,
            "model": refinement_result.model,
            "error": refinement_result.error,
        }

        return prepared_result, refinement_metadata

    def ask(
        self,
        question: str,
        view_result: Optional[Dict[str, Any]] = None,
        module1_result: Optional[Dict[str, Any]] = None,
        module2_result: Optional[Dict[str, Any]] = None,
    ) -> ChatResponse:

        query = ChatQuery(
            question=question
        )

        normalized_question = query.normalized()

        if not normalized_question:
            return self.response_formatter.format_blocked_query(
                reason="empty_question",
                category="empty",
            )

        query_guardrail = (
            self.query_guardrail.evaluate(query)
        )

        if not query_guardrail.allowed:
            return self.response_formatter.format_blocked_query(
                reason=query_guardrail.reason,
                category=query_guardrail.category,
            )

        prepared_module2_result, refinement_metadata = (
            self._prepare_module2_result(
                module2_result
            )
        )

        complete_evidence = self.context_builder.build(
            view_result=view_result,
            module1_result=module1_result,
            module2_result=prepared_module2_result,
        )

        evidence = (
            self.context_builder.select_relevant_evidence(
                question=normalized_question,
                evidence=complete_evidence,
            )
        )

        source_modules = (
            self.context_builder.get_available_sources(
                evidence
            )
        )

        if evidence.is_empty():
            return self.response_formatter.format_blocked_query(
                reason=(
                    "No module evidence is available "
                    "for this question."
                ),
                category="no_module_evidence",
            )

        system_prompt = (
            self.prompt_builder.build_system_prompt(
                evidence
            )
        )

        user_prompt = (
            self.prompt_builder.build_user_prompt(
                normalized_question
            )
        )

        ollama_response = self.ollama_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )

        if not ollama_response.success:
            return self.response_formatter.format_ollama_error(
                error=(
                    ollama_response.error
                    or "Unknown Ollama error."
                ),
                model=ollama_response.model,
                source_modules=source_modules,
            )

        output_guardrail = (
            self.output_guardrail.validate(
                ollama_response.text
            )
        )

        if not output_guardrail.allowed:
            fallback_text = (
                self.output_guardrail.fallback_response(
                    output_guardrail
                )
            )

            return (
                self.response_formatter
                .format_guardrail_failure(
                    text=fallback_text,
                    reason=output_guardrail.reason,
                    category=output_guardrail.category,
                    model=ollama_response.model,
                    source_modules=source_modules,
                )
            )

        return ChatResponse(
            text=ollama_response.text,
            allowed=True,
            category="cxr_evidence_response",
            source_modules=source_modules,
            guardrail_passed=True,
            output_guardrail_passed=True,
            model=ollama_response.model,
            error=None,
            metadata={
                "query_guardrail": (
                    query_guardrail.to_dict()
                ),
                "output_guardrail": (
                    output_guardrail.to_dict()
                ),
                "evidence_intent": (
                    self.context_builder.detect_intent(
                        normalized_question
                    )
                ),
                "narrative_refinement": refinement_metadata,
            },
        )

    def check_ollama(self) -> bool:
        return self.ollama_client.is_available()

    def check_model(self) -> bool:
        return self.ollama_client.model_available()

    def get_status(self) -> Dict[str, Any]:
        ollama_available = self.check_ollama()

        if not ollama_available:
            return {
                "ollama_available": False,
                "model": self.ollama_client.model,
                "model_available": False,
            }

        return {
            "ollama_available": True,
            "model": self.ollama_client.model,
            "model_available": self.check_model(),
        }