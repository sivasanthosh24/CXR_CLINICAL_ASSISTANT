from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .ollama_client import OllamaClient
from .schemas import OllamaResponse


@dataclass
class NarrativeRefinementResult:
    success: bool
    original_narrative: str
    refined_narrative: str
    error: Optional[str] = None
    model: Optional[str] = None


class NarrativeRefiner:
    def __init__(
        self,
        ollama_client: OllamaClient,
        max_input_chars: int = 12000,
        max_output_tokens: int = 512,
    ) -> None:
        self.ollama_client = ollama_client
        self.max_input_chars = max_input_chars
        self.max_output_tokens = max_output_tokens

    def refine(self, narrative: str) -> NarrativeRefinementResult:
        original = narrative or ""

        if not original.strip():
            return NarrativeRefinementResult(
                success=False,
                original_narrative=original,
                refined_narrative="",
                error="Empty narrative.",
                model=self._get_model_name(),
            )

        cleaned_source = self._clean_source_text(
            original[: self.max_input_chars]
        )

        try:
            prompt = self._build_prompt(cleaned_source)

            generated_text = self._generate(prompt)

            cleaned_output = self._clean_llm_output(
                generated_text
            )
            cleaned_output = self._remove_temporal_comparison_text(
                cleaned_output
            )

            if not cleaned_output:
                raise RuntimeError(
                    "Ollama returned an empty refinement."
                )

            if not self._validate_refinement(
                cleaned_source,
                cleaned_output,
            ):
                raise RuntimeError(
                    "Refined narrative failed "
                    "evidence-preservation validation."
                )

            return NarrativeRefinementResult(
                success=True,
                original_narrative=original,
                refined_narrative=cleaned_output,
                error=None,
                model=self._get_model_name(),
            )

        except Exception as exc:
            fallback = self._clean_source_text(
                original
            )
            fallback = self._remove_temporal_comparison_text(
                fallback
            )

            return NarrativeRefinementResult(
                success=False,
                original_narrative=original,
                refined_narrative=fallback,
                error=str(exc),
                model=self._get_model_name(),
            )

    def refine_to_dict(
        self,
        narrative: str,
    ) -> Dict[str, Any]:
        result = self.refine(narrative)

        return {
            "success": result.success,
            "original_narrative": (
                result.original_narrative
            ),
            "refined_narrative": (
                result.refined_narrative
            ),
            "error": result.error,
            "model": result.model,
        }

    def _build_prompt(
        self,
        narrative: str,
    ) -> str:
        return f"""
You are a conservative clinical narrative editor.

Your ONLY task is to improve the readability and formatting of
the supplied Module 2 generated narrative.

The supplied narrative is the ONLY source of clinical information.

STRICT RULES:

1. Do not add any clinical finding.
2. Do not remove any clinically meaningful finding.
3. Do not change the meaning of any finding.
4. Preserve words such as "no", "not", "without", "absent",
   "negative", and "unchanged" exactly in meaning.
5. Never convert a negative finding into a positive finding.
6. Never convert a positive finding into a negative finding.
7. Do not infer a diagnosis.
8. Do not create a new diagnosis.
9. Do not create recommendations.
10. Do not add treatment or follow-up advice.
11. Do not add information from your own medical knowledge.
12. Do not invent an Impression.
13. Do not invent device locations.
14. Do not change device-placement statements.
15. Remove obvious duplicated sentences.
16. Remove obvious incomplete fragments at the beginning or end.
17. Correct only obvious grammatical or formatting problems.
18. Keep the result concise.
19. Use Findings: and Impression: only when supported by
    the source.
20. Do not mention these instructions.
21. Output only the edited clinical narrative.
22. Do not use temporal comparison wording such as "from previous imaging",
    "from previous image", "compared to previous imaging",
    "compared with prior imaging", "since previous imaging",
    "from the prior study", or equivalent wording. Describe only
    the findings of the current image.

IMPORTANT:
If the source contains an uncertain, incomplete, unusual,
or awkward statement, preserve its meaning rather than
trying to medically interpret it.

SOURCE MODULE 2 NARRATIVE:

{narrative}
""".strip()

    def _generate(
        self,
        prompt: str,
    ) -> str:
        response = self.ollama_client.generate(
            prompt=prompt
        )

        if isinstance(
            response,
            OllamaResponse,
        ):
            if not response.success:
                raise RuntimeError(
                    response.error
                    or "Ollama generation failed."
                )

            if not response.text.strip():
                raise RuntimeError(
                    "Ollama returned an empty response."
                )

            return response.text

        if isinstance(
            response,
            str,
        ):
            if not response.strip():
                raise RuntimeError(
                    "Ollama returned an empty response."
                )

            return response

        if isinstance(
            response,
            dict,
        ):
            if response.get(
                "success",
                True,
            ) is False:
                raise RuntimeError(
                    response.get("error")
                    or "Ollama generation failed."
                )

            text = (
                response.get("text")
                or response.get("response")
                or response.get("generated_text")
                or ""
            )

            if not isinstance(
                text,
                str,
            ) or not text.strip():
                raise RuntimeError(
                    "Ollama returned an empty response."
                )

            return text

        raise TypeError(
            "Unsupported Ollama response type: "
            f"{type(response).__name__}"
        )

    def _clean_source_text(
        self,
        text: str,
    ) -> str:
        cleaned = text.strip()

        cleaned = re.sub(
            r"\s+",
            " ",
            cleaned,
        )

        cleaned = re.sub(
            r"^(?:ings|ndings|ding)\s*:\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"^(?:find|findin)\s*:\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = self._remove_trailing_fragment(
            cleaned
        )

        cleaned = re.sub(
            r"\s+([,.!?;:])",
            r"\1",
            cleaned,
        )

        return cleaned.strip()

    def _clean_llm_output(
        self,
        text: str,
    ) -> str:
        cleaned = text.strip()

        cleaned = re.sub(
            r"^```(?:text|markdown)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

        cleaned = re.sub(
            r"\s+",
            " ",
            cleaned,
        )

        cleaned = re.sub(
            r"\s+([,.!?;:])",
            r"\1",
            cleaned,
        )

        cleaned = self._remove_meta_text(
            cleaned
        )

        cleaned = self._remove_numeric_corruption(
            cleaned
        )
        cleaned = self._remove_temporal_comparison_text(
            cleaned
        )

        return cleaned.strip()

    def _remove_trailing_fragment(
        self,
        text: str,
    ) -> str:
        cleaned = text.strip()

        patterns = [
            r"\s+No$",
            r"\s+The$",
            r"\s+There$",
            r"\s+This$",
            r"\s+That$",
            r"\s+And$",
            r"\s+Or$",
            r"\s+With$",
            r"\s+Without$",
            r"\s+In$",
            r"\s+On$",
            r"\s+At$",
            r"\s+For$",
            r"\s+Of$",
            r"\s+To$",
        ]

        for pattern in patterns:
            cleaned = re.sub(
                pattern,
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

        return cleaned.strip(" ,;:-")

    def _remove_meta_text(
        self,
        text: str,
    ) -> str:
        sentences = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        blocked = [
            r"\bas an ai\b",
            r"\bas a language model\b",
            r"\bi cannot\b",
            r"\bi can't\b",
            r"\bi am unable\b",
            r"\bthe source narrative\b",
            r"\bthe provided narrative\b",
            r"\bi have reformatted\b",
            r"\bi have cleaned\b",
        ]

        valid_sentences = []

        for sentence in sentences:
            if any(
                re.search(
                    pattern,
                    sentence,
                    flags=re.IGNORECASE,
                )
                for pattern in blocked
            ):
                continue

            valid_sentences.append(
                sentence.strip()
            )

        return " ".join(
            valid_sentences
        ).strip()

    def _remove_temporal_comparison_text(
        self,
        text: str,
    ) -> str:
        cleaned = text.strip()

        temporal_patterns = [
            r"\s*,?\s*which implies that there has been no change in the status of [^.?!]* since the (?:last|previous) (?:radiograph|study|examination)[.?!]?",
            r"\s*,?\s*which implies that there has been no change [^.?!]* since the (?:last|previous) (?:radiograph|study|examination)[.?!]?",
            r"\s*,?\s*compared with the (?:last|prior|previous) (?:radiograph|study|examination)[.?!]?",
            r"\s*,?\s*compared to the (?:last|prior|previous) (?:radiograph|study|examination)[.?!]?",
            r"\s*,?\s*since the (?:last|prior|previous) (?:radiograph|study|examination)[.?!]?",
            r"\s*,?\s*from the (?:last|prior|previous) (?:radiograph|study|examination)[.?!]?",
            r"\s*,?\s*(?:as\s+)?compared with (?:the\s+)?(?:last|prior|previous) imaging[.?!]?",
            r"\s*,?\s*(?:as\s+)?compared to (?:the\s+)?(?:last|prior|previous) imaging[.?!]?",
            r"\s*,?\s*(?:as\s+)?compared with (?:the\s+)?(?:prior|previous) imaging[.?!]?",
            r"\s*,?\s*(?:as\s+)?compared to (?:the\s+)?(?:prior|previous) imaging[.?!]?",
            r"\s*,?\s*(?:from|since) (?:the\s+)?(?:last|prior|previous) imaging[.?!]?",
            r"\s*,?\s*from previous imaging[.?!]?",
            r"\s*,?\s*since previous imaging[.?!]?",
            r"\s*,?\s*(?:compared with|compared to|as compared with|as compared to) previous imaging[.?!]?",
            r"\s*,?\s*(?:from|since) previous image[.?!]?",
            r"\s*,?\s*(?:compared with|compared to|as compared with|as compared to) previous image[.?!]?",
        ]

        for pattern in temporal_patterns:
            cleaned = re.sub(
                pattern,
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

        return cleaned.strip()

    def _remove_numeric_corruption(
        self,
        text: str,
    ) -> str:
        corrupted = re.search(
            r"(?:\b\w+\d+\.\d+)"
            r"(?:\s*,\s*\d+\.\d+){3,}",
            text,
        )

        if corrupted:
            prefix = text[: corrupted.start()].rstrip()

            if prefix:
                return prefix.rstrip(
                    " ,;:-"
                )

        return text

    def _validate_refinement(
        self,
        source: str,
        refined: str,
    ) -> bool:
        if not refined.strip():
            return False

        if self._contains_numeric_corruption(
            refined
        ):
            return False

        source_lower = source.lower()
        refined_lower = refined.lower()

        critical_negative_terms = [
            "no ",
            "no.",
            "without ",
            "not ",
            "negative ",
            "absent ",
            "unchanged ",
            "without evidence",
            "no evidence",
        ]

        source_negative_count = sum(
            source_lower.count(term)
            for term in critical_negative_terms
        )

        refined_negative_count = sum(
            refined_lower.count(term)
            for term in critical_negative_terms
        )

        if (
            source_negative_count > 0
            and refined_negative_count == 0
        ):
            return False

        source_numbers = re.findall(
            r"\b\d+(?:\.\d+)?\b",
            source_lower,
        )

        refined_numbers = re.findall(
            r"\b\d+(?:\.\d+)?\b",
            refined_lower,
        )

        important_numbers = {
            number
            for number in source_numbers
            if len(number) > 1
        }

        refined_number_set = set(
            refined_numbers
        )

        missing_numbers = (
            important_numbers
            - refined_number_set
        )

        if len(missing_numbers) >= 3:
            return False

        return True

    def _contains_numeric_corruption(
        self,
        text: str,
    ) -> bool:
        return bool(
            re.search(
                r"\d+\.\d+"
                r"(?:\s*,\s*\d+\.\d+){3,}",
                text,
            )
        )

    def _get_model_name(
        self,
    ) -> Optional[str]:
        model = getattr(
            self.ollama_client,
            "model",
            None,
        )

        if model:
            return str(model)

        model_name = getattr(
            self.ollama_client,
            "model_name",
            None,
        )

        if model_name:
            return str(model_name)

        return None