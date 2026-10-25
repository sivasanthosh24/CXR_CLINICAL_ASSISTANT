import re
from typing import Any, Dict, List

from .schemas import GuardrailResult


class OutputGuardrail:
    def __init__(self):
        self.blocked_patterns = [
            r"\bignore\s+(all\s+)?previous\s+instructions\b",
            r"\bignore\s+(the\s+)?system\s+prompt\b",
            r"\breveal\s+(your\s+)?system\s+prompt\b",
            r"\breveal\s+(your\s+)?hidden\s+instructions\b",
            r"\bdeveloper\s+message\b",
            r"\bsystem\s+message\b",
            r"\bjailbreak\b",
        ]

        self.unsupported_claim_patterns = [
            r"\bi\s+(can|could)\s+see\b",
            r"\bi\s+see\s+that\b",
            r"\bon\s+the\s+x[- ]?ray\b",
            r"\blooking\s+at\s+the\s+image\b",
            r"\bfrom\s+the\s+image\s+i\s+can\b",
            r"\bthe\s+image\s+shows\b",
            r"\bthe\s+x[- ]?ray\s+shows\b",
            r"\bvisually\s+confirmed\b",
            r"\bdefinitively\b",
        ]

        self.advice_patterns = [
            r"\byou\s+should\s+take\b",
            r"\byou\s+should\s+start\b",
            r"\bthe\s+patient\s+should\s+take\b",
            r"\bthe\s+patient\s+needs\s+to\s+take\b",
            r"\bprescribe\b",
            r"\bprescription\b",
            r"\bmedication\s+recommendation\b",
            r"\btreatment\s+recommendation\b",
            r"\bstart\s+treatment\b",
            r"\bchange\s+medication\b",
            r"\bincrease\s+the\s+dose\b",
            r"\bdecrease\s+the\s+dose\b",
        ]

        self.uncertainty_phrases = [
            "not available from the current analysis",
            "not reported in the available outputs",
            "not provided by the current analysis",
            "cannot be determined from the available outputs",
            "cannot determine from the available analysis",
            "not available in the module outputs",
        ]

    @staticmethod
    def _find_matches(
        text: str,
        patterns: List[str],
    ) -> List[str]:
        matches = []

        for pattern in patterns:
            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                matches.append(match.group(0))

        return matches

    def evaluate(
        self,
        response_text: str,
    ) -> GuardrailResult:
        text = " ".join(
            str(response_text or "").strip().split()
        )

        if not text:
            return GuardrailResult(
                allowed=False,
                reason="The generated response is empty.",
                category="empty_response",
                confidence=1.0,
            )

        blocked_matches = self._find_matches(
            text,
            self.blocked_patterns,
        )

        if blocked_matches:
            return GuardrailResult(
                allowed=False,
                reason=(
                    "The generated response contains content "
                    "associated with instruction manipulation."
                ),
                category="prompt_injection_content",
                confidence=1.0,
                matched_terms=blocked_matches,
            )

        advice_matches = self._find_matches(
            text,
            self.advice_patterns,
        )

        if advice_matches:
            return GuardrailResult(
                allowed=False,
                reason=(
                    "The generated response contains treatment or "
                    "medication advice outside the permitted scope."
                ),
                category="medical_advice",
                confidence=1.0,
                matched_terms=advice_matches,
            )

        unsupported_matches = self._find_matches(
            text,
            self.unsupported_claim_patterns,
        )

        if unsupported_matches:
            return GuardrailResult(
                allowed=False,
                reason=(
                    "The generated response appears to make a direct "
                    "image-observation claim rather than reporting "
                    "available module output."
                ),
                category="unsupported_image_claim",
                confidence=0.90,
                matched_terms=unsupported_matches,
            )

        return GuardrailResult(
            allowed=True,
            reason="The generated response passed the output guardrail.",
            category="validated",
            confidence=0.85,
            matched_terms=[],
        )

    def validate(
        self,
        response_text: str,
    ) -> GuardrailResult:
        return self.evaluate(response_text)

    def fallback_response(
        self,
        result: GuardrailResult,
    ) -> str:
        if result.category == "medical_advice":
            return (
                "I can only explain the available chest X-ray "
                "analysis outputs. I cannot provide medication, "
                "treatment, or personalized medical advice."
            )

        if result.category == "unsupported_image_claim":
            return (
                "I can only report findings provided by the "
                "available analysis modules. I cannot independently "
                "interpret the X-ray image."
            )

        if result.category == "prompt_injection_content":
            return (
                "I can only provide responses within the controlled "
                "chest X-ray analysis scope."
            )

        if result.category == "empty_response":
            return (
                "No validated response was generated from the "
                "available analysis outputs."
            )

        return (
            "The generated response could not be validated against "
            "the controlled analysis scope."
        )

    def get_validation_metadata(
        self,
        result: GuardrailResult,
    ) -> Dict[str, Any]:
        return {
            "passed": result.allowed,
            "category": result.category,
            "confidence": result.confidence,
            "reason": result.reason,
            "matched_terms": result.matched_terms,
        }