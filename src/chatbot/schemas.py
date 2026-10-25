from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModuleEvidence:
    view_classification: Dict[str, Any] = field(default_factory=dict)
    module1_prediction: Dict[str, Any] = field(default_factory=dict)
    gradcam: Dict[str, Any] = field(default_factory=dict)
    weak_localization: Dict[str, Any] = field(default_factory=dict)
    module2_narrative: Dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not any(
            [
                self.view_classification,
                self.module1_prediction,
                self.gradcam,
                self.weak_localization,
                self.module2_narrative,
            ]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "view_classification": self.view_classification,
            "module1_prediction": self.module1_prediction,
            "gradcam": self.gradcam,
            "weak_localization": self.weak_localization,
            "module2_narrative": self.module2_narrative,
        }


@dataclass
class ChatQuery:
    question: str
    session_id: Optional[str] = None

    def normalized(self) -> str:
        return " ".join(self.question.strip().split())


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str
    category: str
    confidence: float = 1.0
    matched_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "category": self.category,
            "confidence": self.confidence,
            "matched_terms": self.matched_terms,
        }


@dataclass
class OllamaResponse:
    text: str
    model: str
    success: bool
    error: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "success": self.success,
            "error": self.error,
            "raw_response": self.raw_response,
        }


@dataclass
class ChatResponse:
    text: str
    allowed: bool
    category: str
    source_modules: List[str] = field(default_factory=list)
    guardrail_passed: bool = False
    output_guardrail_passed: bool = False
    model: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "allowed": self.allowed,
            "category": self.category,
            "source_modules": self.source_modules,
            "guardrail_passed": self.guardrail_passed,
            "output_guardrail_passed": self.output_guardrail_passed,
            "model": self.model,
            "error": self.error,
            "metadata": self.metadata,
        }