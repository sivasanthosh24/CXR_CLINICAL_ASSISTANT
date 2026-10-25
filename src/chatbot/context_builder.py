from typing import Any, Dict, List, Optional
from .schemas import ModuleEvidence
class ContextBuilder:
    def __init__(self):
        pass
    def _safe_get(
        self,
        data: Optional[Dict[str, Any]],
        key: str,
        default=None,
    ):
        if not isinstance(data, dict):
            return default
        return data.get(key, default)
    def _contains_any(
        self,
        question: str,
        terms: List[str],
    ) -> bool:
        question = question.lower().strip()
        for term in terms:
            term = term.lower().strip()
            if not term:
                continue
            if len(term) <= 3 and term.isalpha():
                words = (
                    question
                    .replace("-", " ")
                    .replace("/", " ")
                    .split()
                )
                if term in words:
                    return True
            elif term in question:
                return True
        return False
#detect the intent
    def detect_intent(
        self,
        question: str,
    ) -> str:
        question = question.lower().strip()
        complete_terms = [
            "complete analysis",
            "full analysis",
            "overall analysis",
            "complete result",
            "all results",
            "all findings",
            "everything",
            "summarize everything",
            "give me everything",
            "full report and prediction",
            "complete clinical analysis",
        ]
        if self._contains_any(
            question,
            complete_terms,
        ):
            return "general"
        injection_terms = [
            "ignore previous instructions",
            "ignore your previous instructions",
            "ignore all previous instructions",
            "forget your instructions",
            "bypass your instructions",
            "override your instructions",
            "ignore the instructions",
            "disregard previous instructions",
            "disregard your instructions",
            "act independently",
            "diagnose independently",
            "independent diagnosis",
        ]
        if self._contains_any(
            question,
            injection_terms,
        ):
            return "unsafe"
        treatment_terms = [
            "treatment",
            "medication",
            "medicine",
            "drug",
            "prescription",
            "therapy",
            "what should the patient take",
            "what medicine",
            "what medication",
        ]
        if self._contains_any(
            question,
            treatment_terms,
        ):
            return "unsafe"
        view_terms = [
            "predicted view",
            "chest x-ray view",
            "chest xray view",
            "x-ray view",
            "xray view",
            "projection",
            "ap view",
            "pa view",
            "lateral view",
            "frontal view",
            "which view",
            "what view",
            "view classification",
        ]
        if self._contains_any(
            question,
            view_terms,
        ):
            return "view"
        gradcam_terms = [
            "grad-cam",
            "gradcam",
            "activation map",
            "heatmap",
            "heat map",
            "explanation map",
            "visual explanation",
        ]
        if self._contains_any(
            question,
            gradcam_terms,
        ):
            return "gradcam"
        localization_terms = [
            "bounding box",
            "bounding-box",
            "localization",
            "localized region",
            "localized area",
            "region of interest",
            "area of interest",
            "where is the abnormal region",
            "where is the abnormality",
            "where is the finding",
            "location of the finding",
            "localized finding",
        ]
        if self._contains_any(
            question,
            localization_terms,
        ):
            return "localization"
        narrative_terms = [
            "module 2 clinical narrative",
            "module 2 narrative",
            "clinical narrative",
            "clinical report",
            "radiology report",
            "radiology narrative",
            "generated report",
            "generated narrative",
            "summarize the report",
            "summarize the narrative",
            "clinical caption",
            "caption",
            "report",
            "narrative",
        ]
        if self._contains_any(
            question,
            narrative_terms,
        ):
            return "narrative"
        disease_terms = [
            "disease probability",
            "disease probabilities",
            "disease prediction",
            "disease predictions",
            "predicted disease",
            "predicted diseases",
            "abnormality probability",
            "abnormality probabilities",
            "abnormality prediction",
            "abnormality predictions",
            "predicted abnormality",
            "predicted abnormalities",
            "positive finding",
            "positive findings",
            "disease",
            "diseases",
            "prediction",
            "predictions",
            "abnormality",
            "abnormalities",
            "probability",
            "probabilities",
            "confidence score",
            "disease score",
            "disease scores",
            "what diseases",
            "which diseases",
            "what abnormalities",
            "which abnormalities",
            "module 1 prediction",
            "module 1 predictions",
            "module 1 probabilities",
        ]
        if self._contains_any(
            question,
            disease_terms,
        ):
            return "disease_prediction"
        finding_terms = [
            "key findings",
            "key finding",
            "system findings",
            "reported findings",
            "reported finding",
            "findings reported by the system",
            "findings produced by the system",
        ]
        if self._contains_any(
            question,
            finding_terms,
        ):
            return "general"
        return "general"
    def _extract_view_evidence(
        self,
        view_result: Optional[Dict[str, Any]],
    ):
        if not isinstance(view_result, dict):
            return None
        predicted_view = self._safe_get(
            view_result,
            "predicted_view",
        )
        confidence = self._safe_get(
            view_result,
            "confidence",
        )
        if (
            predicted_view is None
            and confidence is None
        ):
            return None
        return {
            "predicted_view": predicted_view,
            "confidence": confidence,
        }
    def _extract_module1_evidence(
        self,
        module1_result: Optional[Dict[str, Any]],
    ):
        if not isinstance(module1_result, dict):
            return None
        predictions = self._safe_get(
            module1_result,
            "disease_predictions",
        )
        if predictions is None:
            predictions = self._safe_get(
                module1_result,
                "predictions",
            )
        if predictions is None:
            predictions = self._safe_get(
                module1_result,
                "disease_probabilities",
            )
        if predictions is None:
            prediction_result = self._safe_get(
                module1_result,
                "prediction",
            )
            if isinstance(prediction_result, dict):
                predictions = self._safe_get(
                    prediction_result,
                    "ranked_findings",
                )
                if predictions is None:
                    predictions = self._safe_get(
                        prediction_result,
                        "positive_findings",
                    )
                if predictions is None:
                    predictions = prediction_result
        if predictions is None:
            return None
        if isinstance(predictions, list):
            unique_predictions = []
            seen = set()
            for item in predictions:
                if not isinstance(item, dict):
                    key = str(item)
                else:
                    key = (
                        item.get("disease")
                        or item.get("label")
                        or item.get("finding")
                        or item.get("name")
                        or str(item)
                    )
                key = str(key).strip().lower()
                if key not in seen:
                    seen.add(key)
                    unique_predictions.append(item)
            predictions = unique_predictions
        return {
            "disease_predictions": predictions,
        }
    def _extract_module2_evidence(
        self,
        module2_result: Optional[Dict[str, Any]],
    ):
        if not isinstance(module2_result, dict):
            return None
        narrative = self._safe_get(
            module2_result,
            "narrative",
        )
        if narrative is None:
            narrative = self._safe_get(
                module2_result,
                "clinical_narrative",
            )
        if narrative is None:
            narrative = self._safe_get(
                module2_result,
                "report",
            )
        if narrative is None:
            narrative = self._safe_get(
                module2_result,
                "caption",
            )
        if narrative is None:
            return None
        return {
            "narrative": narrative,
        }
    def _extract_gradcam_evidence(
        self,
        module1_result: Optional[Dict[str, Any]],
    ):
        if not isinstance(module1_result, dict):
            return None
        gradcam = self._safe_get(
            module1_result,
            "gradcam",
        )
        if gradcam is None:
            gradcam = self._safe_get(
                module1_result,
                "grad_cam",
            )
        if gradcam is None:
            gradcam = self._safe_get(
                module1_result,
                "gradcam_result",
            )
        if gradcam is None:
            return None
        return {
            "gradcam": gradcam,
        }
    def _extract_localization_evidence(
        self,
        module1_result: Optional[Dict[str, Any]],
    ):
        if not isinstance(module1_result, dict):
            return None
        localization = self._safe_get(
            module1_result,
            "bounding_box",
        )
        if localization is None:
            localization = self._safe_get(
                module1_result,
                "bounding_box_result",
            )
        if localization is None:
            localization = self._safe_get(
                module1_result,
                "localization",
            )
        if localization is None:
            localization = self._safe_get(
                module1_result,
                "weak_localization",
            )
        if localization is None:
            prediction = self._safe_get(
                module1_result,
                "prediction",
                {},
            )
            if isinstance(prediction, dict):
                localization = self._safe_get(
                    prediction,
                    "weak_localization",
                )
        if localization is None:
            return None
        return {
            "localization": localization,
        }
    def build(
        self,
        view_result: Optional[Dict[str, Any]] = None,
        module1_result: Optional[Dict[str, Any]] = None,
        module2_result: Optional[Dict[str, Any]] = None,
    ) -> ModuleEvidence:
        return ModuleEvidence(
            view_classification=self._extract_view_evidence(
                view_result
            ),
            module1_prediction=self._extract_module1_evidence(
                module1_result
            ),
            gradcam=self._extract_gradcam_evidence(
                module1_result
            ),
            weak_localization=(
                self._extract_localization_evidence(
                    module1_result
                )
            ),
            module2_narrative=(
                self._extract_module2_evidence(
                    module2_result
                )
            ),
        )
    def build_evidence(
        self,
        view_result: Optional[Dict[str, Any]] = None,
        module1_result: Optional[Dict[str, Any]] = None,
        module2_result: Optional[Dict[str, Any]] = None,
    ) -> ModuleEvidence:
        return self.build(
            view_result=view_result,
            module1_result=module1_result,
            module2_result=module2_result,
        )
    def select_relevant_evidence(
        self,
        question: str,
        evidence: ModuleEvidence,
    ) -> ModuleEvidence:
        intent = self.detect_intent(question)
        if intent == "unsafe":
            return ModuleEvidence()
        if intent == "view":
            return ModuleEvidence(
                view_classification=evidence.view_classification
            )
        if intent == "disease_prediction":
            return ModuleEvidence(
                module1_prediction=evidence.module1_prediction
            )
        if intent == "gradcam":
            return ModuleEvidence(
                gradcam=evidence.gradcam
            )
        if intent == "localization":
            return ModuleEvidence(
                weak_localization=evidence.weak_localization
            )
        if intent == "narrative":
            return ModuleEvidence(
                module2_narrative=evidence.module2_narrative
            )
        if intent == "general":
            return evidence
        return evidence
    def get_available_sources(
        self,
        evidence: ModuleEvidence,
    ) -> List[str]:
        sources = []
        if evidence.view_classification:
            sources.append(
                "View Classification"
            )
        if evidence.module1_prediction:
            sources.append(
                "Module 1 Prediction"
            )
        if evidence.gradcam:
            sources.append(
                "Grad-CAM"
            )
        if evidence.weak_localization:
            sources.append(
                "Weak Localization"
            )
        if evidence.module2_narrative:
            sources.append(
                "Module 2 Clinical Narrative"
            )
        return sources
    def build_text_context(
        self,
        evidence: ModuleEvidence,
    ) -> str:
        sections = []
        if evidence.view_classification:
            view = evidence.view_classification
            predicted_view = view.get(
                "predicted_view"
            )
            confidence = view.get(
                "confidence"
            )
            section = "[VIEW CLASSIFICATION]\n"
            if predicted_view is not None:
                section += (
                    f"predicted_view: "
                    f"{predicted_view}\n"
                )
            if confidence is not None:
                section += (
                    f"confidence: "
                    f"{confidence}\n"
                )
            sections.append(
                section.rstrip()
            )
        if evidence.module1_prediction:
            prediction = (
                evidence.module1_prediction
            )
            disease_predictions = (
                prediction.get(
                    "disease_predictions"
                )
            )
            if disease_predictions is not None:
                section = (
                    "[MODULE 1 PREDICTION]\n"
                )
                section += (
                    "disease_predictions: "
                    f"{disease_predictions}"
                )
                sections.append(section)
        if evidence.gradcam:
            section = "[GRAD-CAM]\n"
            section += (
                f"gradcam: "
                f"{evidence.gradcam}"
            )
            sections.append(section)
        if evidence.weak_localization:
            section = (
                "[WEAK LOCALIZATION]\n"
            )
            section += (
                "localization: "
                f"{evidence.weak_localization}"
            )
            sections.append(section)
        if evidence.module2_narrative:
            narrative = (
                evidence.module2_narrative.get(
                    "narrative"
                )
            )
            if narrative is not None:
                section = (
                    "[MODULE 2 CLINICAL NARRATIVE]\n"
                )
                section += (
                    f"narrative: {narrative}"
                )
                sections.append(section)
        if not sections:
            return (
                "[NO RELEVANT MODULE "
                "EVIDENCE AVAILABLE]"
            )
        return "\n\n".join(sections)
    def build_text(
        self,
        evidence: ModuleEvidence,
    ) -> str:
        return self.build_text_context(
            evidence
        )
