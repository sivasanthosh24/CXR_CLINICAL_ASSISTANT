import re
from typing import List

from .schemas import ChatQuery, GuardrailResult


class QueryGuardrail:
    def __init__(self):
        self.relevant_patterns = [
            r"\bview\b",
            r"\bap\b",
            r"\blateral\b",
            r"\bx[- ]?ray\b",
            r"\bcxr\b",
            r"\bchest\b",
            r"\bdisease\b",
            r"\bfinding\b",
            r"\bfindings\b",
            r"\babnormalit(?:y|ies)\b",
            r"\bprobabilit(?:y|ies)\b",
            r"\bconfidence\b",
            r"\bprediction\b",
            r"\bpredicted\b",
            r"\bdetected\b",
            r"\bdiagnos(?:is|e|ed|tic)\b",
            r"\bmodule\s*1\b",
            r"\bmodule\s*2\b",
            r"\bclinical\s+caption\b",
            r"\bclinical\s+narrative\b",
            r"\breport\b",
            r"\bnarrative\b",
            r"\bgrad[- ]?cam\b",
            r"\bheatmap\b",
            r"\blocalization\b",
            r"\bbounding\s+box\b",
            r"\bhighlight(?:ed|s)?\b",
            r"\bregion\b",
            r"\barea\b",
            r"\blung\b",
            r"\blungs\b",
            r"\bheart\b",
            r"\bcardiac\b",
            r"\bpleura\b",
            r"\bpleural\b",
            r"\bmediastinum\b",
            r"\bopacity\b",
            r"\bopacities\b",
            r"\binfiltrat(?:e|es|ion)\b",
            r"\bedema\b",
            r"\bpneumonia\b",
            r"\bconsolidation\b",
            r"\beffusion\b",
            r"\bpneumothorax\b",
            r"\bcardiomegaly\b",
            r"\batelectasis\b",
            r"\bnodule\b",
            r"\bmass\b",
            r"\bfracture\b",
            r"\bhernia\b",
            r"\bnormal\b",
            r"\babnormal\b",
            r"\bimpression\b",
            r"\bsummar(?:y|ize|ise|ized|ised)\b",
            r"\bexplain\b",
            r"\bexplanation\b",
            r"\bdescribe\b",
            r"\bdescription\b",
            r"\banaly(?:ze|sis|sed|se)\b",
            r"\banalysis\b",
            r"\bresult(?:s)?\b",
            r"\boutput(?:s)?\b",
            r"\bwhat\s+did\b",
            r"\bwhat\s+was\b",
            r"\bwhat\s+are\b",
            r"\bwhat\s+is\b",
            r"\bwhich\b",
            r"\bhow\s+confident\b",
        ]

        self.blocked_patterns = [
            r"\bignore\s+(all\s+)?previous\s+instructions\b",
            r"\bignore\s+(the\s+)?system\s+prompt\b",
            r"\bignore\s+(your\s+)?instructions\b",
            r"\bforget\s+(all\s+)?previous\s+instructions\b",
            r"\breveal\s+(your\s+)?system\s+prompt\b",
            r"\bshow\s+(me\s+)?your\s+system\s+prompt\b",
            r"\breveal\s+(your\s+)?hidden\s+instructions\b",
            r"\bdisclose\s+(your\s+)?instructions\b",
            r"\bdeveloper\s+message\b",
            r"\bsystem\s+message\b",
            r"\bprompt\s+injection\b",
            r"\bact\s+as\s+an?\s+unrestricted\b",
            r"\bdo\s+anything\s+now\b",
            r"\bjailbreak\b",
            r"\bdisable\s+(the\s+)?guardrail\b",
            r"\bdisable\s+(the\s+)?safety\b",
        ]

        self.high_risk_patterns = [
            r"\bprescri(?:be|ption)\b",
            r"\bmedication\b",
            r"\bmedicine\b",
            r"\bdosage\b",
            r"\bdose\b",
            r"\btreatment\s+plan\b",
            r"\bwhat\s+drug\b",
            r"\bwhich\s+drug\b",
            r"\bshould\s+i\s+take\b",
            r"\bshould\s+the\s+patient\s+take\b",
            r"\bself[- ]?treat\b",
            r"\bdiagnose\s+me\b",
            r"\bwhat\s+disease\s+do\s+i\s+have\b",
            r"\bdo\s+i\s+have\b",
            r"\bwill\s+i\s+die\b",
            r"\blife\s+expectancy\b",
        ]

        self.irrelevant_patterns = [
            r"\bweather\b",
            r"\btemperature\b",
            r"\brecipe\b",
            r"\bcook\b",
            r"\bsports?\b",
            r"\bfootball\b",
            r"\bcricket\b",
            r"\bmovie\b",
            r"\bsong\b",
            r"\bmusic\b",
            r"\bpolitic(?:s|al)\b",
            r"\belection\b",
            r"\bpresident\b",
            r"\bprime\s+minister\b",
            r"\bstock(?:s)?\b",
            r"\bcrypto(?:currency)?\b",
            r"\bbitcoin\b",
            r"\bprogram(?:ming|me)?\b",
            r"\bpython\b",
            r"\bjavascript\b",
            r"\bjava\b",
            r"\bc\+\+\b",
            r"\bcode\b",
            r"\bdebug\b",
            r"\bdebugging\b",
            r"\bhomework\b",
            r"\bessay\b",
            r"\bpoem\b",
            r"\bjoke\b",
            r"\bstory\b",
            r"\btravel\b",
            r"\bflight\b",
            r"\bhotel\b",
        ]

    @staticmethod
    def _find_matches(text: str, patterns: List[str]) -> List[str]:
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

    def evaluate(self, query: ChatQuery) -> GuardrailResult:
        question = query.normalized()

        if not question:
            return GuardrailResult(
                allowed=False,
                reason="The question is empty.",
                category="empty_query",
                confidence=1.0,
            )

        blocked_matches = self._find_matches(
            question,
            self.blocked_patterns,
        )

        if blocked_matches:
            return GuardrailResult(
                allowed=False,
                reason=(
                    "The request contains an instruction attempting "
                    "to bypass the chatbot control policy."
                ),
                category="prompt_injection",
                confidence=1.0,
                matched_terms=blocked_matches,
            )

        high_risk_matches = self._find_matches(
            question,
            self.high_risk_patterns,
        )

        if high_risk_matches:
            return GuardrailResult(
                allowed=False,
                reason=(
                    "The request asks for information or advice that "
                    "cannot be provided from the module outputs."
                ),
                category="clinical_advice_or_independent_diagnosis",
                confidence=1.0,
                matched_terms=high_risk_matches,
            )

        irrelevant_matches = self._find_matches(
            question,
            self.irrelevant_patterns,
        )

        if irrelevant_matches:
            return GuardrailResult(
                allowed=False,
                reason=(
                    "The request is outside the scope of the current "
                    "chest X-ray analysis modules."
                ),
                category="out_of_scope",
                confidence=1.0,
                matched_terms=irrelevant_matches,
            )

        relevant_matches = self._find_matches(
            question,
            self.relevant_patterns,
        )

        if relevant_matches:
            return GuardrailResult(
                allowed=True,
                reason=(
                    "The request is related to the available "
                    "chest X-ray analysis outputs."
                ),
                category="module_related",
                confidence=0.85,
                matched_terms=relevant_matches,
            )

        return GuardrailResult(
            allowed=False,
            reason=(
                "The request could not be associated with the "
                "available Module 1, Module 2, view classification, "
                "Grad-CAM, or weak-localization outputs."
            ),
            category="unknown_scope",
            confidence=0.75,
        )