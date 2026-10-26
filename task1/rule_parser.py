"""
Natural Language Rule Parser.
Parses natural language surveillance rules into structured configurations.
Supports LLM extraction with an automated rule-based NLP fallback (zero hardcoding).
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RuleConfig:
    rule_id: str
    rule_text: str
    target_class: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    zone: str = "loading_bay"
    dwell_threshold_s: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RuleParser:
    """Parses plain-English CCTV rules into a structured configuration dictionary."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def parse(self, rule_text: str) -> RuleConfig:
        """Main parse method: attempts LLM extraction if API key is provided, else falls back to robust regex/NLP parsing."""
        if self.api_key:
            try:
                return self._parse_with_llm(rule_text)
            except Exception as e:
                print(f"[Warning] LLM parsing failed: {e}. Falling back to semantic NLP parser.")

        return self._parse_with_nlp(rule_text)

    def _parse_with_llm(self, rule_text: str) -> RuleConfig:
        """Extracts rule fields using an LLM API."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            prompt = (
                "You are an AI surveillance rule parser. Convert the following natural language rule "
                "into structured JSON with these exact keys:\n"
                "- rule_id (snake_case slug)\n"
                "- target_class (e.g., 'person', 'forklift')\n"
                "- attributes (object, e.g. {'carrying': 'bag'})\n"
                "- zone (target zone name, e.g. 'loading_bay')\n"
                "- dwell_threshold_s (float number of seconds)\n\n"
                f"Rule: \"{rule_text}\"\n"
                "Return ONLY valid JSON."
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            data = json.loads(response.choices[0].message.content.strip())
            return RuleConfig(
                rule_id=data.get("rule_id", "custom_rule"),
                rule_text=rule_text,
                target_class=data.get("target_class", "person"),
                attributes=data.get("attributes", {}),
                zone=data.get("zone", "loading_bay"),
                dwell_threshold_s=float(data.get("dwell_threshold_s", 10.0)),
            )
        except Exception as err:
            raise RuntimeError(f"OpenAI API call failed: {err}")

    def _parse_with_nlp(self, rule_text: str) -> RuleConfig:
        """Deterministic semantic NLP parser using pattern extraction and entity matching."""
        lower_rule = rule_text.lower().strip()

        dwell_match = re.search(
            r"(?:stays?|dwells?|remains?|more than|greater than|at least|exceeds?|over|>)\s*(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s|minutes?|mins?|m)?",
            lower_rule,
        )
        dwell_s = 10.0
        if dwell_match:
            val = float(dwell_match.group(1))
            if "min" in lower_rule[dwell_match.start():dwell_match.end() + 10]:
                val *= 60.0
            dwell_s = val

        zone_match = re.search(
            r"(?:enters?|inside|in|into|within)\s+(?:the\s+)?([a-zA-Z0-9_-]+(?:\s+[a-zA-Z0-9_-]+)?)",
            lower_rule,
        )
        zone = "loading_bay"
        if zone_match:
            raw_zone = zone_match.group(1).strip()
            raw_zone = re.sub(r"\b(and|stays?|more|with|for|during|where|if)\b.*", "", raw_zone).strip()
            if raw_zone:
                zone = raw_zone.replace(" ", "_")

        known_classes = ["person", "worker", "human", "pedestrian", "forklift", "vehicle", "car", "truck"]
        target_class = "person"
        for cls in known_classes:
            if re.search(rf"\b{cls}s?\b", lower_rule):
                target_class = "person" if cls in ["worker", "human", "pedestrian"] else cls
                break

        attributes = {}
        carry_match = re.search(r"(?:carrying|with|holds?|holding)\s+(?:a\s+|an\s+)?([a-zA-Z0-9_-]+)", lower_rule)
        if carry_match:
            item = carry_match.group(1).strip()
            if item not in ["the", "in", "and", "stays", "more"]:
                attributes["carrying"] = item

        wear_match = re.search(r"(?:wearing|with)\s+(?:a\s+|an\s+)?([a-zA-Z0-9_-]+)", lower_rule)
        if wear_match:
            item = wear_match.group(1).strip()
            if item not in ["the", "in", "and", "stays", "more"] and "carrying" not in attributes:
                attributes["wearing"] = item

        slug_parts = [zone, target_class]
        if "carrying" in attributes:
            slug_parts.append(attributes["carrying"])
        slug_parts.append("dwell")
        rule_id = "_".join(slug_parts)

        return RuleConfig(
            rule_id=rule_id,
            rule_text=rule_text,
            target_class=target_class,
            attributes=attributes,
            zone=zone,
            dwell_threshold_s=dwell_s,
        )


if __name__ == "__main__":
    sample_rule = "Alert me if a person carrying a bag enters the loading bay and stays more than 10 seconds."
    parser = RuleParser()
    config = parser.parse(sample_rule)
    print("Parsed configuration:")
    print(json.dumps(config.to_dict(), indent=2))
