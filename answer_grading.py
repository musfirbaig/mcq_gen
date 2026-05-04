"""
Utilities for deterministic MCQ answer grading metadata.
"""

from datetime import datetime
from typing import Dict, Any

VALID_OPTION_KEYS = ("a", "b", "c", "d")


def _normalize_key(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def build_answer_grading(mcq: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build deterministic answer_grading payload for an MCQ.
    Falls back to skipped_invalid_shape when options/correct answer are invalid.
    """
    options = mcq.get("options") or {}
    correct_key = _normalize_key(mcq.get("correct_answer"))
    created_at = datetime.utcnow()

    has_valid_shape = (
        isinstance(options, dict)
        and all(key in options for key in VALID_OPTION_KEYS)
        and correct_key in VALID_OPTION_KEYS
    )

    if not has_valid_shape:
        return {
            "method": "skipped_invalid_shape",
            "graded_at": created_at,
            "correct_keys": [],
            "incorrect_keys": [],
            "per_option": {},
        }

    per_option = {}
    incorrect_keys = []
    for key in VALID_OPTION_KEYS:
        is_correct = key == correct_key
        per_option[key] = {"is_correct": is_correct}
        if not is_correct:
            incorrect_keys.append(key)

    return {
        "method": "deterministic_key_match",
        "graded_at": created_at,
        "correct_keys": [correct_key],
        "incorrect_keys": incorrect_keys,
        "per_option": per_option,
    }
