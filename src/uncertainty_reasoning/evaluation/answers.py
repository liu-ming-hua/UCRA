from __future__ import annotations

import json
import re
from fractions import Fraction


def extract_terminal_json_answer(text: str) -> str | None:
    """Extract an arbitrary scalar answer from a terminal JSON commitment."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    candidate = lines[-1]
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r'(\{\s*"answer"\s*:\s*.+\})\s*$', text)
        if not match:
            return None
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict) or "answer" not in value:
        return None
    answer = value["answer"]
    if isinstance(answer, (str, int, float, bool)):
        return str(answer).strip()
    return None


def extract_boxed(text: str) -> str | None:
    """Extract the last balanced \boxed{...} or \fbox{...} expression."""
    starts = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    for match in reversed(starts):
        depth = 1
        begin = match.end()
        for index in range(begin, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[begin:index]
    return None


def extract_explicit_answer(text: str) -> str | None:
    """Extract an explicitly marked final answer, without guessing from the last line."""
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    patterns = (
        r"(?i)(?:final answer|answer is)\s*[:=]?\s*([^\n]+)",
        r"(?m)^\s*####\s*(.+)$",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]
    return None


def extract_answer(text: str) -> str:
    """Backward-compatible extraction with a last-line fallback.

    New trajectory generation must use ``extract_explicit_answer`` so unfinished
    reasoning is not silently converted into a final answer.
    """
    explicit = extract_explicit_answer(text)
    if explicit is not None:
        return explicit
    return text.strip().splitlines()[-1] if text.strip() else ""


def normalize_answer(answer: str) -> str:
    value = answer.strip()
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    whole_text = re.fullmatch(r"\\text\{([^{}]+)\}", value)
    if whole_text:
        value = whole_text.group(1)
    value = value.replace("$", "").replace("\\,", "").replace(" ", "")
    value = value.replace("\\left", "").replace("\\right", "")
    value = re.sub(r"(?i)(?:degrees?|°)$", r"^\\circ", value)
    value = value.rstrip(".。")
    frac = re.fullmatch(r"-?\\frac\{(-?\d+)\}\{(-?\d+)\}", value)
    if frac:
        sign = -1 if value.startswith("-") else 1
        numerator = abs(int(frac.group(1))) * sign
        denominator = int(frac.group(2))
        if denominator:
            return str(Fraction(numerator, denominator))
    try:
        return str(Fraction(value.replace(",", "")))
    except (ValueError, ZeroDivisionError):
        return value.lower()


def normalize_bbh_answer(answer: str) -> str:
    """Normalize BBH short answers, including interchangeable ``(A)``/``A`` labels."""

    value = normalize_answer(answer)
    label = re.fullmatch(r"\(([a-z])\)", value)
    return label.group(1) if label else value


def answers_equal(prediction: str, gold: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(gold)
