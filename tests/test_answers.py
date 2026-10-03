import unittest

from uncertainty_reasoning.evaluation.answers import (
    extract_answer,
    extract_explicit_answer,
    extract_terminal_json_answer,
    normalize_answer,
    normalize_bbh_answer,
)


def test_extract_terminal_json_answer_accepts_short_answer() -> None:
    assert extract_terminal_json_answer('Reasoning\n{"answer": "(C)"}') == "(C)"


def test_extract_terminal_json_answer_rejects_nonterminal_commitment() -> None:
    assert extract_terminal_json_answer('{"answer": "A"}\nMore reasoning') is None


def test_normalize_bbh_choice_label() -> None:
    assert normalize_bbh_answer("(B)") == normalize_bbh_answer("B") == "b"


class AnswerTests(unittest.TestCase):
    def test_nested_box(self):
        self.assertEqual(extract_answer(r"work \boxed{\frac{1}{2}}"), r"\frac{1}{2}")

    def test_last_box_wins(self):
        self.assertEqual(extract_answer(r"\boxed{2}, corrected: \boxed{3}"), "3")

    def test_explicit_extraction_does_not_guess_from_unfinished_work(self):
        self.assertIsNone(extract_explicit_answer("unfinished reasoning\ntherefore maybe 12"))

    def test_numeric_normalization(self):
        self.assertEqual(normalize_answer("0.50"), "1/2")
        self.assertEqual(normalize_answer(r"\frac{2}{4}"), "1/2")
        self.assertEqual(normalize_answer(r"\dfrac{2}{4}"), "1/2")
        self.assertEqual(normalize_answer(r"\text{Evelyn}"), "evelyn")
        self.assertEqual(normalize_answer("90 degrees"), normalize_answer(r"90^\circ"))


if __name__ == "__main__":
    unittest.main()
