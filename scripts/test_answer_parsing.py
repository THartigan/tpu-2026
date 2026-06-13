import unittest
from decimal import Decimal
import re

from answer_parsing import (
    NUMBER_PATTERN,
    extract_dataset_answer,
    extract_number,
    normalise_number,
    numeric_ratio,
)


ANSWER_RE = re.compile(r"<answer>.*?({})".format(NUMBER_PATTERN.pattern), re.DOTALL)


class AnswerParsingTest(unittest.TestCase):
    def test_gsm8k_answer_extraction(self):
        self.assertEqual(extract_dataset_answer("work #### 42"), "42")
        self.assertEqual(extract_dataset_answer("work #### 42\nextra"), "42")
        self.assertEqual(extract_dataset_answer("work #### 42."), "42")

    def test_legacy_gsm8k_answer_extraction(self):
        self.assertEqual(
            extract_dataset_answer("work #### 42\nextra", legacy_gsm8k=True),
            "42\nextra",
        )

    def test_metamath_numeric_answer_extraction(self):
        cases = {
            "The answer is: 1,200.": "1,200",
            "The answer is: \\boxed{36}": "36",
            "The answer is: 3/4": "3/4",
            "The answer is: 12.5%": "12.5%",
            "The answer is: \\frac{3}{4}": "\\frac{3}{4}",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(extract_dataset_answer(text), expected)

    def test_symbolic_metamath_answers_are_filtered(self):
        self.assertIsNone(extract_dataset_answer("The answer is: x+1"))
        self.assertIsNone(extract_dataset_answer("The answer is: \\sqrt{2}"))

    def test_number_normalisation(self):
        cases = {
            "1200": Decimal("1200"),
            "1,200": Decimal("1200"),
            "$1,200.50": Decimal("1200.50"),
            "3/4": Decimal("0.75"),
            "\\frac{3}{4}": Decimal("0.75"),
            "12.5%": Decimal("0.125"),
            ".5": Decimal("0.5"),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(normalise_number(text), expected)

    def test_numeric_ratio(self):
        self.assertEqual(numeric_ratio("90", "100"), Decimal("0.9"))
        self.assertEqual(numeric_ratio("110", "100"), Decimal("1.1"))
        self.assertEqual(numeric_ratio("1,200", "1200"), Decimal("1"))
        self.assertEqual(numeric_ratio("50%", "0.5"), Decimal("1"))
        self.assertIsNone(numeric_ratio("1", "0"))
        self.assertIsNone(numeric_ratio("x", "1"))

    def test_answer_tag_number_extraction(self):
        response = "<reasoning>ok</reasoning><answer>$1,200.50</answer>"
        match = ANSWER_RE.search(response)
        self.assertIsNotNone(match)
        self.assertEqual(extract_number(match.group(1)), "$1,200.50")
        self.assertEqual(normalise_number(match.group(1)), Decimal("1200.50"))


if __name__ == "__main__":
    unittest.main()
