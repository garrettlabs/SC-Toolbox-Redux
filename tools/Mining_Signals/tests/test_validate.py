"""Unit tests for ocr.sc_ocr.validate — pure parsers / range guards.

Covers each public validator: signal, mass, percentage, instability,
and refinery cost. Focus is on real OCR-style inputs (junk chars,
missing decimals, leading icon glyphs) and the documented in-band /
out-of-band acceptance ranges.
"""

from __future__ import annotations

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.sc_ocr.validate import (
    SIGNAL_MIN,
    SIGNAL_MAX,
    MASS_MAX,
    validate_signal,
    validate_mass,
    validate_pct,
    validate_instability,
    validate_refinery_cost,
)


class TestValidateSignal(unittest.TestCase):

    def test_typical_value(self):
        self.assertEqual(validate_signal("5350"), 5350)

    def test_with_thousands_separator(self):
        # Comma stripped → 16050 is in range
        self.assertEqual(validate_signal("16,050"), 16050)

    def test_min_boundary(self):
        self.assertEqual(validate_signal(str(SIGNAL_MIN)), SIGNAL_MIN)

    def test_max_boundary(self):
        self.assertEqual(validate_signal(str(SIGNAL_MAX)), SIGNAL_MAX)

    def test_below_min_returns_none(self):
        self.assertIsNone(validate_signal("999"))

    def test_above_max_returns_none(self):
        # Random 6-digit value where stripping the leading char also
        # falls outside the band → None
        self.assertIsNone(validate_signal("999999"))

    def test_leading_glyph_dropped(self):
        # The HUD sometimes prefixes a decorative digit-like icon — the
        # validator should drop one leading char and re-try.  '95350'
        # is out of band but '5350' is in band.
        self.assertEqual(validate_signal("95350"), 5350)

    def test_empty_string(self):
        self.assertIsNone(validate_signal(""))

    def test_letters_only(self):
        # All non-digit content gets stripped to nothing
        self.assertIsNone(validate_signal("abcd"))

    def test_mixed_letters_and_digits(self):
        # Letters stripped — '5O350' → '5350' which is in band
        self.assertEqual(validate_signal("5O350"), 5350)

    def test_whitespace_stripped(self):
        self.assertEqual(validate_signal("  10700  "), 10700)


class TestValidateMass(unittest.TestCase):

    def test_typical_value(self):
        self.assertEqual(validate_mass("1234.5"), 1234.5)

    def test_integer(self):
        self.assertEqual(validate_mass("500"), 500.0)

    def test_double_dot_collapsed(self):
        # OCR sometimes injects a stray dot; collapse to one
        self.assertEqual(validate_mass("12..5"), 12.5)

    def test_below_min_returns_none(self):
        self.assertIsNone(validate_mass("0"))
        self.assertIsNone(validate_mass("0.05"))

    def test_above_max_returns_none(self):
        self.assertIsNone(validate_mass(str(MASS_MAX + 1)))

    def test_at_min_boundary(self):
        self.assertEqual(validate_mass("0.1"), 0.1)

    def test_letters_stripped(self):
        # 'kg' suffix is common — should be stripped
        self.assertEqual(validate_mass("1500.0kg"), 1500.0)

    def test_empty_string(self):
        self.assertIsNone(validate_mass(""))

    def test_dots_only(self):
        self.assertIsNone(validate_mass("..."))


class TestValidatePct(unittest.TestCase):

    def test_typical_value(self):
        self.assertEqual(validate_pct("87.5"), 87.5)

    def test_percent_sign_stripped(self):
        self.assertEqual(validate_pct("87.5%"), 87.5)

    def test_zero(self):
        self.assertEqual(validate_pct("0"), 0.0)

    def test_hundred(self):
        self.assertEqual(validate_pct("100"), 100.0)

    def test_above_hundred_recovered(self):
        # OCR trailing digit was a misread of '%' — recovers by
        # dropping from the right until it falls in [0, 100].
        result = validate_pct("875")
        # 875 is out of band, 87.5 isn't a candidate (we drop digits,
        # not insert dots), but 87 is — accepted.
        self.assertEqual(result, 87.0)

    def test_unrecoverable_returns_none(self):
        # All-digit cleaned form has nothing left after stripping
        self.assertIsNone(validate_pct("xyz"))

    def test_empty_string(self):
        self.assertIsNone(validate_pct(""))

    def test_whitespace_only(self):
        self.assertIsNone(validate_pct("   "))


class TestValidateInstability(unittest.TestCase):

    def test_typical_in_band(self):
        self.assertEqual(validate_instability("12.5"), 12.5)

    def test_zero(self):
        self.assertEqual(validate_instability("0"), 0.0)

    def test_no_dot_in_broad_range(self):
        # Without confidences, 12318 falls in the broad [0, 100000]
        # band and is accepted as-is.
        self.assertEqual(validate_instability("12318"), 12318.0)

    def test_decimal_recovery_with_low_confidence(self):
        # raw='12318' — char at index 2 is a low-confidence misread of
        # the decimal point. Recovery REPLACES the char with a dot,
        # yielding '12.18' which lands in [0, 200] and is preferred
        # over the broad-band 12318.0.
        confidences = [0.95, 0.95, 0.30, 0.95, 0.95]
        result = validate_instability("12318", confidences=confidences)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 12.18, places=2)

    def test_decimal_recovery_skipped_when_all_confident(self):
        # All chars have high confidence — no recovery, accept the
        # broad-band value.
        confidences = [0.95, 0.95, 0.95, 0.95, 0.95]
        result = validate_instability("12318", confidences=confidences)
        self.assertEqual(result, 12318.0)

    def test_empty_string(self):
        self.assertIsNone(validate_instability(""))

    def test_above_broad_max_returns_none(self):
        # 200000 exceeds even the broad [0, 100000] band
        self.assertIsNone(validate_instability("200000"))

    def test_dot_already_present_skips_recovery(self):
        # When the raw already has a dot, recovery shouldn't re-fire
        # — accept as-is.
        confidences = [0.95, 0.30, 0.95, 0.95]
        self.assertEqual(
            validate_instability("12.5", confidences=confidences), 12.5,
        )


class TestValidateRefineryCost(unittest.TestCase):

    def test_typical_value(self):
        self.assertEqual(validate_refinery_cost("12345"), 12345.0)

    def test_with_thousands_commas(self):
        # Commas stripped before float() — "12,345.67" → 12345.67
        self.assertEqual(validate_refinery_cost("12,345.67"), 12345.67)

    def test_with_currency_label(self):
        self.assertEqual(validate_refinery_cost("Cost: 1,500 aUEC"), 1500.0)

    def test_empty_returns_none(self):
        self.assertIsNone(validate_refinery_cost(""))

    def test_no_digits_returns_none(self):
        self.assertIsNone(validate_refinery_cost("aUEC"))

    def test_below_min_returns_none(self):
        # 0 is below the min (1.0)
        self.assertIsNone(validate_refinery_cost("0"))

    def test_above_max_returns_none(self):
        # 1e10 exceeds the 1e9 cap
        self.assertIsNone(validate_refinery_cost("9999999999"))

    def test_first_number_wins(self):
        # Multiple numbers — the regex picks the first one
        result = validate_refinery_cost("Pay 250 aUEC, deposit 50 aUEC")
        self.assertEqual(result, 250.0)


if __name__ == "__main__":
    unittest.main()
