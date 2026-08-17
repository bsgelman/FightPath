"""Direct coverage for the extracted name matchers.

Before extraction these lived in prop_lines.py and had no direct tests — they were
exercised only incidentally through resolve_to_card and find_fighter. Each case
below pins a regression named in the original docstrings.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ufc.ingest.name_norm import normalize, token_surname_match


class TestNormalize:
    def test_strips_diacritics(self):
        assert normalize("José Aldó") == "jose aldo"

    def test_punctuation_becomes_space_not_deleted(self):
        """Compound surnames must tokenize the same whether the source uses a
        hyphen or a space, instead of fusing into one unmatchable token."""
        assert normalize("Saint-Denis") == "saint denis"
        assert normalize("Saint Denis") == "saint denis"

    def test_collapses_whitespace_and_lowercases(self):
        assert normalize("  Islam   MAKHACHEV ") == "islam makhachev"

    def test_apostrophes_split_rather_than_fuse(self):
        assert normalize("O'Malley") == "o malley"


class TestTokenSurnameMatch:
    def test_lima_substring_bug_does_not_phantom_match(self):
        """The documented bug: a raw substring test matched 'Murtazali Magomedov'
        against 'Andre Lima'. Surnames must match as WHOLE tokens."""
        assert not token_surname_match(normalize("Murtazali Magomedov"),
                                       normalize("Andre Lima"))

    def test_exact_surname_and_first_name_matches(self):
        assert token_surname_match(normalize("Islam Makhachev"),
                                   normalize("Islam Makhachev"))

    def test_first_name_abbreviation_is_compatible(self):
        """'Sharabutdin' <-> 'Shara' — one a prefix of the other."""
        assert token_surname_match(normalize("Sharabutdin Magomedov"),
                                   normalize("Shara Magomedov"))

    def test_single_initial_is_compatible(self):
        assert token_surname_match(normalize("I Makhachev"),
                                   normalize("Islam Makhachev"))

    def test_different_first_name_same_surname_rejected(self):
        """The Garza->Glaza class of bug: same surname is not enough."""
        assert not token_surname_match(normalize("John Smith"),
                                       normalize("Michael Smith"))

    def test_short_surnames_rejected(self):
        """Surnames under 3 chars are too weak to match on."""
        assert not token_surname_match(normalize("Jon Li"), normalize("Bob Li"))

    def test_empty_input_is_false_not_error(self):
        assert not token_surname_match("", normalize("Islam Makhachev"))
        assert not token_surname_match(normalize("Islam Makhachev"), "")

    def test_dropped_first_name_is_rejected(self):
        """Documents a deliberate conservative limitation: when a source omits the
        first name, the surnames still agree ('garry') but the leading tokens
        ('machado' vs 'ian') are incompatible, so the match is refused. Erring
        toward a miss is correct here — a false match silently prices the wrong
        fighter, which is the failure mode this guard exists to prevent."""
        assert not token_surname_match(normalize("Machado Garry"),
                                       normalize("Ian Machado Garry"))

    def test_differing_middle_token_still_matches_on_first_and_surname(self):
        assert token_surname_match(normalize("Ian Garry"),
                                   normalize("Ian Machado Garry"))
