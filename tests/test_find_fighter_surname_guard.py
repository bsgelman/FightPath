"""find_fighter fuzzy-match guard: reject difflib matches that aren't the same surname."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd
import pytest

from ufc.inference.predict_core import load_reference_data
from ufc.inference.matchup import find_fighter


@pytest.fixture(scope="module")
def fighters_df():
# Needs the built parquet dataset, which is not distributed with the repo
# (see DATA.md). Skips cleanly on a fresh clone; runs whenever data is present.
    try:
        fighters_df, _, _ = load_reference_data()
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"reference data unavailable: {exc}")
    return fighters_df


def test_suffix_still_matches(fighters_df):
    """A suffixed query still resolves. The canonical name now carries the suffix
    itself (roster names supersede the append-only tott source), so assert on a
    successful resolution rather than the pre-rename spelling."""
    fid, name = find_fighter("Kai Kamaka III", fighters_df)
    assert fid and name.startswith("Kai Kamaka")


def test_first_name_expansion_still_matches(fighters_df):
    """Abbreviated first name resolves to the same fighter regardless of which
    spelling the roster currently carries."""
    fid, name = find_fighter("Zach Reese", fighters_df)
    assert fid and name.endswith("Reese")


def test_different_fighter_raises():
    # The live-bug pair (2026-07-11): 'John Garza' (not yet in the roster) fuzzy-matched
    # 'Jason Glaza'. Synthetic df so the case stays valid after Garza joins the roster.
    df = pd.DataFrame({"fighter_id": ["x1"], "fighter_name": ["Jason Glaza"]})
    with pytest.raises(ValueError):
        find_fighter("John Garza", df)
