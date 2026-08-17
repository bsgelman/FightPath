"""Tests for scripts/_reindex_vault.py pure functions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from _reindex_vault import (
    parse_note, collect_notes, collect_bmad_docs, build_reverse_index, render_index,
)


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_parse_note_full_frontmatter(tmp_path):
    note = tmp_path / "docs" / "notes" / "example.md"
    _write(note, """---
type: research
date: 2026-07-01
model_version: v8.38
status: active
related_code_files: [src/ufc/training/train_all.py]
related_runs_entry: "RUNS.md#v838"
tags: [duration]
---

# Example note
""")
    result = parse_note(note, tmp_path)
    assert result["type"] == "research"
    assert result["date"] == "2026-07-01"
    assert result["model_version"] == "v8.38"
    assert result["status"] == "active"
    assert result["related_code_files"] == ["src/ufc/training/train_all.py"]
    assert result["related_runs_entry"] == "RUNS.md#v838"
    assert result["tags"] == ["duration"]
    assert result["_warnings"] == []
    assert result["_path"] == "docs/notes/example.md"


def test_parse_note_missing_field(tmp_path):
    note = tmp_path / "docs" / "notes" / "partial.md"
    _write(note, """---
type: research
date: 2026-07-01
---

# Partial note
""")
    result = parse_note(note, tmp_path)
    assert "missing field: status" in result["_warnings"]


def test_parse_note_malformed_yaml(tmp_path):
    note = tmp_path / "docs" / "notes" / "broken.md"
    _write(note, """---
type: [unclosed
---

# Broken note
""")
    result = parse_note(note, tmp_path)
    assert result["_warnings"] == ["invalid YAML frontmatter"]
    assert result["type"] == "unknown"


def test_parse_note_no_frontmatter(tmp_path):
    note = tmp_path / "docs" / "notes" / "nofm.md"
    _write(note, "# Just a heading\n\nSome text.\n")
    result = parse_note(note, tmp_path)
    assert result["_warnings"] == ["no frontmatter found"]


def test_collect_notes_sorted(tmp_path):
    notes_dir = tmp_path / "docs" / "notes"
    _write(notes_dir / "b.md", "---\ntype: research\ndate: 2026-07-01\nstatus: active\n---\n")
    _write(notes_dir / "a.md", "---\ntype: research\ndate: 2026-07-01\nstatus: active\n---\n")
    notes = collect_notes(notes_dir, tmp_path)
    assert [n["_path"] for n in notes] == ["docs/notes/a.md", "docs/notes/b.md"]


def test_collect_notes_missing_dir(tmp_path):
    notes = collect_notes(tmp_path / "docs" / "notes", tmp_path)
    assert notes == []


def test_collect_bmad_docs(tmp_path):
    bmad_dir = tmp_path / "_bmad-output"
    _write(bmad_dir / "planning-artifacts" / "prd.md", "# PRD\n")
    docs = collect_bmad_docs(bmad_dir, tmp_path)
    assert docs == ["_bmad-output/planning-artifacts/prd.md"]


def test_collect_bmad_docs_missing_dir(tmp_path):
    docs = collect_bmad_docs(tmp_path / "_bmad-output", tmp_path)
    assert docs == []


def test_build_reverse_index():
    notes = [
        {"_path": "docs/notes/a.md", "related_code_files": ["src/ufc/models/winner.py"]},
        {"_path": "docs/notes/b.md", "related_code_files": ["src/ufc/models/winner.py", "src/ufc/api/app.py"]},
    ]
    reverse = build_reverse_index(notes)
    assert reverse == {
        "src/ufc/api/app.py": ["docs/notes/b.md"],
        "src/ufc/models/winner.py": ["docs/notes/a.md", "docs/notes/b.md"],
    }


def test_render_index_empty():
    content = render_index([], [], {}, [], [])
    assert "No notes yet" in content
    assert "No BMAD artifacts found" in content
    assert content.endswith("\n")
    assert not content.endswith("\n\n")


def test_render_index_with_content():
    notes = [{
        "_path": "docs/notes/a.md", "type": "research", "model_version": "v8.38",
        "_warnings": [],
    }]
    bmad_docs = ["_bmad-output/planning-artifacts/prd.md"]
    reverse_index = {"src/ufc/models/winner.py": ["docs/notes/a.md"]}
    content = render_index(notes, bmad_docs, reverse_index, [], [])
    assert "docs/notes/a.md" in content
    assert "_bmad-output/planning-artifacts/prd.md" in content
    assert "src/ufc/models/winner.py" in content


def test_parse_note_defaults_are_independent(tmp_path):
    """Verify two notes using defaults get independent list objects."""
    note_a = tmp_path / "docs" / "notes" / "a.md"
    note_b = tmp_path / "docs" / "notes" / "b.md"
    _write(note_a, "---\ntype: research\ndate: 2026-07-01\nstatus: active\n---\n")
    _write(note_b, "---\ntype: research\ndate: 2026-07-01\nstatus: active\n---\n")
    result_a = parse_note(note_a, tmp_path)
    result_b = parse_note(note_b, tmp_path)
    # Both should have default empty lists
    assert result_a["related_code_files"] == []
    assert result_b["related_code_files"] == []
    assert result_a["tags"] == []
    assert result_b["tags"] == []
    # But they should be independent objects, not shared references
    assert result_a["related_code_files"] is not result_b["related_code_files"]
    assert result_a["tags"] is not result_b["tags"]
    # Mutating one should not affect the other
    result_a["related_code_files"].append("mutated")
    assert result_b["related_code_files"] == []
    result_a["tags"].append("tag")
    assert result_b["tags"] == []


def test_render_index_version_sort_numeric():
    """Verify model_version strings sort numerically, not lexicographically.

    Tests that v8.2, v8.9, v8.10 appear in numeric order (8.2 < 8.9 < 8.10),
    and that 'n/a' appears last.
    """
    notes = [
        {
            "_path": "docs/notes/v8_9.md",
            "type": "research",
            "model_version": "v8.9",
            "_warnings": [],
        },
        {
            "_path": "docs/notes/v8_10.md",
            "type": "research",
            "model_version": "v8.10",
            "_warnings": [],
        },
        {
            "_path": "docs/notes/v8_2.md",
            "type": "research",
            "model_version": "v8.2",
            "_warnings": [],
        },
        {
            "_path": "docs/notes/no_version.md",
            "type": "research",
            "model_version": "n/a",
            "_warnings": [],
        },
    ]
    content = render_index(notes, [], {}, [], [])
    # Verify all notes are in the output
    assert "docs/notes/v8_9.md" in content
    assert "docs/notes/v8_10.md" in content
    assert "docs/notes/v8_2.md" in content
    assert "docs/notes/no_version.md" in content
    # Verify numeric sort order: v8.2 before v8.9 before v8.10
    idx_v8_2 = content.index("**v8.2**")
    idx_v8_9 = content.index("**v8.9**")
    idx_v8_10 = content.index("**v8.10**")
    idx_n_a = content.index("**n/a**")
    assert idx_v8_2 < idx_v8_9, "v8.2 should appear before v8.9"
    assert idx_v8_9 < idx_v8_10, "v8.9 should appear before v8.10"
    assert idx_v8_10 < idx_n_a, "v8.10 should appear before n/a"
