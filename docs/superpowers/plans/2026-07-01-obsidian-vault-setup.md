# Obsidian Knowledge Vault Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up `docs/` as an Obsidian-browsable knowledge vault — frontmatter template, a
periodically-regenerated `Master_Index.md`, and one real note proving the pipeline — per
`docs/superpowers/specs/2026-07-01-obsidian-vault-design.md`.

**Architecture:** A single self-contained utility script (`scripts/_reindex_vault.py`, following
the repo's existing `_name.py` = manual-utility convention) parses YAML frontmatter from
`docs/notes/**/*.md` with `yaml.safe_load` (regex-delimited, no new dependency), scans
`_bmad-output/**/*.md` for a separate unvalidated section, and writes a deterministic
`docs/Master_Index.md`. Pure functions are unit-tested via the repo's existing test convention
(`tests/test_*.py` with `sys.path.insert` — mirrored here against `scripts/` instead of `src/`).

**Tech Stack:** Python 3.12, `pyyaml` (already a dependency — see `requirements.txt`), `pytest`
(already a dependency), no new packages.

## Global Constraints

- Preserve all existing import paths, module dependencies, and Git history — no unrelated changes.
- `RUNS.md` remains the sole source of truth for run history; nothing in this plan duplicates it.
- `architecture_spec.md` is not touched or restructured.
- No retroactive frontmatter added to existing files (`architecture_spec.md`, `RUNS.md`,
  `outputs/reports/*.md`) — going-forward convention only, per approved spec Decision 3.
- No comment pointers injected into `src/` files — reverse linking is index-level only
  (`Master_Index.md`'s "Code → notes" section), per spec's explicit non-goal.
- No CI hook, no Gate, no wiring into `dev.bat` — `scripts/_reindex_vault.py` is manual/periodic
  only, matching the existing `_prod_calibration_report.py` / `_joint_coherence_check.py` pattern.
- Follow the existing `scripts/` naming convention: utility scripts are `_name.py`, not numbered.
- Follow the existing `tests/` convention: flat pytest functions, `sys.path.insert` for imports,
  no test classes/fixtures beyond pytest's built-in `tmp_path`.

---

### Task 1: Core reindex logic + tests

**Files:**
- Create: `scripts/_reindex_vault.py`
- Test: `tests/test_reindex_vault.py`

**Interfaces:**
- Produces: `parse_note(path: Path, repo_root: Path) -> dict` — keys `_path`, `_warnings`,
  `type`, `date`, `status`, `model_version`, `related_code_files`, `related_runs_entry`, `tags`.
- Produces: `collect_notes(notes_dir: Path, repo_root: Path) -> list[dict]` — sorted by `_path`.
- Produces: `collect_bmad_docs(bmad_dir: Path, repo_root: Path) -> list[str]` — sorted relative paths.
- Produces: `build_reverse_index(notes: list[dict]) -> dict[str, list[str]]` — code path → sorted note paths.
- Produces: `render_index(notes: list[dict], bmad_docs: list[str], reverse_index: dict[str, list[str]]) -> str`.
- Produces: `main() -> None` — CLI entry point (used by Task 2/3's manual verification runs).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reindex_vault.py`:

```python
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
    content = render_index([], [], {})
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
    content = render_index(notes, bmad_docs, reverse_index)
    assert "docs/notes/a.md" in content
    assert "_bmad-output/planning-artifacts/prd.md" in content
    assert "src/ufc/models/winner.py" in content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_reindex_vault.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named '_reindex_vault'`

- [ ] **Step 3: Write the implementation**

Create `scripts/_reindex_vault.py`:

```python
"""Regenerate docs/Master_Index.md from docs/notes/ frontmatter + _bmad-output/ artifacts.

Manual/periodic utility (not a Gate, not wired into dev.bat or CI). Run after adding or
editing vault notes:

    python scripts/_reindex_vault.py
"""
import re
from pathlib import Path

import yaml

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)
REQUIRED_FIELDS = ["type", "date", "status"]
TYPE_ORDER = ["research", "decision", "evidence", "reference"]

_DEFAULTS = {
    "type": "unknown",
    "date": "",
    "status": "unknown",
    "model_version": "n/a",
    "related_code_files": [],
    "related_runs_entry": "",
    "tags": [],
}


def parse_note(path: Path, repo_root: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    rel_path = path.relative_to(repo_root).as_posix()
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {"_path": rel_path, "_warnings": ["no frontmatter found"], **_DEFAULTS}

    try:
        data = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return {"_path": rel_path, "_warnings": ["invalid YAML frontmatter"], **_DEFAULTS}

    warnings = [f"missing field: {f}" for f in REQUIRED_FIELDS if not data.get(f)]
    result = {"_path": rel_path, "_warnings": warnings}
    for key, default in _DEFAULTS.items():
        value = data.get(key, default)
        result[key] = str(value) if key == "date" and value else (value if value else default)
    return result


def collect_notes(notes_dir: Path, repo_root: Path) -> list:
    paths = sorted(notes_dir.glob("**/*.md"))
    return [parse_note(p, repo_root) for p in paths]


def collect_bmad_docs(bmad_dir: Path, repo_root: Path) -> list:
    paths = sorted(bmad_dir.glob("**/*.md"))
    return [p.relative_to(repo_root).as_posix() for p in paths]


def build_reverse_index(notes: list) -> dict:
    reverse: dict = {}
    for note in notes:
        for code_path in note.get("related_code_files", []):
            reverse.setdefault(code_path, []).append(note["_path"])
    return {k: sorted(v) for k, v in sorted(reverse.items())}


def render_index(notes: list, bmad_docs: list, reverse_index: dict) -> str:
    lines = [
        "# Master Index",
        "",
        "Generated by `scripts/_reindex_vault.py` — do not hand-edit.",
        "",
        "## Notes by type",
        "",
    ]

    if not notes:
        lines.append("_No notes yet — add markdown files under `docs/notes/`._")
        lines.append("")
    else:
        types_present = sorted(
            {n["type"] for n in notes},
            key=lambda t: (TYPE_ORDER.index(t) if t in TYPE_ORDER else len(TYPE_ORDER), t),
        )
        for t in types_present:
            lines.append(f"### {t}")
            lines.append("")
            group = [n for n in notes if n["type"] == t]
            versions = sorted({n["model_version"] for n in group})
            for v in versions:
                lines.append(f"**{v}**")
                lines.append("")
                for n in sorted(group, key=lambda n: n["_path"]):
                    if n["model_version"] != v:
                        continue
                    warn = f" _(warnings: {', '.join(n['_warnings'])})_" if n["_warnings"] else ""
                    lines.append(f"- [{n['_path']}]({n['_path']}){warn}")
                lines.append("")

    lines.append("## Code -> notes (reverse lookup)")
    lines.append("")
    if not reverse_index:
        lines.append("_No notes reference `related_code_files` yet._")
        lines.append("")
    else:
        for code_path, note_paths in reverse_index.items():
            lines.append(f"- `{code_path}`")
            for np_ in note_paths:
                lines.append(f"  - [{np_}]({np_})")
        lines.append("")

    lines.append("## BMAD artifacts (`_bmad-output/`)")
    lines.append("")
    if not bmad_docs:
        lines.append("_No BMAD artifacts found._")
    else:
        for path in bmad_docs:
            lines.append(f"- [{path}]({path})")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notes_dir = repo_root / "docs" / "notes"
    bmad_dir = repo_root / "_bmad-output"
    index_path = repo_root / "docs" / "Master_Index.md"

    notes = collect_notes(notes_dir, repo_root)
    bmad_docs = collect_bmad_docs(bmad_dir, repo_root)
    reverse_index = build_reverse_index(notes)

    for note in notes:
        for w in note["_warnings"]:
            print(f"WARNING: {note['_path']}: {w}")

    index_path.write_text(render_index(notes, bmad_docs, reverse_index), encoding="utf-8")
    print(f"Wrote docs/Master_Index.md ({len(notes)} notes, {len(bmad_docs)} BMAD docs)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_reindex_vault.py -v`
Expected: PASS — 11 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/_reindex_vault.py tests/test_reindex_vault.py
git commit -m "feat(vault): add docs vault reindex script with tests"
```

---

### Task 2: Frontmatter template + first real run

**Files:**
- Create: `docs/templates/research-note-template.md`

**Interfaces:**
- Consumes: `main()` from Task 1 (`scripts/_reindex_vault.py`).

- [ ] **Step 1: Create the template**

Create `docs/templates/research-note-template.md`:

```markdown
---
type: research
date: {{date:YYYY-MM-DD}}
model_version: n/a
status: draft
related_code_files: []
related_runs_entry: ""
tags: []
---

# Title

## Summary

## Evidence

## Related
```

`{{date:YYYY-MM-DD}}` is Obsidian's native Templates-plugin placeholder syntax — it expands to
today's date when the template is inserted via the Templates plugin, and is otherwise inert
plain text (not consumed by `scripts/_reindex_vault.py`, which only parses committed notes
under `docs/notes/`).

- [ ] **Step 2: Run the reindexer against the real repo**

Run: `python scripts/_reindex_vault.py`

Expected output: `Wrote docs/Master_Index.md (0 notes, 0 BMAD docs)` — `docs/notes/` doesn't
exist yet and `_bmad-output/planning-artifacts|implementation-artifacts` are currently empty, so
both sections should render their "no content yet" placeholders. No warnings should print (there
are zero notes to warn about).

- [ ] **Step 3: Verify the generated file**

Read `docs/Master_Index.md` and confirm it contains `_No notes yet` and `_No BMAD artifacts found_`.

- [ ] **Step 4: Commit**

```bash
git add docs/templates/research-note-template.md docs/Master_Index.md
git commit -m "feat(vault): add research note template, first Master_Index.md generation"
```

---

### Task 3: First real vault note (proves the pipeline end-to-end)

**Files:**
- Create: `docs/notes/2026-07-01-obsidian-vault-setup.md`

**Interfaces:**
- Consumes: `main()` from Task 1.

- [ ] **Step 1: Write the note**

Create `docs/notes/2026-07-01-obsidian-vault-setup.md`:

```markdown
---
type: decision
date: 2026-07-01
model_version: n/a
status: active
related_code_files: [scripts/_reindex_vault.py, docs/templates/research-note-template.md]
related_runs_entry: ""
tags: [vault, documentation]
---

# Obsidian knowledge vault setup

## Summary

Established `docs/notes/` as the home for hand-written research/decision/evidence/reference
notes, with frontmatter-driven indexing into `docs/Master_Index.md`. Full rationale and the
approved design in `docs/superpowers/specs/2026-07-01-obsidian-vault-design.md`.

## Evidence

Not a model-behavior change — no `related_runs_entry`. `RUNS.md` remains the sole evidence log
for changes that move Gates A-D; this note documents tooling only.

## Related

- Design spec: `docs/superpowers/specs/2026-07-01-obsidian-vault-design.md`
- Implementation plan: `docs/superpowers/plans/2026-07-01-obsidian-vault-setup.md`
```

- [ ] **Step 2: Re-run the reindexer**

Run: `python scripts/_reindex_vault.py`
Expected output: `Wrote docs/Master_Index.md (1 notes, 0 BMAD docs)`, no warnings printed.

- [ ] **Step 3: Verify the generated index**

Read `docs/Master_Index.md` and confirm:
- A `### decision` section under `## Notes by type` lists `docs/notes/2026-07-01-obsidian-vault-setup.md`.
- The `## Code -> notes (reverse lookup)` section lists both
  `scripts/_reindex_vault.py` and `docs/templates/research-note-template.md`, each pointing back
  to the new note.

- [ ] **Step 4: Commit**

```bash
git add docs/notes/2026-07-01-obsidian-vault-setup.md docs/Master_Index.md
git commit -m "docs(vault): add first vault note documenting the vault setup itself"
```

---

### Task 4: Obsidian template wiring + CLAUDE.md steward convention

**Files:**
- Create: `.obsidian/templates.json`
- Modify: `CLAUDE.md`

**Interfaces:**
- None (terminal task, no code consumed/produced).

- [ ] **Step 1: Wire the Templates core plugin**

Create `.obsidian/templates.json`:

```json
{
  "folder": "docs/templates"
}
```

This points Obsidian's already-enabled Templates core plugin (confirmed on in
`.obsidian/core-plugins.json`) at the template folder from Task 2. Note: this file's exact
schema can't be verified without launching the Obsidian app — if it doesn't take effect, the
user can set the same folder manually via Settings → Core plugins → Templates → Template folder
location.

- [ ] **Step 2: Add the CLAUDE.md vault section**

Modify `CLAUDE.md` — add a new section after the existing "## Deploy" section (end of file):

```markdown

## Knowledge vault (Obsidian, /docs)
Research notes live in `docs/notes/` (frontmatter: type/date/model_version/status/
related_code_files/related_runs_entry/tags — template at `docs/templates/research-note-template.md`).
Index: `docs/Master_Index.md`, regenerate via `python scripts/_reindex_vault.py`.
RUNS.md remains the source of truth for run history; a vault note points at a RUNS.md entry via
`related_runs_entry` rather than duplicating it. Before a model-behavior change, check
Master_Index.md for prior evidence; if a proposed change lacks an evidence trail, ask before
proceeding.
```

- [ ] **Step 3: Verify**

Read `CLAUDE.md` and confirm the new section is present and the existing content above it is
byte-for-byte unchanged (diff should show a pure addition, no reformatting of existing lines).

- [ ] **Step 4: Commit**

```bash
git add .obsidian/templates.json CLAUDE.md
git commit -m "chore(vault): wire Obsidian templates plugin, document steward convention in CLAUDE.md"
```
