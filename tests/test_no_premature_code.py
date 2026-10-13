# tests/test_no_premature_code.py
"""
Enforces PRD 12.0 scope rule:
M1 only permitted after Gate M0a is passed (docs/gates/M0a.md exists and is complete).
M2 only permitted after Gate M0b is passed (docs/gates/M0b.md exists and is complete).
No premature shadow_agents or shadow_arena directories without respective gate documents.
"""

from pathlib import Path
import pytest


def test_no_premature_agents():
    root = Path(__file__).resolve().parent.parent
    agents_dir = root / "shadow_agents"
    gate_m0a = root / "docs" / "gates" / "M0a.md"

    if agents_dir.exists():
        assert gate_m0a.exists(), (
            "Premature code: shadow_agents/ exists but docs/gates/M0a.md does not!"
        )
        content = gate_m0a.read_text(encoding="utf-8")
        assert "GATE M0a: PASSED" in content, (
            "Premature code: shadow_agents/ exists before Gate M0a is marked PASSED!"
        )


def test_no_premature_arena():
    root = Path(__file__).resolve().parent.parent
    arena_dir = root / "shadow_arena"
    gate_m1 = root / "docs" / "gates" / "M1.md"

    if arena_dir.exists():
        assert gate_m1.exists(), (
            "Premature code: shadow_arena/ exists but docs/gates/M1.md does not!"
        )
        content = gate_m1.read_text(encoding="utf-8")
        assert "GATE M1: PASSED" in content, (
            "Premature code: shadow_arena/ exists before Gate M1 is marked PASSED!"
        )
