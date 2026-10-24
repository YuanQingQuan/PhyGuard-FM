from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = [ROOT / "experiments", ROOT / "scripts"]


FRAME_ITER = {"offsets", "offset", "frames", "frame", "off_list", "offs"}

SEED_CALLS = {
    ("torch", "manual_seed"),
    ("np", "random", "seed"),
    ("numpy", "random", "seed"),
}


def _dotted(node: ast.AST) -> tuple[str, ...]:

    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return tuple(reversed(parts))


def _iter_names(loop: ast.For) -> set[str]:

    it = loop.iter
    names = set()
    if isinstance(it, ast.Name):
        names.add(it.id)
    elif isinstance(it, (ast.Tuple, ast.List)):
        names.update(e.id for e in it.elts if isinstance(e, ast.Name))
    elif isinstance(it, ast.Call) and isinstance(it.func, ast.Name):
        names.add(it.func.id)
    return names


def _seed_calls(node: ast.AST) -> list[tuple[int, str]]:
    out = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and _dotted(sub.func) in SEED_CALLS:
            out.append((sub.lineno, ".".join(_dotted(sub.func))))
    return out


def _frame_loops(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and (_iter_names(node) & FRAME_ITER):
            yield node


@pytest.mark.parametrize("path", sorted(
    p for d in SCAN_DIRS if d.exists() for p in d.rglob("*.py")
), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_reseeding_inside_frame_loop(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = []
    for loop in _frame_loops(tree):
        for lineno, name in _seed_calls(loop):
            offenders.append(f"line {lineno}: {name}() inside a per-frame loop")
    assert not offenders, (
        f"{path.relative_to(ROOT)} reseeds the RNG per frame; hoist the seed "
        f"above the frame loop so frames are independent draws:\n  "
        + "\n  ".join(sorted(set(offenders)))
    )
