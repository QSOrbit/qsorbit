"""Phase 3's standing rule, enforced across the whole UI package.

    *No widget hardcodes a colour, ever. All styling flows from theme
    tokens; the waterfall colormap is part of the theme. If a colour
    literal shows up in UI code, that's a defect.*

This test exists because the rule was broken on the first pass and the
check used to find it was too narrow to notice. Both spectrum panels
drew the DC marker with ``Qt.GlobalColor.yellow``; the search that
"proved" the package was clean matched ``Qt.yellow`` and never saw it.
It reached a real screen as a bright yellow line that stayed yellow in
Night Ops -- the one theme whose entire purpose is that nothing on
screen costs the operator their dark adaptation.

A grep run once by hand proves nothing about the next commit. This does,
and it is deliberately broader than the literals that got through:
``QColor(...)``, ``Qt.GlobalColor.*``, bare ``Qt.<colour>``, hex
literals, ``rgb()``/``rgba()``, and inline ``setStyleSheet`` calls.

**Comments and strings are stripped before matching**, via ``tokenize``
rather than a regex, so a docstring may name the literal it is warning
against -- which the two DC-marker docstrings now do. A test that
punished a module for *explaining* the mistake would push the reasoning
out of the code, and this project deliberately keeps the argument
against a wrong version next to the right one.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

import pytest

import qsorbit.ui

#: The theme system itself is where colours are allowed to be named --
#: it is the module whose whole job is turning tokens into colours, and
#: it has its own test that every colour it emits came from a theme.
EXEMPT = {"theme.py", "theme_qss.py", "theme_manager.py"}

UI_DIR = Path(qsorbit.ui.__file__).parent

FORBIDDEN = re.compile(
    r"""
      QColor\s*\(                                   # QColor(0, 0, 0)
    | Qt\.GlobalColor\.\w+                          # Qt.GlobalColor.yellow
    | (?<![\w.])Qt\.(?:white|black|red|green|blue   # Qt.yellow
        |cyan|magenta|yellow|gray|darkGray
        |lightGray|transparent)\b
    | \#[0-9a-fA-F]{3,8}\b                          # #ff0 / #ffcc00
    | \brgba?\s*\(                                  # rgb(...) / rgba(...)
    | setStyleSheet\s*\(                            # inline stylesheets
    """,
    re.VERBOSE,
)


def ui_modules() -> list[Path]:
    """Every module under ``ui/``, at any depth.

    ``rglob`` rather than ``glob``, closed in Chunk C PR2 before anyone
    fell in: the check walked only the top level, so the first ``ui/``
    subpackage anybody added -- and PR3's Custom tab plus Chunk D's map
    are both plausible candidates -- would have been silently exempt
    from the standing rule. A hole in a rule that nobody has reached yet
    is still a hole, and this one would have been found the way the last
    one was, by a yellow line on a real screen in Night Ops.
    """
    return sorted(p for p in UI_DIR.rglob("*.py") if p.name not in EXEMPT)


def code_only(source: str) -> str:
    """The source with comments and *docstrings* removed -- not all strings.

    The distinction is the whole point. A docstring naming
    ``Qt.GlobalColor.yellow`` is prose about a mistake and must not
    trip the check; a string *passed to something* is code, and
    ``setStyleSheet("color: #ff0000")`` is the likeliest shape a
    hardcoded colour actually takes in Qt. An earlier version of this
    helper stripped every string and was therefore blind to exactly the
    case it most needed to catch -- which is the same failure as the
    grep this file replaces, one level further in.
    """
    tree = ast.parse(source)
    docstrings: set[tuple[int, int]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                docstrings.add((first.value.lineno, first.value.col_offset))

    lines = source.splitlines(keepends=True)
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        is_comment = token.type == tokenize.COMMENT
        is_docstring = token.type == tokenize.STRING and token.start in docstrings
        if not (is_comment or is_docstring):
            continue
        (row0, col0), (row1, col1) = token.start, token.end
        for row in range(row0, row1 + 1):
            line = lines[row - 1]
            start = col0 if row == row0 else 0
            end = col1 if row == row1 else len(line)
            lines[row - 1] = line[:start] + " " * (end - start) + line[end:]
    return "".join(lines)


def test_there_are_ui_modules_to_check():
    """A loop over an empty list passes and proves nothing."""
    names = [p.name for p in ui_modules()]
    assert len(names) > 5
    assert "waterfall_widget.py" in names
    assert "spectrum_line_widget.py" in names


@pytest.mark.parametrize("module", ui_modules(), ids=lambda p: p.name)
def test_no_widget_hardcodes_a_colour(module: Path):
    source = module.read_text(encoding="utf-8")
    offenders = sorted(set(FORBIDDEN.findall(code_only(source))))
    assert not offenders, (
        f"{module.name} contains colour literal(s): {offenders}. "
        f"Every colour must come from the active theme's palette - see "
        f"qsorbit.ui.theme.Palette."
    )


def test_the_exempt_modules_are_the_theme_system_and_nothing_else():
    """An exemption list is a hole in the rule, so it is asserted rather
    than trusted -- adding a module to it should be a visible decision."""
    assert EXEMPT == {"theme.py", "theme_qss.py", "theme_manager.py"}
    for name in EXEMPT:
        assert (UI_DIR / name).is_file()


def test_the_check_can_actually_fail():
    """Canary. A verifier that cannot fail is not a verifier.

    The real defect this file was written for was invisible to the
    previous check, so this one asserts it catches each shape it claims
    to catch rather than being taken on faith.
    """
    assert FORBIDDEN.search(code_only("painter.setPen(Qt.GlobalColor.yellow)"))
    assert FORBIDDEN.search(code_only("painter.setPen(Qt.red)"))
    assert FORBIDDEN.search(code_only("pen = QColor(255, 0, 0)"))
    assert FORBIDDEN.search(code_only("w.setStyleSheet(sheet)"))
    assert FORBIDDEN.search(code_only('w.setStyleSheet("color: #ff0000")'))
    assert FORBIDDEN.search(code_only('pen = QColor("#ffcc00")'))
    # And that it does not fire on prose about the mistake.
    assert not FORBIDDEN.search(code_only('"""It used to be Qt.GlobalColor.yellow."""'))
    assert not FORBIDDEN.search(code_only("# was #ffcc00 before the theme pass"))
    assert not FORBIDDEN.search(code_only('def f():\n    """Uses #ffcc00."""\n    return 1\n'))
    # Nor on the legitimate way to get a colour.
    assert not FORBIDDEN.search(code_only("painter.setPen(self.palette().windowText().color())"))
