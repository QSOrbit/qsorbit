"""Turning a :class:`~qsorbit.ui.theme.Theme` into a Qt stylesheet.

No Qt import in here either -- a stylesheet is a string, and generating
it is string formatting over theme tokens. Keeping it Qt-free means the
whole of "what does this theme actually style" is testable without
PySide6, and the assertion that matters most is a plain substring check:
**no colour appears in the output that is not one of the theme's own
tokens.**

Styling reaches widgets through *two* channels, and both are needed.

1. **This stylesheet**, which handles standard Qt widgets -- tabs,
   buttons, combo boxes, labels, scroll bars. Qt applies it from the
   application down.
2. **A QPalette**, built by :mod:`qsorbit.ui.theme_manager` from
   :func:`palette_roles` below. Custom-painted widgets already read
   ``self.palette().windowText().color()`` rather than naming a colour
   -- ``spectrum_axis_paint`` has said so in its docstring since Chunk I
   -- and a stylesheet does not reliably update QPalette. Setting only
   one of the two leaves half the screen unthemed, which is exactly the
   bug this module's tests exist to catch.

**Chrome has a part this module cannot reach.** ``style = "lcars"``
wants accent bars down the left edge of every card, and ``style =
"crt"`` wants scanlines over the whole window; the mockup draws both
with CSS pseudo-elements, and QSS has no equivalent. Those are child
widgets and a painted overlay respectively, and they belong to the
shell. What lives here is everything chrome *can* express as style --
border treatment, radii, typography, letter spacing -- plus
:func:`chrome_structure`, which tells the shell which structural pieces
this theme wants. The split is: this module decides how things look,
the shell decides what things exist.
"""

from __future__ import annotations

from dataclasses import dataclass

from qsorbit.ui.theme import Theme

#: Fallback monospace stack when a theme's ``[chrome]`` names none.
#: Every readout, frequency and axis label in the app is monospace --
#: digits that change width as they count are unreadable at a glance,
#: which matters more here than in most UIs because these numbers move
#: continuously during a pass.
DEFAULT_MONO_STACK: tuple[str, ...] = (
    "Cascadia Code",
    "JetBrains Mono",
    "Consolas",
    "DejaVu Sans Mono",
    "monospace",
)


@dataclass(frozen=True)
class ChromeStructure:
    """Which structural pieces the shell should build for a chrome style.

    Returned by :func:`chrome_structure`. Everything here is something
    QSS cannot do, so the shell reads this rather than parsing the
    stylesheet.

    Args:
        accent_bars: Draw a coloured bar down the leading edge of every
            card, LCARS-style, instead of a line border.
        scanlines: Overlay horizontal scanlines across the whole window.
        glow: Apply a soft text glow to headline elements -- the logo,
            frequency readouts, the selected tab.
        uppercase_headings: Render card headings and tab labels in caps.
    """

    accent_bars: bool = False
    scanlines: bool = False
    glow: bool = False
    uppercase_headings: bool = False


def chrome_structure(theme: Theme) -> ChromeStructure:
    """The structural chrome pieces ``theme`` asks the shell to build."""
    if theme.chrome.style == "lcars":
        return ChromeStructure(accent_bars=True, uppercase_headings=True)
    if theme.chrome.style == "crt":
        return ChromeStructure(scanlines=True, glow=True, uppercase_headings=True)
    return ChromeStructure()


def palette_roles(theme: Theme) -> dict[str, str]:
    """Map Qt ``QPalette`` role names to this theme's colours.

    Keys are :class:`QPalette.ColorRole` member names, so
    :mod:`qsorbit.ui.theme_manager` can apply them by ``getattr`` without
    this module importing Qt. The roles chosen are the ones QSOrbit's
    custom-painted widgets actually read: ``WindowText`` for axis ticks
    and labels, ``Window`` and ``Base`` for backgrounds, ``Highlight``
    for the tracked-frequency marker.
    """
    p = theme.palette
    return {
        "Window": p.bg,
        "WindowText": p.text,
        "Base": p.inset,
        "AlternateBase": p.panel_alt,
        "Text": p.text,
        "Button": p.panel_alt,
        "ButtonText": p.text,
        "PlaceholderText": p.dim,
        "Highlight": p.accent,
        "HighlightedText": p.bg,
        "ToolTipBase": p.panel_alt,
        "ToolTipText": p.text,
        "Link": p.accent,
    }


def mono_families(theme: Theme) -> tuple[str, ...]:
    """The monospace family stack for ``theme``, most preferred first."""
    if theme.chrome.mono:
        return (theme.chrome.mono, *DEFAULT_MONO_STACK)
    return DEFAULT_MONO_STACK


def _mono_css(theme: Theme) -> str:
    return ", ".join(f'"{family}"' for family in mono_families(theme))


def _ui_css(theme: Theme) -> str:
    if theme.chrome.font:
        return f'"{theme.chrome.font}", "Segoe UI", system-ui, sans-serif'
    return '"Segoe UI", system-ui, sans-serif'


def build_stylesheet(theme: Theme) -> str:
    """Build the application-wide Qt stylesheet for ``theme``.

    Every colour in the result comes from ``theme.palette``; nothing in
    here is a literal. That is the standing Phase 3 rule -- no widget
    hardcodes a colour, ever -- applied to the one module whose entire
    job is emitting colours, and it is checked by a test rather than
    left to review. **Including the comments in the emitted CSS**, which
    is worth knowing before writing one: that test does not strip them,
    correctly, because a comment inside a generated stylesheet is
    shipped text rather than source prose. An explanatory comment naming
    two hex values got caught by it during Chunk C PR2, which is the
    check working exactly as intended.

    **``QWidget`` sets typography and foreground colour but deliberately
    no background**, and that one omission is what makes a card visible.
    A universal ``background`` rule reaches every *nested* QWidget as
    well, so a card painted in ``panel`` was immediately painted over in
    ``bg`` by its own children -- measured rather than guessed, by
    sampling a rendered card interior and finding the background token
    where the panel token should have been. A plain QWidget paints no
    background of its own unless told to, so naming the background only
    where it is wanted (the window, cards, insets, controls) leaves
    ordinary containers transparent and lets the card show through.
    Nothing that predates cards changes appearance: those children were
    painting the window's own colour anyway.
    """
    p = theme.palette
    structure = chrome_structure(theme)
    mono = _mono_css(theme)
    ui = _ui_css(theme)

    if theme.chrome.style == "lcars":
        radius, card_radius, tab_radius = "0px", "0px 16px 16px 0px", "13px"
        card_border = "none"
        tab_rules = f"""
QTabBar::tab {{
    background: {p.panel_alt};
    color: {p.text};
    border: none;
    border-radius: {tab_radius};
    padding: 7px 20px;
    margin: 3px 4px 3px 0px;
    font-weight: 600;
}}
QTabBar::tab:selected {{ background: {p.accent}; color: {p.bg}; }}
QTabBar::tab:hover:!selected {{ background: {p.edge}; }}
"""
    elif theme.chrome.style == "crt":
        radius, card_radius, tab_radius = "0px", "0px", "0px"
        card_border = f"1px solid {p.edge}"
        tab_rules = f"""
QTabBar::tab {{
    background: transparent;
    color: {p.dim};
    border: none;
    border-bottom: 2px solid transparent;
    padding: 9px 22px;
    letter-spacing: 2px;
}}
QTabBar::tab:selected {{ color: {p.accent}; border-bottom-color: {p.accent}; }}
QTabBar::tab:hover:!selected {{ color: {p.text}; }}
"""
    else:
        radius, card_radius, tab_radius = "4px", "6px", "0px"
        card_border = f"1px solid {p.edge}"
        tab_rules = f"""
QTabBar::tab {{
    background: transparent;
    color: {p.dim};
    border: none;
    border-bottom: 2px solid transparent;
    padding: 9px 22px;
}}
QTabBar::tab:selected {{ color: {p.accent}; border-bottom-color: {p.accent}; }}
QTabBar::tab:hover:!selected {{ color: {p.text}; }}
"""

    heading_transform = "uppercase" if structure.uppercase_headings else "none"
    # LCARS puts its card headings in a filled pill; every other chrome
    # leaves them as small dim caps.
    if theme.chrome.style == "lcars":
        heading_rule = f"""
QLabel[role="heading"] {{
    background: {p.warn};
    color: {p.bg};
    border-radius: 11px;
    padding: 2px 14px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}}
"""
    else:
        heading_rule = f"""
QLabel[role="heading"] {{
    color: {p.dim};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: {heading_transform};
}}
"""

    return (
        f"""
/* {theme.name} -- generated from theme tokens; no literal colours. */

/* Typography and colour apply to everything; the background does not.
   See build_stylesheet's docstring for why -- a universal background
   rule reaches every nested widget and paints over the card it sits in. */
QWidget {{
    color: {p.text};
    font-family: {ui};
    font-size: 13px;
}}

QMainWindow, QDialog {{
    background: {p.bg};
}}

QWidget#Card {{
    background: {p.panel};
    border: {card_border};
    border-radius: {card_radius};
}}

QWidget#Inset, QAbstractScrollArea, QPlainTextEdit, QTextEdit {{
    background: {p.inset};
    border: 1px solid {p.edge};
    border-radius: {radius};
}}
{heading_rule}
QLabel {{ background: transparent; }}
QLabel[role="dim"] {{ color: {p.dim}; }}
QLabel[role="readout"] {{ font-family: {mono}; font-size: 30px; color: {p.text}; }}
QLabel[role="value"] {{ font-family: {mono}; color: {p.text}; }}
QLabel[role="accent"] {{ color: {p.accent}; }}
QLabel[role="ok"] {{ color: {p.ok}; }}
QLabel[role="warn"] {{ color: {p.warn}; }}
QLabel[role="alarm"] {{ color: {p.alarm}; }}
QLabel[role="ghost"] {{ color: {p.dim}; font-style: italic; font-size: 11px; }}

QTabWidget::pane {{ border: none; background: {p.bg}; }}
QTabBar {{ background: {p.panel}; }}
{tab_rules}
QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: {p.panel_alt};
    color: {p.text};
    border: 1px solid {p.edge};
    border-radius: {radius};
    padding: 4px 10px;
}}
QPushButton:hover, QComboBox:hover {{ border-color: {p.accent}; }}
QPushButton:pressed {{ background: {p.edge}; }}
QPushButton:disabled, QComboBox:disabled, QLabel:disabled {{ color: {p.dim}; }}
QComboBox QAbstractItemView {{
    background: {p.panel_alt};
    color: {p.text};
    border: 1px solid {p.edge};
    selection-background-color: {p.accent};
    selection-color: {p.bg};
}}

QProgressBar {{
    background: {p.inset};
    border: 1px solid {p.edge};
    border-radius: {radius};
    text-align: center;
    color: {p.text};
}}
QProgressBar::chunk {{ background: {p.ok}; border-radius: {radius}; }}

QGroupBox {{
    background: {p.panel};
    border: {card_border};
    border-radius: {card_radius};
    margin-top: 10px;
    padding-top: 8px;
}}
QGroupBox::title {{ color: {p.dim}; subcontrol-origin: margin; left: 10px; }}

QHeaderView::section {{
    background: {p.panel};
    color: {p.dim};
    border: none;
    border-bottom: 1px solid {p.edge};
    padding: 6px 10px;
}}
QTableView, QTreeView, QListView {{
    background: {p.panel};
    alternate-background-color: {p.panel_alt};
    color: {p.text};
    gridline-color: {p.edge};
    selection-background-color: {p.accent};
    selection-color: {p.bg};
    border: {card_border};
    border-radius: {card_radius};
}}

QScrollBar:vertical, QScrollBar:horizontal {{ background: {p.bg}; border: none; }}
QScrollBar::handle {{ background: {p.edge}; border-radius: {radius}; min-height: 24px; }}
QScrollBar::handle:hover {{ background: {p.dim}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0px; width: 0px; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QToolTip {{
    background: {p.panel_alt};
    color: {p.text};
    border: 1px solid {p.edge};
    padding: 4px;
}}

QSplitter::handle {{ background: {p.edge}; }}
QStatusBar {{ background: {p.panel}; color: {p.dim}; }}
QMenuBar, QMenu {{ background: {p.panel}; color: {p.text}; }}
QMenu::item:selected, QMenuBar::item:selected {{ background: {p.accent}; color: {p.bg}; }}
""".strip()
        + "\n"
    )
