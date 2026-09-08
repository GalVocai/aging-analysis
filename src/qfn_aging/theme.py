"""Visual theme for the QFN-AGING GUI.

Kept apart from :mod:`qfn_aging.gui` so the window code stays about
behaviour and this file stays about looks. Colours follow the project's
green identity (the same family as the analysis plots and the app icon)
against warm neutrals, rather than Qt's default grey.
"""

from __future__ import annotations

# -- palette ---------------------------------------------------------------

GREEN_DARK = "#0f5132"      # primary actions, selected tab
GREEN = "#198754"           # accents, focus rings
GREEN_LIGHT = "#d1e7dd"     # selection wash
GREEN_TINT = "#f2f8f5"      # alternating rows

INK = "#1f2933"             # primary text
INK_SOFT = "#52606d"        # secondary text
MUTED = "#9aa5b1"           # excluded rows
LINE = "#dfe3e8"            # borders
SURFACE = "#ffffff"
CANVAS = "#f5f7f9"

DANGER = "#b42318"

FONT = '"Segoe UI", "Inter", system-ui, sans-serif'
MONO = '"Cascadia Mono", "Consolas", monospace'


STYLESHEET = f"""
QWidget {{
    background: {CANVAS};
    color: {INK};
    font-family: {FONT};
    font-size: 10pt;
}}

/* ---------- tabs ---------- */
QTabWidget::pane {{
    background: {SURFACE};
    border: 1px solid {LINE};
    border-radius: 8px;
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {INK_SOFT};
    padding: 8px 18px;
    margin-right: 2px;
    border: 1px solid transparent;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-weight: 500;
}}
QTabBar::tab:hover {{
    color: {GREEN_DARK};
    background: {GREEN_TINT};
}}
QTabBar::tab:selected {{
    background: {SURFACE};
    color: {GREEN_DARK};
    border: 1px solid {LINE};
    border-bottom-color: {SURFACE};
    font-weight: 600;
}}

/* ---------- buttons ---------- */
QPushButton {{
    background: {SURFACE};
    border: 1px solid {LINE};
    border-radius: 6px;
    padding: 6px 14px;
    color: {INK};
}}
QPushButton:hover {{
    border-color: {GREEN};
    color: {GREEN_DARK};
    background: {GREEN_TINT};
}}
QPushButton:pressed {{
    background: {GREEN_LIGHT};
}}
QPushButton:disabled {{
    color: {MUTED};
    border-color: {LINE};
    background: {CANVAS};
}}
QPushButton[primary="true"] {{
    background: {GREEN_DARK};
    color: #ffffff;
    border: 1px solid {GREEN_DARK};
    font-weight: 600;
}}
QPushButton[primary="true"]:hover {{
    background: {GREEN};
    border-color: {GREEN};
}}
QPushButton[primary="true"]:disabled {{
    background: {MUTED};
    border-color: {MUTED};
    color: #ffffff;
}}

/* ---------- inputs ---------- */
QLineEdit, QComboBox {{
    background: {SURFACE};
    border: 1px solid {LINE};
    border-radius: 6px;
    padding: 6px 10px;
    selection-background-color: {GREEN_LIGHT};
    selection-color: {INK};
}}
QLineEdit:focus, QComboBox:focus {{
    border-color: {GREEN};
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {SURFACE};
    border: 1px solid {LINE};
    selection-background-color: {GREEN_LIGHT};
    selection-color: {INK};
    outline: none;
}}

QCheckBox {{ spacing: 7px; color: {INK}; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 1px solid #b6bfc9;
    border-radius: 4px;
    background: {SURFACE};
}}
QCheckBox::indicator:hover {{ border-color: {GREEN}; }}
QCheckBox::indicator:checked {{
    background: {GREEN_DARK};
    border-color: {GREEN_DARK};
    image: none;
}}

/* ---------- trees ---------- */
QTreeWidget {{
    background: {SURFACE};
    alternate-background-color: {GREEN_TINT};
    border: 1px solid {LINE};
    border-radius: 8px;
    outline: none;
    show-decoration-selected: 1;
}}
QTreeWidget::item {{
    padding: 5px 4px;
    border-bottom: 1px solid #f0f2f4;
}}
QTreeWidget::item:hover {{ background: {GREEN_TINT}; }}
QTreeWidget::item:selected {{
    background: {GREEN_LIGHT};
    color: {INK};
}}
QTreeWidget::indicator {{
    width: 16px; height: 16px;
    border: 1px solid #b6bfc9;
    border-radius: 4px;
    background: {SURFACE};
}}
QTreeWidget::indicator:checked {{
    background: {GREEN_DARK};
    border-color: {GREEN_DARK};
}}
QTreeWidget::indicator:indeterminate {{
    background: {GREEN};
    border-color: {GREEN};
}}
QTableWidget {{
    background: {SURFACE};
    alternate-background-color: {GREEN_TINT};
    border: 1px solid {LINE};
    border-radius: 8px;
    gridline-color: #edf0f2;
    outline: none;
}}
QTableWidget::item {{ padding: 5px 6px; }}
QHeaderView::section {{
    background: {CANVAS};
    color: {INK_SOFT};
    padding: 8px 6px;
    border: none;
    border-bottom: 1px solid {LINE};
    font-weight: 600;
}}

/* ---------- log ---------- */
QPlainTextEdit {{
    background: {SURFACE};
    border: 1px solid {LINE};
    border-radius: 8px;
    padding: 8px;
    font-family: {MONO};
    font-size: 9pt;
    color: {INK_SOFT};
    selection-background-color: {GREEN_LIGHT};
    selection-color: {INK};
}}

/* ---------- misc ---------- */
QProgressBar {{
    border: 1px solid {LINE};
    border-radius: 6px;
    background: {SURFACE};
    text-align: center;
    height: 18px;
    color: {INK_SOFT};
}}
QProgressBar::chunk {{
    background: {GREEN};
    border-radius: 5px;
}}
QStatusBar {{
    background: {SURFACE};
    border-top: 1px solid {LINE};
    color: {INK_SOFT};
}}
QStatusBar::item {{ border: none; }}
QLabel[hint="true"] {{ color: {INK_SOFT}; }}
QLabel[heading="true"] {{
    color: {GREEN_DARK};
    font-size: 11pt;
    font-weight: 600;
}}

QScrollBar:vertical {{
    background: transparent; width: 11px; margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: #cbd2d9; border-radius: 5px; min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{ background: {MUTED}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
QScrollBar:horizontal {{
    background: transparent; height: 11px; margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: #cbd2d9; border-radius: 5px; min-width: 28px;
}}
"""
