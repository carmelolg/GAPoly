"""Coloured ``--help``: argparse builds the plain text, this module paints it.

Colours are added only on a terminal (and not when ``NO_COLOR`` is set; ``FORCE_COLOR`` forces them).
They never change the layout: removing the escape codes gives back the plain help text.
"""
from __future__ import annotations

import os
import re
import sys

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BOLD_CYAN = "\033[1;36m"
BOLD_MAGENTA = "\033[1;35m"

_OPTION = re.compile(r"(?<![\w-])(--?[A-Za-z][\w-]*)")
_METAVAR = re.compile(r"\b([A-Z][A-Z_]+|[A-Z]|0-1)\b")
_DEFAULT = re.compile(r"\(default:[^)]*\)")
_INVOCATION = re.compile(r"^(  -\S.*?)(\s{2,})(\S.*)?$")
_POSITIONAL = re.compile(r"^  (target)(?=\s)")
_HEADING = re.compile(r"^\S.*:$")


def use_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


def _paint(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


def _paint_options(text: str) -> str:
    return _OPTION.sub(lambda m: _paint(m.group(1), GREEN), text)


def _paint_invocation(text: str) -> str:
    """Option names green, value placeholders (N, PX, 0-1...) yellow."""
    parts = re.split(r"(--?[A-Za-z][\w-]*)", text)
    return "".join(
        _paint(part, GREEN) if i % 2 else _METAVAR.sub(lambda m: _paint(m.group(1), YELLOW), part)
        for i, part in enumerate(parts)
    )


def _dim_defaults(text: str) -> str:
    def dim(match: re.Match[str]) -> str:  # a default can wrap onto two lines: close the colour per line
        return "\n".join(_paint(line, DIM) for line in match.group(0).split("\n"))

    return _DEFAULT.sub(dim, text)


def colorize(text: str) -> str:
    if "\033[" in text:  # already coloured (newer argparse versions do it themselves)
        return text
    lines = _dim_defaults(text).split("\n")
    in_usage = True
    title_done = False
    out: list[str] = []
    for line in lines:
        plain = re.sub(r"\033\[[0-9;]*m", "", line)
        if in_usage:
            if not plain.strip():
                in_usage = False
                out.append(line)
                continue
            line = line.replace("usage:", _paint("usage:", BOLD), 1)
            out.append(_paint_invocation(line) if not plain.startswith("usage:") else _usage_line(line))
            continue
        if not title_done and plain.strip():
            title_done = True
            out.append(_paint(line, BOLD))
            continue
        if _HEADING.match(plain):
            out.append(_paint(line, BOLD_CYAN))
            continue
        match = _INVOCATION.match(plain)
        if match and match.group(1) == plain[: len(match.group(1))]:
            left_len = len(match.group(1))
            out.append(_paint_invocation(line[:left_len]) + _paint_options(line[left_len:]))
            continue
        line = _POSITIONAL.sub(lambda m: "  " + _paint(m.group(1), YELLOW), line)
        out.append(_paint_options(line))
    return "\n".join(out).replace("gapoly", _paint("gapoly", BOLD_MAGENTA))


def _usage_line(line: str) -> str:
    head, _, rest = line.partition("gapoly")
    return head + "gapoly" + _paint_invocation(rest)
