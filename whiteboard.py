"""Teacher whiteboard tools for Mia (the LiveKit agent).

Each tool publishes ONE drawing operation to the room as a reliable LiveKit
data message on topic "whiteboard". The teach.html console renders them on a
canvas, scaled from a normalized 1000x1000 space (origin top-left).

Timing note: tools execute the moment the LLM emits them, which is BEFORE
the spoken reply finishes generating — so drawings land slightly AHEAD of
the narration. The system prompt leans into this (draw-then-explain per
step), which reads as natural teacher behavior: "here's the setup ... now
watch what happens".

All tools degrade gracefully: if publishing fails they return a short
spoken fallback instead of raising, so Mia carries on with voice only.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from livekit.agents import RunContext, function_tool

WHITEBOARD_TOPIC = "whiteboard"

# Normalized board space: 0..1000 on both axes, origin top-left. Integers
# keep the LLM's coordinate math simple; the console scales to pixels.
_BOARD_MAX = 1000

_HEX_COLOR = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
_NAMED_COLORS = {
    "white", "black", "red", "green", "blue", "yellow", "orange",
    "purple", "pink", "cyan", "teal", "gray", "grey",
}
_DEFAULT_COLOR = "#e6e8ee"


def _clamp_int(value: float, lo: int = 0, hi: int = _BOARD_MAX) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        number = lo
    return max(lo, min(hi, number))


def _color(value: str) -> str:
    text = (value or "").strip()
    if _HEX_COLOR.match(text):
        return text
    if text.lower() in _NAMED_COLORS:
        return text.lower()
    return _DEFAULT_COLOR


async def _publish(ctx: RunContext, op: dict[str, Any]) -> bool:
    """Send one drawing op to the room. Never raises; False means the board
    is off or unreachable and Mia should just keep talking."""
    if os.getenv("WHITEBOARD_ENABLED", "1") != "1":
        return False
    try:
        room = ctx.session.room_io.room
    except Exception:
        return False
    try:
        await room.local_participant.publish_data(
            json.dumps(op), reliable=True, topic=WHITEBOARD_TOPIC
        )
    except Exception:
        return False
    return True


def _ok(summary: str) -> str:
    return f"On the board now: {summary} Explain it in voice as you go."


def _unavailable() -> str:
    return (
        "The whiteboard is not available right now. "
        "Carry on with voice only, do not mention the board."
    )


@function_tool
async def wb_clear(ctx: RunContext) -> str:
    """Wipe the teacher whiteboard clean. Use when starting a fresh
    explanation or when the board is full and you need space."""
    if await _publish(ctx, {"type": "wb.clear"}):
        return "Board cleared. " + _ok("a blank board")
    return _unavailable()


@function_tool
async def wb_text(
    ctx: RunContext,
    x: int,
    y: int,
    text: str,
    size: int = 30,
    color: str = "#e6e8ee",
) -> str:
    """Write a short label or word on the whiteboard, e.g. naming parts of a
    diagram. Keep it to a few words - never sentences or paragraphs.

    Args:
        x: Horizontal position, 0 (left) to 1000 (right).
        y: Vertical position, 0 (top) to 1000 (bottom).
        text: The words to write, max ~120 characters.
        size: Text size in board units, 12 to 80.
        color: Hex like '#60a5fa' or a plain name like 'yellow'.
    """
    words = (text or "").strip()[:120]
    if not words:
        return "Nothing to write - the text was empty."
    op = {
        "type": "wb.text",
        "x": _clamp_int(x),
        "y": _clamp_int(y),
        "text": words,
        "size": _clamp_int(size, 12, 80),
        "color": _color(color),
    }
    if await _publish(ctx, op):
        return _ok(f"label '{words}'")
    return _unavailable()


@function_tool
async def wb_line(
    ctx: RunContext,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: str = "#60a5fa",
    width: int = 5,
) -> str:
    """Draw a straight line on the whiteboard, e.g. an axis, a timeline, or
    connecting two parts of a diagram. Coordinates are 0-1000, origin top-left.

    Args:
        x1: Start horizontal position, 0 to 1000.
        y1: Start vertical position, 0 to 1000.
        x2: End horizontal position, 0 to 1000.
        y2: End vertical position, 0 to 1000.
        color: Hex like '#60a5fa' or a plain name like 'yellow'.
        width: Line width in board units, 1 to 20.
    """
    op = {
        "type": "wb.line",
        "x1": _clamp_int(x1),
        "y1": _clamp_int(y1),
        "x2": _clamp_int(x2),
        "y2": _clamp_int(y2),
        "color": _color(color),
        "width": _clamp_int(width, 1, 20),
    }
    if await _publish(ctx, op):
        return _ok("a line")
    return _unavailable()


@function_tool
async def wb_arrow(
    ctx: RunContext,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: str = "#fbbf24",
    width: int = 5,
) -> str:
    """Draw an arrow on the whiteboard, e.g. to show direction, flow, force,
    or cause-and-effect. Coordinates are 0-1000, origin top-left.

    Args:
        x1: Tail horizontal position, 0 to 1000.
        y1: Tail vertical position, 0 to 1000.
        x2: Head horizontal position, 0 to 1000.
        y2: Head vertical position, 0 to 1000.
        color: Hex like '#fbbf24' or a plain name like 'yellow'.
        width: Line width in board units, 1 to 20.
    """
    op = {
        "type": "wb.arrow",
        "x1": _clamp_int(x1),
        "y1": _clamp_int(y1),
        "x2": _clamp_int(x2),
        "y2": _clamp_int(y2),
        "color": _color(color),
        "width": _clamp_int(width, 1, 20),
    }
    if await _publish(ctx, op):
        return _ok("an arrow")
    return _unavailable()


@function_tool
async def wb_rect(
    ctx: RunContext,
    x: int,
    y: int,
    w: int,
    h: int,
    color: str = "#34d399",
    width: int = 5,
) -> str:
    """Draw a rectangle outline on the whiteboard, e.g. a box in a flowchart
    or a container. Coordinates are 0-1000, origin top-left.

    Args:
        x: Left edge, 0 to 1000.
        y: Top edge, 0 to 1000.
        w: Box width in board units, 10 to 1000.
        h: Box height in board units, 10 to 1000.
        color: Hex like '#34d399' or a plain name like 'green'.
        width: Line width in board units, 1 to 20.
    """
    op = {
        "type": "wb.rect",
        "x": _clamp_int(x),
        "y": _clamp_int(y),
        "w": _clamp_int(w, 10, 1000),
        "h": _clamp_int(h, 10, 1000),
        "color": _color(color),
        "width": _clamp_int(width, 1, 20),
    }
    if await _publish(ctx, op):
        return _ok("a box")
    return _unavailable()


@function_tool
async def wb_circle(
    ctx: RunContext,
    x: int,
    y: int,
    r: int,
    color: str = "#f472b6",
    width: int = 5,
) -> str:
    """Draw a circle outline on the whiteboard, e.g. the sun, a cell, a wheel,
    or circling something important. Coordinates are 0-1000, origin top-left.

    Args:
        x: Center horizontal position, 0 to 1000.
        y: Center vertical position, 0 to 1000.
        r: Radius in board units, 10 to 500.
        color: Hex like '#f472b6' or a plain name like 'pink'.
        width: Line width in board units, 1 to 20.
    """
    op = {
        "type": "wb.circle",
        "x": _clamp_int(x),
        "y": _clamp_int(y),
        "r": _clamp_int(r, 10, 500),
        "color": _color(color),
        "width": _clamp_int(width, 1, 20),
    }
    if await _publish(ctx, op):
        return _ok("a circle")
    return _unavailable()


@function_tool
async def wb_stroke(
    ctx: RunContext,
    points: list[float],
    color: str = "#e6e8ee",
    width: int = 5,
) -> str:
    """Draw a freehand stroke through a list of points, e.g. a curve, a wave,
    an underline, or a quick sketch. Keep it under ~30 points.

    Args:
        points: Flat coordinate list [x1, y1, x2, y2, ...], each 0 to 1000,
            origin top-left. At least two points (4 numbers).
        color: Hex like '#e6e8ee' or a plain name like 'white'.
        width: Line width in board units, 1 to 20.
    """
    try:
        coords = [float(p) for p in (points or [])][:60]
    except (TypeError, ValueError):
        coords = []
    if len(coords) < 4:
        return "Nothing drawn - a stroke needs at least two points."
    if len(coords) % 2:
        coords = coords[:-1]
    op = {
        "type": "wb.stroke",
        "points": [_clamp_int(c) for c in coords],
        "color": _color(color),
        "width": _clamp_int(width, 1, 20),
    }
    if await _publish(ctx, op):
        return _ok("a freehand stroke")
    return _unavailable()


@function_tool
async def wb_highlight(
    ctx: RunContext,
    x: int,
    y: int,
) -> str:
    """Pulse a glowing marker on the whiteboard to say 'look here' while you
    explain that spot. Coordinates are 0-1000, origin top-left.

    Args:
        x: Horizontal position, 0 to 1000.
        y: Vertical position, 0 to 1000.
    """
    op = {"type": "wb.highlight", "x": _clamp_int(x), "y": _clamp_int(y)}
    if await _publish(ctx, op):
        return _ok("a highlight marker")
    return _unavailable()
