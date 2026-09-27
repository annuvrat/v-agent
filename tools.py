"""Lightweight function tools for the voice agent.

Each tool is decorated with @function_tool so the LLM (Groq) can invoke it
mid-conversation. Nothing here is required at startup; any tool that needs a
network/API call degrades gracefully and returns a short spoken reply instead
of raising (that way Mia simply tells the user she can't do it right now).

Note: because tools run inside an active session, they are only ever invoked
while a job is live, so plain httpx usage here is safe.
"""

from __future__ import annotations

import ast
import operator
from datetime import datetime
from typing import Any

import httpx
from livekit.agents import function_tool

TAVILY_URL = "https://api.tavily.com/search"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

_HTTP_TIMEOUT = httpx.Timeout(15.0)

# weather_code -> short spoken description (Open-Meteo WMO codes)
_WEATHER_CODES: dict[int, str] = {
    0: "clear skies",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light showers",
    81: "showers",
    82: "heavy showers",
    85: "snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}


def _tavily_key() -> str:
    import os

    return os.getenv("TAVILY_API_KEY", "").strip()


@function_tool
async def web_search(query: str, max_results: int = 4) -> str:
    """Search the web for current, factual information. Only use this when the
    user asks about recent events, live facts, or anything you cannot be sure
    of. Returns a short spoken summary with sources.

    Args:
        query: The search query, keep it specific (e.g. 'latest iPhone 17 release date').
        max_results: How many results to return, keep between 1 and 5.
    """
    if not _tavily_key():
        return (
            "Web search is not configured because no Tavily API key is set. "
            "Tell the user that live web search is not connected right now."
        )
    max_results = max(1, min(int(max_results), 5))
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.post(
                TAVILY_URL,
                json={
                    "api_key": _tavily_key(),
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                    "include_answer": True,
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:  # noqa: BLE001 - degrade to a spoken apology
        return f"Web search failed with an error: {type(exc).__name__}. Tell the user it could not be reached."

    parts: list[str] = []
    answer = (data.get("answer") or "").strip()
    if answer:
        parts.append(answer)
    for item in data.get("results", [])[:max_results]:
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        snippet = (item.get("content") or "").strip().replace("\n", " ")[:220]
        if title:
            parts.append(f"{title}. {snippet} Source: {url}")  # pragma: no cover
    return " ".join(parts) if parts else "The search returned no useful results."


@function_tool
async def get_weather_today(location: str) -> str:
    """Get today's live weather for a place. Use when the user asks about the
    weather, temperature, or what it's like outside somewhere.

    Args:
        location: A city or place name, e.g. 'Mumbai' or 'New York'.
    """
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            geo = await client.get(
                GEOCODING_URL,
                params={"name": location, "count": 1, "language": "en", "format": "json"},
            )
            geo.raise_for_status()
            matches = geo.json().get("results", [])
            if not matches:
                return f"I could not find a place called {location}."
            place = matches[0]
            wx = await client.get(
                WEATHER_URL,
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
                    "timezone": "auto",
                },
            )
            wx.raise_for_status()
            cur = wx.json()["current"]
    except Exception as exc:  # noqa: BLE001
        return f"Getting the weather failed with a {type(exc).__name__} error."

    name = place.get("name", location)
    country = place.get("country", "")
    desc = _WEATHER_CODES.get(int(cur.get("weather_code", 0)), "unusual conditions")
    temp = round(cur.get("temperature_2m", 0))
    feels = round(cur.get("apparent_temperature", temp))
    wind = round(cur.get("wind_speed_10m", 0))
    hum = round(cur.get("relative_humidity_2m", 0))
    text = (
        f"In {name}, {country or 'right now'}, it's {temp} degrees with {desc}. "
        f"It feels like about {feels}. Wind is around {wind} kilometers per hour, "
        f"and humidity is about {hum} percent."
    )
    return text


@function_tool
def what_time_is_it() -> str:
    """Return the current local date and clock time. Use when the user asks what
    time or date it is.

    Returns:
        The current date and time as spoken text.
    """
    now = datetime.now()
    body = now.strftime("%I:%M").lstrip("0")
    ampm = now.strftime("%p").lower()
    date = now.strftime("%A, %B %d, %Y")
    return f"It is {body} {ampm} on {date}."


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Div | ast.FloorDiv | ast.Mod) and right == 0:
            raise ZeroDivisionError("cannot divide by zero")
        return float(_BIN_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_eval(node.operand)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return +_safe_eval(node.operand)
    raise ValueError(f"unsupported expression: {type(node).__name__}")


@function_tool
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression like '12 * 8' or '(45 + 5) / 2'.
    Only numbers and + - * / // % are allowed. Use when the user asks for a
    quick calculation.

    Args:
        expression: A plain arithmetic expression using numbers and operators.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        value = _safe_eval(tree.body)
    except Exception as exc:  # noqa: BLE001
        return f"I couldn't work that one out ({type(exc).__name__}). Please rephrase the calculation."
    if abs(value) >= 1e12 or (abs(value) < 1e-6 and value != 0):
        return f"The result is roughly {value:.3e}."
    text = f"{value:g}"
    if value.is_integer():
        return f"The answer is {int(value)}."
    return f"The answer is about {value:.3f}."