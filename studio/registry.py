"""Tool registry.

A tool is a pure function `fn(img, **params) -> img` (BGR uint8 in, BGR uint8 out) plus a
declarative list of parameters. The UI builds widgets from the parameter list, and the edit
history stores only (tool name, params), so any step can be undone or replayed.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

CATEGORIES = [
    "Adjust",
    "Color",
    "Geometry",
    "Reshape",
    "Blur and noise",
    "Edges",
    "Threshold and morphology",
    "Histogram",
    "Artistic",
    "Annotate",
    "Masks and blending",
    "Detect",
]


@dataclass
class Param:
    kind: str  # slider, range, select, toggle, color, text, textarea, emoji, image
    key: str
    label: str
    default: Any = None
    min: Any = None
    max: Any = None
    step: Any = None
    options: Optional[list] = None
    help: Optional[str] = None
    when: Optional[Callable[[dict], bool]] = None  # show only if when(values_so_far) is True


@dataclass
class Tool:
    name: str
    category: str
    fn: Callable
    params: list = field(default_factory=list)
    explain: str = ""
    heavy: bool = False  # slow tools get preview on demand instead of live


TOOLS: dict = {}


def tool(category, name, params=(), explain="", heavy=False):
    def register(fn):
        TOOLS[name] = Tool(name, category, fn, list(params), explain.strip(), heavy)
        return fn

    return register


def by_category(category):
    return [t for t in TOOLS.values() if t.category == category]


def defaults(t):
    return {p.key: p.default for p in t.params}


def run(t, img, params, aux=None):
    """Apply a tool. Image parameters are stored as ids and resolved from `aux` here."""
    values = dict(params)
    for p in t.params:
        if p.kind == "image":
            values[p.key] = (aux or {}).get(values.get(p.key))
    return t.fn(img.copy(), **values)


def describe(t, params):
    """Short human-readable summary of a step for the history list."""
    parts = []
    for p in t.params:
        if p.when and not p.when(params):
            continue
        v = params.get(p.key)
        if p.kind == "image":
            v = "image" if v else "none"
        elif isinstance(v, float):
            v = round(v, 2)
        elif isinstance(v, str) and len(v) > 18:
            v = v[:18] + "..."
        parts.append(f"{p.label.lower()}={v}")
    return ", ".join(parts[:4])
