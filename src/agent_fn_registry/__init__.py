"""agent-fn-registry - one registry per LLM agent tool.

Every agent loop has the same boilerplate: a dict of tool-name → callable,
a parallel list of tool schemas you hand to the LLM, optional default
args you merge in, side-effect tags you check before dispatching. They
get out of sync.

`Registry` is the small shared owner of all three:

    from agent_fn_registry import Registry, ToolNotFoundError

    reg = Registry()

    @reg.tool(
        schema={
            "name": "search_web",
            "description": "Search the web for a query.",
            "input_schema": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        },
        side_effects=["read", "network"],
        defaults={"max_results": 10, "timeout": 30},
    )
    def search_web(q: str, *, max_results: int = 10, timeout: int = 30):
        return real_search(q, max_results=max_results, timeout=timeout)

    # Dispatch by name. Defaults filled in automatically.
    result = reg.dispatch("search_web", args={"q": "anthropic"})

    # Build the tools list to hand to Anthropic.
    tools = reg.anthropic_tools()
    # [{"name": "search_web", "description": "...", "input_schema": {...}}, ...]

    # Build OpenAI shape.
    tools = reg.openai_functions()
    # [{"type": "function", "function": {...}}, ...]

    # Inspect.
    reg.has("search_web")
    reg.tool_names()
    reg.side_effects_of("search_web")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

__version__ = "0.1.0"
__all__ = [
    "Registry",
    "ToolEntry",
    "ToolNotFoundError",
]


class ToolNotFoundError(KeyError):
    """Raised when `get`/`dispatch` is called for an unregistered tool."""

    def __init__(self, name: str):
        self.name = name
        super().__init__(name)


@dataclass
class ToolEntry:
    """One registered tool."""

    name: str
    fn: Callable[..., Any]
    schema: dict[str, Any]
    side_effects: frozenset[str] = field(default_factory=frozenset)
    defaults: dict[str, Any] = field(default_factory=dict)


class Registry:
    """In-process registry of LLM agent tools.

    A `Registry` knows, for each tool:

      - the Python callable (`fn`)
      - the JSON Schema shape (`schema` — Anthropic `input_schema`-style dict)
      - the side-effect tag set (e.g. {"read", "network"})
      - optional default kwargs to merge in at dispatch time

    The registry does NOT validate args against the schema or coerce types;
    those concerns belong to companion libs (`agentvet`, `tool-arg-coerce-py`).
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}

    # ---- registration ------------------------------------------------

    def tool(
        self,
        *,
        schema: dict[str, Any],
        side_effects: Iterable[str] | None = None,
        defaults: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator. Registers the decorated function as a tool.

            @reg.tool(schema={...}, side_effects=["read"])
            def search(q): ...
        """
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            resolved = name or schema.get("name") or fn.__name__
            self.register(
                resolved,
                fn,
                schema=schema,
                side_effects=side_effects,
                defaults=defaults,
            )
            return fn

        return decorator

    def register(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        schema: dict[str, Any],
        side_effects: Iterable[str] | None = None,
        defaults: dict[str, Any] | None = None,
    ) -> ToolEntry:
        """Register a tool by name."""
        if not callable(fn):
            raise TypeError(f"fn for {name!r} must be callable")
        if not isinstance(schema, dict):
            raise TypeError(f"schema for {name!r} must be a dict")
        # ensure the schema's `name` field matches the registry key
        schema = dict(schema)
        schema["name"] = name
        entry = ToolEntry(
            name=name,
            fn=fn,
            schema=schema,
            side_effects=frozenset(side_effects or ()),
            defaults=dict(defaults) if defaults else {},
        )
        self._tools[name] = entry
        return entry

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def clear(self) -> None:
        self._tools.clear()

    # ---- inspection --------------------------------------------------

    def has(self, name: str) -> bool:
        return name in self._tools

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> ToolEntry:
        if name not in self._tools:
            raise ToolNotFoundError(name)
        return self._tools[name]

    def get_schema(self, name: str) -> dict[str, Any]:
        return dict(self.get(name).schema)

    def side_effects_of(self, name: str) -> frozenset[str]:
        return self.get(name).side_effects

    def defaults_of(self, name: str) -> dict[str, Any]:
        return dict(self.get(name).defaults)

    # ---- dispatch ---------------------------------------------------

    def dispatch(
        self,
        name: str,
        args: dict[str, Any] | None = None,
    ) -> Any:
        """Look up the tool and call it.

        Defaults are merged in first (caller-supplied keys win on conflict),
        then the merged kwargs are passed to the registered function.
        """
        entry = self.get(name)
        merged = dict(entry.defaults)
        if args:
            merged.update(args)
        return entry.fn(**merged)

    # ---- bulk schema export ----------------------------------------

    def anthropic_tools(self) -> list[dict[str, Any]]:
        """Return all schemas in Anthropic Messages-API shape.

        Each entry is the schema dict itself (already shaped as
        `{name, description, input_schema}`).
        """
        return [dict(e.schema) for e in self._tools.values()]

    def openai_functions(self) -> list[dict[str, Any]]:
        """Return all schemas in OpenAI function-calling shape."""
        out: list[dict[str, Any]] = []
        for e in self._tools.values():
            schema = e.schema
            out.append({
                "type": "function",
                "function": {
                    "name": schema["name"],
                    "description": schema.get("description", ""),
                    "parameters": schema.get(
                        "input_schema",
                        schema.get("parameters", {}),
                    ),
                },
            })
        return out

    # ---- filtering -------------------------------------------------

    def with_side_effect(self, effect: str) -> list[ToolEntry]:
        """All tools that carry `effect` in their side_effects set."""
        return [e for e in self._tools.values() if effect in e.side_effects]

    def without_side_effect(self, effect: str) -> list[ToolEntry]:
        return [e for e in self._tools.values() if effect not in e.side_effects]
