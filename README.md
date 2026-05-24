# agent-fn-registry

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/agent-fn-registry.svg)](https://pypi.org/project/agent-fn-registry/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**One registry for LLM agent tools.** Pair a Python callable with its schema, side-effect tags, and default args. Dispatch by name. Build Anthropic or OpenAI tools lists. Zero deps.

```python
from agent_fn_registry import Registry

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

# dispatch
result = reg.dispatch("search_web", args={"q": "anthropic prompt cache"})

# tools list for the LLM
anthropic_tools = reg.anthropic_tools()   # [{"name", "description", "input_schema"}]
openai_tools    = reg.openai_functions()  # [{"type": "function", "function": {...}}]
```

## Why

Three things go out of sync without a shared owner:

1. The dict of `tool_name → callable` your dispatcher uses.
2. The list of tool schemas you hand to the LLM.
3. The side-effect tags your retry/scheduler/confirmation layer needs.

`Registry` is one in-process owner of all three. The decorator adds a tool everywhere at once. Dispatch merges declared defaults under caller args. Inspection helpers filter by side effect for safe-tool-scheduling.

The registry does NOT validate args against the schema or coerce types — that's the job of the LLM-args pipeline ([`tool-arg-rename`](https://github.com/MukundaKatta/tool-arg-rename) → [`tool-arg-defaults`](https://github.com/MukundaKatta/tool-arg-defaults) → [`tool-arg-coerce-py`](https://github.com/MukundaKatta/tool-arg-coerce-py) → [`tool-arg-fuzzy`](https://github.com/MukundaKatta/tool-arg-fuzzy) → [`agentvet`](https://github.com/MukundaKatta/agentvet)). Zero deps so you can compose the registry into any of them.

## Install

```bash
pip install agent-fn-registry
```

## API

```python
from agent_fn_registry import Registry, ToolEntry, ToolNotFoundError

reg = Registry()

# decorator
@reg.tool(schema=..., side_effects=..., defaults=..., name=None)
def fn(...): ...

# direct
reg.register(name, fn, *, schema, side_effects=None, defaults=None) -> ToolEntry
reg.unregister(name) -> bool
reg.clear()

# inspect
reg.has(name); name in reg; len(reg)
reg.tool_names()                  # sorted
reg.get(name) -> ToolEntry        # raises ToolNotFoundError
reg.get_schema(name)              # copy
reg.side_effects_of(name)         # frozenset
reg.defaults_of(name)             # copy
reg.with_side_effect("read")      # list[ToolEntry]
reg.without_side_effect("destructive")

# dispatch
reg.dispatch(name, args=None)     # defaults merged, caller args override

# bulk export
reg.anthropic_tools()             # list of schema dicts
reg.openai_functions()            # OpenAI function-calling shape
```

## Companion libraries

- [`tool-schema-from-fn`](https://github.com/MukundaKatta/tool-schema-from-fn) — generate the `schema` arg from a function signature + docstring.
- [`tool-side-effects-tag`](https://github.com/MukundaKatta/tool-side-effects-tag) — pre-defined tag set and `is_parallel_safe` / `is_retry_safe` helpers.
- [`tool-arg-*`](https://github.com/MukundaKatta?tab=repositories&q=tool-arg) — the LLM-args repair pipeline that runs before `dispatch`.

## License

MIT
