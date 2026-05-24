"""Tests for agent_fn_registry.Registry."""

from __future__ import annotations

import pytest

from agent_fn_registry import Registry, ToolEntry, ToolNotFoundError


SAMPLE_SCHEMA = {
    "name": "search",
    "description": "Search the web.",
    "input_schema": {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    },
}


# ---- register / get ---------------------------------------------------


def test_register_returns_entry():
    reg = Registry()
    entry = reg.register("search", lambda q: f"found:{q}", schema=SAMPLE_SCHEMA)
    assert isinstance(entry, ToolEntry)
    assert entry.name == "search"


def test_register_normalizes_schema_name():
    reg = Registry()
    schema = {"name": "wrong", "description": "x", "input_schema": {}}
    reg.register("right", lambda: None, schema=schema)
    assert reg.get_schema("right")["name"] == "right"


def test_get_raises_for_unknown():
    reg = Registry()
    with pytest.raises(ToolNotFoundError):
        reg.get("missing")


def test_get_schema_returns_copy():
    reg = Registry()
    reg.register("s", lambda: None, schema=SAMPLE_SCHEMA)
    snap = reg.get_schema("s")
    snap["description"] = "MUTATED"
    assert reg.get_schema("s")["description"] == "Search the web."


def test_register_rejects_non_callable():
    reg = Registry()
    with pytest.raises(TypeError):
        reg.register("x", "not callable", schema=SAMPLE_SCHEMA)  # type: ignore[arg-type]


def test_register_rejects_non_dict_schema():
    reg = Registry()
    with pytest.raises(TypeError):
        reg.register("x", lambda: None, schema="not a dict")  # type: ignore[arg-type]


def test_register_with_side_effects_and_defaults():
    reg = Registry()
    reg.register(
        "search",
        lambda q, max_results=10: (q, max_results),
        schema=SAMPLE_SCHEMA,
        side_effects=["read", "network"],
        defaults={"max_results": 5},
    )
    e = reg.get("search")
    assert e.side_effects == frozenset({"read", "network"})
    assert e.defaults == {"max_results": 5}


# ---- decorator ----------------------------------------------------


def test_decorator_registers_by_schema_name():
    reg = Registry()

    @reg.tool(schema=SAMPLE_SCHEMA)
    def search(q):
        return f"found:{q}"

    assert reg.has("search")
    assert search("x") == "found:x"  # decorator returns the original fn


def test_decorator_explicit_name_overrides_schema_name():
    reg = Registry()

    @reg.tool(schema=SAMPLE_SCHEMA, name="custom")
    def search(q):
        return q

    assert reg.has("custom")
    assert not reg.has("search")


def test_decorator_falls_back_to_function_name_when_schema_has_no_name():
    reg = Registry()

    @reg.tool(schema={"description": "no name in schema", "input_schema": {}})
    def my_tool():
        return 1

    assert reg.has("my_tool")


# ---- dispatch ----------------------------------------------------


def test_dispatch_calls_fn_with_args():
    reg = Registry()
    reg.register("search", lambda q: f"found:{q}", schema=SAMPLE_SCHEMA)
    assert reg.dispatch("search", {"q": "anthropic"}) == "found:anthropic"


def test_dispatch_merges_defaults_under_args():
    reg = Registry()
    reg.register(
        "search",
        lambda q, max_results=1: (q, max_results),
        schema=SAMPLE_SCHEMA,
        defaults={"max_results": 10},
    )
    assert reg.dispatch("search", {"q": "x"}) == ("x", 10)
    assert reg.dispatch("search", {"q": "x", "max_results": 5}) == ("x", 5)


def test_dispatch_unknown_raises():
    reg = Registry()
    with pytest.raises(ToolNotFoundError):
        reg.dispatch("nope")


def test_dispatch_no_args_passes_only_defaults():
    reg = Registry()
    reg.register(
        "f",
        lambda x=1: x * 2,
        schema=SAMPLE_SCHEMA,
        defaults={"x": 3},
    )
    assert reg.dispatch("f") == 6


# ---- bulk export ------------------------------------------------


def test_anthropic_tools_lists_all():
    reg = Registry()
    reg.register("a", lambda: None, schema={"name": "a", "description": "A", "input_schema": {}})
    reg.register("b", lambda: None, schema={"name": "b", "description": "B", "input_schema": {}})
    tools = reg.anthropic_tools()
    names = sorted(t["name"] for t in tools)
    assert names == ["a", "b"]


def test_openai_functions_shape():
    reg = Registry()
    reg.register("search", lambda q: q, schema=SAMPLE_SCHEMA)
    fns = reg.openai_functions()
    assert len(fns) == 1
    assert fns[0]["type"] == "function"
    assert fns[0]["function"]["name"] == "search"
    assert fns[0]["function"]["description"] == "Search the web."
    assert "parameters" in fns[0]["function"]


def test_openai_functions_uses_parameters_key_when_no_input_schema():
    reg = Registry()
    reg.register("x", lambda: None, schema={"name": "x", "parameters": {"type": "object"}})
    fns = reg.openai_functions()
    assert fns[0]["function"]["parameters"] == {"type": "object"}


# ---- inspection / filtering -----------------------------------


def test_has_contains_len():
    reg = Registry()
    assert "x" not in reg
    assert len(reg) == 0
    reg.register("x", lambda: None, schema=SAMPLE_SCHEMA)
    assert reg.has("x")
    assert "x" in reg
    assert len(reg) == 1


def test_contains_rejects_non_string():
    reg = Registry()
    reg.register("x", lambda: None, schema=SAMPLE_SCHEMA)
    assert 42 not in reg
    assert None not in reg


def test_tool_names_sorted():
    reg = Registry()
    reg.register("zebra", lambda: None, schema=SAMPLE_SCHEMA)
    reg.register("alpha", lambda: None, schema=SAMPLE_SCHEMA)
    assert reg.tool_names() == ["alpha", "zebra"]


def test_side_effects_of_returns_set():
    reg = Registry()
    reg.register("a", lambda: None, schema=SAMPLE_SCHEMA, side_effects=["read"])
    assert reg.side_effects_of("a") == frozenset({"read"})


def test_defaults_of_returns_copy():
    reg = Registry()
    reg.register("a", lambda: None, schema=SAMPLE_SCHEMA, defaults={"x": 1})
    snap = reg.defaults_of("a")
    snap["x"] = 999
    assert reg.defaults_of("a")["x"] == 1


def test_with_side_effect_filters():
    reg = Registry()
    reg.register("r1", lambda: None, schema=SAMPLE_SCHEMA, side_effects=["read"])
    reg.register("w1", lambda: None, schema=SAMPLE_SCHEMA, side_effects=["write"])
    reg.register("r2", lambda: None, schema=SAMPLE_SCHEMA, side_effects=["read", "network"])

    reads = reg.with_side_effect("read")
    assert sorted(e.name for e in reads) == ["r1", "r2"]


def test_without_side_effect_filters():
    reg = Registry()
    reg.register("safe", lambda: None, schema=SAMPLE_SCHEMA, side_effects=["read"])
    reg.register("risky", lambda: None, schema=SAMPLE_SCHEMA, side_effects=["destructive"])

    safe = reg.without_side_effect("destructive")
    assert [e.name for e in safe] == ["safe"]


# ---- mutation ------------------------------------------------


def test_unregister_returns_bool():
    reg = Registry()
    reg.register("x", lambda: None, schema=SAMPLE_SCHEMA)
    assert reg.unregister("x") is True
    assert reg.unregister("x") is False


def test_clear_empties_registry():
    reg = Registry()
    reg.register("a", lambda: None, schema=SAMPLE_SCHEMA)
    reg.register("b", lambda: None, schema=SAMPLE_SCHEMA)
    reg.clear()
    assert len(reg) == 0
