"""Model context is explicit and provider calls remain mockable."""

import importlib
import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from experiments.agent_policy_search import model
from experiments.agent_policy_search.config import ModelConfig


def _response(content: Any) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}]}


def test_request_contains_only_supplied_context_and_provider_settings() -> None:
    config = ModelConfig(
        model="provider/model-name",
        api_base="https://provider.example/v1",
        max_tokens=2048,
        temperature=0.2,
        timeout_seconds=40,
    )
    task = {"action_shape": [2], "dtype": "float32", "bounds": [-1, 1]}
    feedback = {"returns": [1.0, 3.0], "mean_return": 2.0}
    latest = {"source": "bad_attempt", "error": "invalid action shape"}
    request = model.build_request(
        config,
        task=task,
        incumbent_source="def act(obs, rng): ...",
        incumbent_feedback=feedback,
        latest=latest,
    )

    assert set(request) == {
        "model",
        "messages",
        "max_tokens",
        "timeout",
        "temperature",
        "api_base",
    }
    assert request["model"] == config.model
    assert request["max_tokens"] == 2048
    assert request["timeout"] == 40
    assert request["temperature"] == 0.2
    assert request["api_base"] == config.api_base
    assert [message["role"] for message in request["messages"]] == ["system", "user"]
    context = json.loads(request["messages"][1]["content"])
    assert context == {
        "task": task,
        "incumbent": {"source": "def act(obs, rng): ...", "feedback": feedback},
        "latest_attempt": latest,
    }
    assert "stateless" in request["messages"][0]["content"]
    assert "act(obs, rng) -> action" in request["messages"][0]["content"]
    json.dumps(request, allow_nan=False)


def test_next_request_does_not_retain_earlier_context() -> None:
    config = ModelConfig(model="provider/model-name")
    common = {"task": {}, "incumbent_source": "initial", "incumbent_feedback": {}}
    first = model.build_request(config, **common, latest={"marker": "old attempt"})
    second = model.build_request(config, **common, latest=None)
    assert "old attempt" in json.dumps(first)
    assert "old attempt" not in json.dumps(second)
    assert "temperature" not in second
    assert "api_base" not in second
    assert "response_format" not in second


def test_import_and_request_build_do_not_initialize_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "litellm", None)
    importlib.reload(model)
    request = model.build_request(
        ModelConfig(model="provider/model-name"),
        task={},
        incumbent_source="initial",
        incumbent_feedback={},
        latest=None,
    )
    assert request["model"] == "provider/model-name"


def test_complete_preserves_full_response_and_forwards_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = {"model": "provider/model-name", "messages": [], "timeout": 20}
    payload = {
        "choices": [{"message": {"content": "example"}}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 20},
        "id": "response-123",
        "model": "provider-model-version",
        "provider_specific_fields": {"trace_id": "trace-123"},
    }
    calls: list[dict[str, Any]] = []

    def completion(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)

        def model_dump(*, mode: str) -> dict[str, Any]:
            assert mode == "json"
            return payload

        return SimpleNamespace(model_dump=model_dump)

    provider = ModuleType("litellm")
    provider.completion = completion
    monkeypatch.setitem(sys.modules, "litellm", provider)
    assert model.complete(request) == payload
    assert calls == [request]


def test_provider_error_propagates_without_retry_or_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def completion(**kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("provider authentication failed")

    provider = ModuleType("litellm")
    provider.completion = completion
    monkeypatch.setitem(sys.modules, "litellm", provider)
    with pytest.raises(RuntimeError, match="provider authentication failed"):
        model.complete({"model": "provider/model-name", "messages": []})
    assert calls == 1


@pytest.mark.parametrize("fence", [None, "json", ""])
def test_parse_normal_and_fenced_json(fence: str | None) -> None:
    source = "import jax.numpy as jnp\n\ndef act(obs, rng):\n    return jnp.zeros(2)\n"
    content = json.dumps({"source": source, "description": "Use a constant baseline"})
    if fence is not None:
        content = f"```{fence}\n{content}\n```"
    proposal = model.parse_proposal(_response(content))
    assert proposal.source == source
    assert proposal.description == "Use a constant baseline"


@pytest.mark.parametrize(
    "response, message",
    [
        ({}, "no message content"),
        ({"choices": []}, "no message content"),
        (_response(None), "nonempty text"),
        (_response("   "), "nonempty text"),
        (_response("not JSON"), "JSON object"),
        (_response("[]"), "only source and description"),
        (_response('{"source":"x"}'), "only source and description"),
        (
            _response('{"source":"x","description":"y","extra":true}'),
            "only source and description",
        ),
        (_response('{"source":"","description":"y"}'), "source.*nonempty"),
        (_response('{"source":"x","description":42}'), "description.*nonempty"),
    ],
)
def test_reject_invalid_or_empty_proposals(
    response: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        model.parse_proposal(response)
