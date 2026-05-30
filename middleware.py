"""
Qwen Tool-Call Fixer Middleware
===============================
A transparent OpenAI-compatible proxy that fixes multiple Qwen3.5 tool-call bugs:
- Tool calls emitted inside reasoning_content instead of the tool_calls field
- Malformed tool-call XML (merged tags, wrong wrappers, bare function tags, etc.)
- Empty tool_calls arrays that crash clients
- Reasoning-only responses that stall agentic loops

Sits between any OpenAI-compatible client and an upstream API (LiteLLM, vLLM, etc.).
Handles both streaming (SSE) and non-streaming responses.

See README.md for full documentation.
"""

import os
import re
import json
import uuid
import logging
import time
from typing import Optional

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import httpx

VLLM_ERROR_TYPES = ()
try:
    from vllm.exceptions import VLLMValidationError
    VLLM_ERROR_TYPES = (VLLMValidationError,)
except ImportError:
    pass
try:
    from litellm.exceptions import (
        BadRequestError,
        ContextWindowExceededError,
        ExceptionType,
    )
    _LITELM_ERROR_TYPES = (
        BadRequestError,
        ContextWindowExceededError,
    )
except ImportError:
    _LITELM_ERROR_TYPES = ()


def _is_context_length_error(exc: BaseException) -> bool:
    """Return True if exc is a context length exceeded validation error."""
    msg = str(exc).lower()
    if VLLM_ERROR_TYPES and isinstance(exc, VLLM_ERROR_TYPES):
        return True
    if _LITELM_ERROR_TYPES and isinstance(exc, _LITELM_ERROR_TYPES):
        return True
    if "maximum context length" in msg or "context_length_exceeded" in msg or "exceed" in msg:
        return True
    return False


def _is_context_length_message(msg: str) -> bool:
    """Return True if the error message indicates a context-length problem."""
    lowered = msg.lower()
    return (
        "maximum context length" in lowered
        or "context_length_exceeded" in lowered
        or "context window" in lowered
        or ("exceed" in lowered and "token" in lowered)
    )


def _normalize_upstream_error(error_payload: dict) -> dict:
    """
    Convert upstream context-length errors into the standard OpenAI shape.

    vLLM / LiteLLM sometimes emit:
        type: "BadRequestError", code: 400
    OpenCode (and other clients) expect:
        type: "context_length_exceeded", code: "context_length_exceeded"
    """
    if not isinstance(error_payload, dict):
        return error_payload

    msg = error_payload.get("message", "")
    if _is_context_length_message(msg):
        error_payload = dict(error_payload)  # shallow copy
        error_payload["type"] = "context_length_exceeded"
        error_payload["code"] = "context_length_exceeded"
        error_payload.setdefault("param", "messages")
    return error_payload


def _error_body_from_exc(exc: BaseException) -> str:
    """Extract a human-readable error message from an exception."""
    detail = str(exc)
    if not detail:
        detail = type(exc).__name__
    if VLLM_ERROR_TYPES and isinstance(exc, VLLM_ERROR_TYPES):
        return detail
    return detail


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
LISTEN_PORT = int(os.environ.get("MIDDLEWARE_PORT", "4001"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "600"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
STRIP_REASONING_HISTORY = os.environ.get("STRIP_REASONING_HISTORY", "true").lower() in ("true", "1", "yes")
RENAME_REASONING_HISTORY = os.environ.get("RENAME_REASONING_HISTORY", "false").lower() in ("true", "1", "yes")
EMIT_NOOP_ON_ORPHAN = os.environ.get("EMIT_NOOP_ON_ORPHAN", "false").lower() in ("true", "1", "yes")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("qwen-toolcall-fixer")

# ---------------------------------------------------------------------------
# Tool-call extraction from reasoning content
#
# Two-tier approach:
#   1. STRICT regex – fast path for well-formed XML
#   2. FUZZY parser – fallback for the many ways Qwen mangles the tags
#
# Known malformation patterns from Qwen3.5:
#   a) Merged opening:  <tool_call>function=edit>  (missing < before function)
#   b) Wrapper param:   <parameter=parameters><parameter=filePath>…  (nested)
#   c) Missing </function> or </tool_call> closing tags
#   d) Missing < or > on individual tags
#   e) Extra whitespace / newlines anywhere
#   f) Wrong outer tag:  <tools> instead of <tool_call>
#   g) Bare function tag: <read> instead of <function=read>
#   h) Mismatched closers: <tools>…</tool_call> or <read>…</function>
# ---------------------------------------------------------------------------

# ---- Strict patterns (well-formed) ----
STRICT_TOOL_CALL = re.compile(
    r"<tool_call>\s*<function=(\w[\w.-]*)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
# Leaf-only: won't match wrapper params containing nested <parameter= tags
STRICT_PARAM = re.compile(
    r"<parameter=(\w[\w.-]*)>\s*((?:(?!<parameter=)[\s\S])*?)\s*</parameter>",
    re.DOTALL,
)

# ---- Fuzzy patterns (fallback) ----
# Outer opening: <tool_call…> or <tools> (Qwen sometimes uses wrong tag name)
_OUTER_OPEN = r"(?:<tool_call[^>]*>|<tools\s*>)"
# Outer closing: </tool_call> or </tools> (either may appear regardless of opener)
_OUTER_CLOSE = r"(?:</tool_call>|</tools>)"

FUZZY_BLOCK_FULL = re.compile(
    _OUTER_OPEN + r"[\s\S]+?" + _OUTER_CLOSE,
    re.DOTALL,
)
FUZZY_BLOCK_INNER = re.compile(
    _OUTER_OPEN + r"([\s\S]+?)" + _OUTER_CLOSE,
    re.DOTALL,
)
# Same but anchored to end-of-string for unclosed blocks
FUZZY_BLOCK_FULL_EOT = re.compile(
    _OUTER_OPEN + r"[\s\S]+$",
    re.DOTALL,
)
FUZZY_BLOCK_INNER_EOT = re.compile(
    _OUTER_OPEN + r"([\s\S]+)$",
    re.DOTALL,
)
# Function name – three strategies in priority order:
#   1) <function=name> or function=name>  (attribute-style, possibly missing <)
#   2) <name>  (bare tag before first <parameter=)   — Qwen sometimes emits this
FUZZY_FUNC_NAME_ATTR = re.compile(r"(?:<\s*)?function\s*=\s*(\w[\w.-]*)\s*>?")
# Bare tag: a tag that is NOT parameter, function, tool_call, tools, or a closing tag
# and appears before <parameter=.  We search only the first few lines of the block.
FUZZY_FUNC_NAME_BARE = re.compile(
    r"<(\w[\w.-]*)>",
)
_BARE_TAG_SKIP = frozenset({
    "parameter", "function", "tool_call", "tools", "tool",
    "system-reminder", "system", "instructions",
})
# Parameter – LEAF-only: value must NOT contain another <parameter= tag.
# This uses a negative lookahead at each char so the regex naturally skips
# wrapper params like <parameter=parameters> and only matches innermost ones.
FUZZY_PARAM = re.compile(
    r"<parameter=(\w[\w.-]*)>\s*((?:(?!<parameter=)[\s\S])*?)\s*</parameter>",
    re.DOTALL,
)

# Quick presence check – does the text even look like it has a tool call?
HAS_TOOL_CALL_HINT = re.compile(r"<tool_call|tool_call>|<function=|<tools>", re.IGNORECASE)
# Weaker hint: orphaned <parameter= tags without any wrapper – not recoverable but worth logging
HAS_ORPHAN_PARAM_HINT = re.compile(r"<parameter=\w", re.IGNORECASE)


def _parse_param_value(raw: str):
    """Try to parse a parameter value as JSON (number, bool, null), fall back to string."""
    stripped = raw.strip()
    if stripped in ("true", "false", "null") or stripped.lstrip("-").replace(".", "", 1).isdigit():
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            pass
    return raw.strip()


def _extract_leaf_params(block: str) -> dict:
    """
    Extract parameter key-value pairs from a block, handling Qwen's nesting bug.

    The FUZZY_PARAM regex uses a negative lookahead so it naturally matches only
    leaf (innermost) parameters — wrapper params like <parameter=parameters> that
    contain nested <parameter= tags are skipped by the regex itself.
    """
    params: dict = {}
    for pm in FUZZY_PARAM.finditer(block):
        params[pm.group(1)] = _parse_param_value(pm.group(2))
    return params


def _make_tool_call_obj(func_name: str, arguments: dict) -> dict:
    return {
        "id": f"chatcmpl-tool-{uuid.uuid4().hex[:16]}",
        "type": "function",
        "function": {
            "name": func_name,
            "arguments": json.dumps(arguments),
        },
    }


_NOOP_MESSAGE = (
    "echo '[qwen-toolcall-fixer] Your previous tool call was malformed and could "
    "not be parsed. The XML structure was missing outer <tool_call> or <function=> "
    "tags. Please retry your tool call with correct formatting.'"
)


def _make_noop_tool_call() -> dict:
    """
    Create a synthetic bash tool call that prints a diagnostic warning.

    The agent harness (Claude Code / OpenCode) will execute it, producing
    output the model sees on the next turn — giving it a chance to
    self-correct instead of silently stalling.
    """
    return _make_tool_call_obj("bash", {"command": _NOOP_MESSAGE})


def _extract_strict(text: str) -> tuple[str, list[dict]]:
    """Tier 1: strict regex for well-formed tool-call XML."""
    tool_calls: list[dict] = []
    for match in STRICT_TOOL_CALL.finditer(text):
        func_name = match.group(1)
        params_block = match.group(2)
        arguments = {}
        for pm in STRICT_PARAM.finditer(params_block):
            arguments[pm.group(1)] = _parse_param_value(pm.group(2))
        tool_calls.append(_make_tool_call_obj(func_name, arguments))

    if not tool_calls:
        return text, []

    cleaned = STRICT_TOOL_CALL.sub("", text).rstrip()
    return cleaned, tool_calls


def _extract_fuzzy(text: str) -> tuple[str, list[dict]]:
    """
    Tier 2: tolerant parser for malformed tool-call XML.

    Handles merged tags, missing angle brackets, wrapper params, unclosed blocks.
    """
    tool_calls: list[dict] = []

    # Try closed blocks first, then fall back to unclosed (to end-of-string)
    full_matches = list(FUZZY_BLOCK_FULL.finditer(text))
    inner_matches = list(FUZZY_BLOCK_INNER.finditer(text))

    if not full_matches:
        full_matches = list(FUZZY_BLOCK_FULL_EOT.finditer(text))
        inner_matches = list(FUZZY_BLOCK_INNER_EOT.finditer(text))

    for inner_m in inner_matches:
        block = inner_m.group(1)

        # Find function name – strategy 1: attribute-style  function=name
        func_m = FUZZY_FUNC_NAME_ATTR.search(block)
        func_name = func_m.group(1) if func_m else None

        # Strategy 2: bare tag <name> before first <parameter=
        if not func_name:
            param_pos = block.find("<parameter=")
            search_region = block[:param_pos] if param_pos != -1 else block[:200]
            for bare_m in FUZZY_FUNC_NAME_BARE.finditer(search_region):
                candidate = bare_m.group(1)
                if candidate.lower() not in _BARE_TAG_SKIP:
                    func_name = candidate
                    break

        if not func_name:
            logger.warning("Fuzzy: found tool_call block but no function name, skipping")
            continue

        # Extract leaf parameters (skips wrapper nesting)
        arguments = _extract_leaf_params(block)

        tool_calls.append(_make_tool_call_obj(func_name, arguments))
        logger.info(
            "Fuzzy-parsed tool call: function=%s, %d param(s)",
            func_name,
            len(arguments),
        )

    if not tool_calls:
        return text, []

    # Remove matched blocks from text
    cleaned = text
    for fm in reversed(full_matches):  # reverse to preserve indices
        cleaned = cleaned[: fm.start()] + cleaned[fm.end() :]
    cleaned = cleaned.rstrip()
    return cleaned, tool_calls


def extract_tool_calls_from_text(text: str) -> tuple[str, list[dict]]:
    """
    Scan *text* for <tool_call>…</tool_call> blocks (well-formed or malformed).

    Uses strict regex first, falls back to fuzzy parser if strict finds nothing
    but tool-call markers are present.

    Returns
    -------
    cleaned : str
        The input with all tool-call blocks removed and trailing whitespace stripped.
    tool_calls : list[dict]
        OpenAI-compatible tool_call objects extracted from the blocks.
    """
    if not text:
        return text, []

    # Quick bail-out: no hint of tool calls at all
    if not HAS_TOOL_CALL_HINT.search(text):
        # Check for orphaned <parameter= tags — not recoverable as a tool call
        # but worth logging so the operator knows the model is misbehaving
        if HAS_ORPHAN_PARAM_HINT.search(text):
            logger.warning(
                "Detected orphaned <parameter= tags in reasoning without any "
                "tool_call/tools/function wrapper — cannot recover a tool call"
            )
            if EMIT_NOOP_ON_ORPHAN:
                logger.info("Emitting synthetic noop tool call to keep agent loop alive")
                return text, [_make_noop_tool_call()]
        return text, []

    # Tier 1: strict
    cleaned, calls = _extract_strict(text)
    if calls:
        logger.debug("Strict parser extracted %d tool call(s)", len(calls))
        return cleaned, calls

    # Tier 2: fuzzy
    logger.debug("Strict parser found nothing, trying fuzzy parser")
    cleaned, calls = _extract_fuzzy(text)
    if calls:
        logger.info("Fuzzy parser recovered %d tool call(s)", len(calls))
    return cleaned, calls


# ---------------------------------------------------------------------------
# Request preprocessor – handle reasoning_content in conversation history
# ---------------------------------------------------------------------------


def strip_reasoning_from_history(body: dict) -> int:
    """
    Remove ``reasoning_content`` from all assistant messages in the request's
    message history.  These tokens bloat the context window without improving
    output quality — the model doesn't benefit from seeing its own prior
    chain-of-thought.

    Returns the number of messages that were stripped.
    """
    stripped = 0
    for msg in body.get("messages", []):
        if msg.get("role") == "assistant" and "reasoning_content" in msg:
            del msg["reasoning_content"]
            stripped += 1
    if stripped:
        logger.info("Stripped reasoning_content from %d assistant message(s)", stripped)
    return stripped


def rename_reasoning_in_history(body: dict) -> int:
    """
    Rename ``reasoning_content`` to ``reasoning`` on assistant messages in the
    request history. This keeps prior assistant reasoning visible to upstreams
    that expect the ``reasoning`` field name instead of ``reasoning_content``.
    If an assistant message already has ``reasoning``, it is left unchanged.

    Returns the number of messages that were renamed.
    """
    renamed = 0
    for msg in body.get("messages", []):
        if msg.get("role") == "assistant" and "reasoning_content" in msg and "reasoning" not in msg:
            msg["reasoning"] = msg["reasoning_content"]
            del msg["reasoning_content"]
            renamed += 1
    if renamed:
        logger.info("Renamed reasoning_content to reasoning in %d assistant message(s)", renamed)
    return renamed


# ---------------------------------------------------------------------------
# Response fixers
# ---------------------------------------------------------------------------


def _is_empty_message(message: dict) -> bool:
    """True if the message has no usable content or tool_calls (only reasoning)."""
    content = (message.get("content") or "").strip()
    tool_calls = message.get("tool_calls")
    has_tool_calls = bool(tool_calls)  # None, [], or missing → False
    return not content and not has_tool_calls


def _normalize_message(message: dict) -> bool:
    """
    Normalize a response message in place.  Always runs (not behind a flag).

    Fixes:
    - ``tool_calls: []`` → ``null``  (empty array breaks OpenCode / Claude Code
      with "Expected 'function.name' to be a string")
    - whitespace-only ``content`` → ``""``

    Returns True if anything was changed.
    """
    changed = False

    # Empty tool_calls array → null
    tc = message.get("tool_calls")
    if isinstance(tc, list) and len(tc) == 0:
        message["tool_calls"] = None
        changed = True
        logger.debug("Normalized empty tool_calls [] → null")

    # Whitespace-only content (spaces/tabs, NOT newlines) → ""
    content = message.get("content")
    if isinstance(content, str) and content != "" and content.strip() == "" and "\n" not in content:
        message["content"] = ""
        changed = True
        logger.debug("Normalized whitespace-only content → empty string")

    return changed


def _normalize_streaming_chunk(chunk: dict) -> None:
    """
    Normalize a streaming SSE chunk in place.

    Fixes the same issues as _normalize_message but on delta objects:
    - ``tool_calls: []`` in delta → remove key
    - whitespace-only ``content`` in delta (spaces/tabs only, NOT newlines)
         → ``""``
    """
    for choice in chunk.get("choices", []):
        delta = choice.get("delta", {})

        tc = delta.get("tool_calls")
        if isinstance(tc, list) and len(tc) == 0:
            del delta["tool_calls"]

        content = delta.get("content")
        # Strip pure whitespace (spaces/tabs) but PRESERVE newlines (\n is meaningful
        # in markdown tables, code blocks, etc.  "\n".strip() == "" would eat them).
        if isinstance(content, str) and content != "" and content.strip() == "" and "\n" not in content:
            delta["content"] = ""


def fix_completion_response(data: dict) -> tuple[dict, bool]:
    """
    Fix a non-streaming chat completion response *in place*.

    Returns (data, was_fixed).
    """
    fixed = False
    for choice in data.get("choices", []):
        message = choice.get("message", {})

        # Always normalize (empty tool_calls, whitespace content)
        if _normalize_message(message):
            fixed = True

        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""

        has_hint = HAS_TOOL_CALL_HINT.search(reasoning) if reasoning else None
        has_orphan = HAS_ORPHAN_PARAM_HINT.search(reasoning) if reasoning else None

        if reasoning and (has_hint or has_orphan):
            cleaned, extracted = extract_tool_calls_from_text(reasoning)
            if extracted:
                message["reasoning_content"] = cleaned
                existing = message.get("tool_calls") or []
                message["tool_calls"] = existing + extracted
                choice["finish_reason"] = "tool_calls"
                fixed = True
                logger.info(
                    "Fixed %d tool call(s) moved from reasoning to tool_calls",
                    len(extracted),
                )
                continue

        # Noop fallback: reasoning-only response with empty content and no tool_calls.
        # The model was thinking but produced nothing actionable — emit a noop so the
        # agent loop doesn't stall.
        if EMIT_NOOP_ON_ORPHAN and reasoning and _is_empty_message(message):
            logger.info(
                "Reasoning-only response with empty content/tool_calls — "
                "emitting synthetic noop to keep agent loop alive"
            )
            message["tool_calls"] = [_make_noop_tool_call()]
            choice["finish_reason"] = "tool_calls"
            fixed = True

    return data, fixed


def _rebuild_streaming_chunks(
    base_id: str,
    base_model: str,
    base_created: int,
    cleaned_reasoning: str,
    content: str,
    tool_calls: list[dict],
) -> list[str]:
    """Build a list of SSE `data: …` lines for a corrected streaming response."""
    lines: list[str] = []

    def _chunk(delta: dict, finish: Optional[str] = None) -> str:
        # Compact JSON: newlines inside string values stay escaped as \n,
        # no extra spaces, single-line output so SSE parser handles it correctly.
        return json.dumps(
            {
                "id": base_id,
                "object": "chat.completion.chunk",
                "created": base_created,
                "model": base_model,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish,
                    }
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    # Role chunk
    lines.append(f"data: {_chunk({'role': 'assistant'})}\n\n")

    # Reasoning in one chunk (could split, but one is fine for correctness)
    if cleaned_reasoning:
        lines.append(f"data: {_chunk({'reasoning_content': cleaned_reasoning})}\n\n")

    # Content
    if content:
        lines.append(f"data: {_chunk({'content': content})}\n\n")

    # Tool calls – one chunk per tool call
    for idx, tc in enumerate(tool_calls):
        lines.append(
            f"data: {_chunk({'tool_calls': [{'index': idx, 'id': tc['id'], 'type': 'function', 'function': {'name': tc['function']['name'], 'arguments': tc['function']['arguments']}}]})}\n\n"
        )

    # Finish reason
    lines.append(f"data: {_chunk({}, finish='tool_calls')}\n\n")
    lines.append("data: [DONE]\n\n")
    return lines


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Qwen Tool-Call Fixer Middleware",
    description="Transparent proxy that fixes Qwen3.5 tool-call-in-reasoning bug",
)


@app.exception_handler(Exception)
async def catch_upstream_errors(request: Request, exc: BaseException):
    if _is_context_length_error(exc):
        logger.warning("Context length error: %s", _error_body_from_exc(exc))
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": _error_body_from_exc(exc),
                    "type": "context_length_exceeded",
                    "param": None,
                    "code": "context_length_exceeded",
                }
            },
        )
    raise


@app.get("/health")
async def health():
    return {"status": "ok", "upstream": LITELLM_BASE_URL}


# --------------- chat/completions (the hot path) ---------------


@app.api_route("/v1/chat/completions", methods=["POST"])
@app.api_route("/chat/completions", methods=["POST"])
async def chat_completions(request: Request):
    body = await request.json()
    is_stream = body.get("stream", False)

    # Rename takes precedence so assistant reasoning is preserved when requested.
    if RENAME_REASONING_HISTORY:
        rename_reasoning_in_history(body)
    elif STRIP_REASONING_HISTORY:
        strip_reasoning_from_history(body)

    # Forward headers (auth etc.) but drop hop-by-hop
    fwd_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding")
    }

    if is_stream:
        return await _handle_streaming(body, fwd_headers)
    else:
        return await _handle_non_streaming(body, fwd_headers)


async def _handle_non_streaming(body: dict, headers: dict) -> Response:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        upstream = await client.post(
            f"{LITELLM_BASE_URL}/v1/chat/completions",
            json=body,
            headers=headers,
        )

    # Handle error responses from upstream (including wrapped vLLM errors)
    if upstream.status_code >= 400:
        data = upstream.json()
        raw_err = data.get("error", {}) if isinstance(data, dict) else {}
        error_msg = raw_err.get("message", upstream.text) if isinstance(raw_err, dict) else upstream.text
        logger.warning("Upstream returned %d: %s", upstream.status_code, error_msg)
        normalized_err = _normalize_upstream_error(raw_err) if isinstance(raw_err, dict) else {"message": error_msg}
        return JSONResponse(
            status_code=upstream.status_code,
            content={"error": normalized_err},
        )

    data = upstream.json()

    # Also check for error responses embedded in 200 OK body (LiteLLM wrapping vLLM)
    if isinstance(data, dict) and "error" in data:
        err = data["error"]
        error_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        logger.warning("Upstream returned 200 with error in body: %s", error_msg)
        normalized_err = _normalize_upstream_error(err) if isinstance(err, dict) else {"message": error_msg}
        return JSONResponse(
            status_code=400,
            content={"error": normalized_err},
        )

    data, was_fixed = fix_completion_response(data)
    if was_fixed:
        logger.debug("Non-streaming response fixed: %s", data.get("id"))
    # Use compact JSON (no indentation) so newlines inside strings stay as \n
    return Response(
        content=json.dumps(data, ensure_ascii=False, separators=(",", ":")),
        media_type="application/json",
        status_code=upstream.status_code,
    )


async def _handle_streaming(body: dict, headers: dict) -> StreamingResponse:
    """
    Buffer the full streaming response, apply the fix if needed, then re-emit.

    Buffering is necessary because we may need to strip tool-call XML from
    reasoning_content deltas that were already emitted.  The latency cost is
    the generation time, which is bounded by the model anyway.
    """

    async def generate():
        raw_chunks: list[dict] = []
        raw_lines: list[str] = []  # fallback if no fix needed

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{LITELLM_BASE_URL}/v1/chat/completions",
                json=body,
                headers=headers,
            ) as resp:
                if resp.status_code >= 400:
                    error_text = await resp.aread()
                    error_msg = error_text.decode() if error_text else "Upstream error"
                    logger.warning("Upstream streaming returned %d: %s", resp.status_code, error_msg)
                    # Try to parse upstream JSON error; fall back to plain text
                    try:
                        err_data = json.loads(error_msg)
                        raw_err = err_data.get("error", {}) if isinstance(err_data, dict) else {}
                        if isinstance(raw_err, dict):
                            normalized_err = _normalize_upstream_error(raw_err)
                        else:
                            normalized_err = {"message": str(raw_err), "type": "upstream_error", "code": "upstream_error"}
                    except (json.JSONDecodeError, ValueError):
                        normalized_err = {"message": error_msg, "type": "upstream_error", "code": "upstream_error"}
                        if _is_context_length_message(error_msg):
                            normalized_err["type"] = "context_length_exceeded"
                            normalized_err["code"] = "context_length_exceeded"
                            normalized_err.setdefault("param", "messages")
                    yield f"data: {json.dumps({'error': normalized_err})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    raw_lines.append(line)
                    if line.startswith("data: "):
                        payload = line[6:].strip()
                        if payload == "[DONE]":
                            continue
                        try:
                            raw_chunks.append(json.loads(payload))
                        except json.JSONDecodeError:
                            pass

        # Reconstruct full reasoning + content from deltas
        full_reasoning = ""
        full_content = ""
        base_id = ""
        base_model = ""
        base_created = 0

        for chunk in raw_chunks:
            if not base_id:
                base_id = chunk.get("id", "")
                base_model = chunk.get("model", "")
                base_created = chunk.get("created", 0)
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                rc = delta.get("reasoning_content") or delta.get("reasoning")
                if rc:
                    full_reasoning += rc
                ct = delta.get("content")
                if ct:
                    full_content += ct

        # Check if fix is needed
        has_hint = HAS_TOOL_CALL_HINT.search(full_reasoning) if full_reasoning else None
        has_orphan = HAS_ORPHAN_PARAM_HINT.search(full_reasoning) if full_reasoning else None
        if full_reasoning and (has_hint or has_orphan):
            cleaned, extracted = extract_tool_calls_from_text(full_reasoning)
            if extracted:
                logger.info(
                    "Streaming response %s: fixed %d tool call(s)",
                    base_id,
                    len(extracted),
                )
                for sse_line in _rebuild_streaming_chunks(
                    base_id, base_model, base_created, cleaned, full_content, extracted
                ):
                    yield sse_line
                return

        # Noop fallback: reasoning-only stream with empty content and no tool_calls
        if EMIT_NOOP_ON_ORPHAN and full_reasoning and not full_content.strip():
            # Check if any chunk carried tool_calls deltas
            has_any_tc = any(
                "tool_calls" in choice.get("delta", {})
                for chunk in raw_chunks
                for choice in chunk.get("choices", [])
            )
            if not has_any_tc:
                logger.info(
                    "Streaming response %s: reasoning-only with empty content/tool_calls "
                    "— emitting synthetic noop",
                    base_id,
                )
                noop = _make_noop_tool_call()
                for sse_line in _rebuild_streaming_chunks(
                    base_id, base_model, base_created, full_reasoning, "", [noop]
                ):
                    yield sse_line
                return

        # No fix needed – re-emit with normalization (strip empty tool_calls etc.)
        for line in raw_lines:
            if line.startswith("data: ") and line[6:].strip() not in ("[DONE]", ""):
                try:
                    chunk = json.loads(line[6:])
                    _normalize_streaming_chunk(chunk)
                    yield f"data: {json.dumps(chunk)}\n\n"
                    continue
                except (json.JSONDecodeError, KeyError):
                    pass
            yield f"{line}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------- pass-through for all other endpoints ---------------


_HOP_BY_HOP_HEADERS = frozenset({
    "connection", "keep-alive", "transfer-encoding",
    "upgrade", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "expect",
})


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_passthrough(request: Request, path: str):
    """
    Transparently proxy everything else (models, embeddings, docs, etc.) to LiteLLM.

    Returns the upstream response with hop-by-hop headers stripped so the
    transfer-encoding / content-length conflict can't corrupt the body.
    """
    fwd_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding")
    }
    body = await request.body()

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.request(
            method=request.method,
            url=f"{LITELLM_BASE_URL}/{path}",
            headers=fwd_headers,
            content=body,
        )

    # Strip hop-by-hop headers that would corrupt the response body
    # (e.g. transfer-encoding: chunked conflicts with our raw bytes payload)
    clean_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in _HOP_BY_HOP_HEADERS
    }

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=clean_headers,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Qwen Tool-Call Fixer on :%d → %s", LISTEN_PORT, LITELLM_BASE_URL)
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT, log_level=LOG_LEVEL.lower())
