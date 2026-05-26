"""
Unit tests for the tool-call extraction logic.
Run:  python -m pytest test_extraction.py -v
"""

import json
import pytest
import middleware
from middleware import (
    extract_tool_calls_from_text,
    fix_completion_response,
    rename_reasoning_in_history,
    strip_reasoning_from_history,
    _normalize_message,
)


# ===========================================================================
# Test fixtures – well-formed XML
# ===========================================================================

REASONING_WITH_SINGLE_TOOL = """\
The retrieval engine is returning concept nodes instead of episode nodes. Let me examine the graph structure:

<tool_call>
<function=bash>
<parameter=command>
python3 -c "
import sys
sys.path.insert(0, '/home/user/projects/my-app')
print('hello')
"
</parameter>
<parameter=description>
Check graph structure
</parameter>
<parameter=timeout>
30000
</parameter>
</function>
</tool_call>"""


REASONING_WITH_TWO_TOOLS = """\
Let me first check the config and then the code.

<tool_call>
<function=Read>
<parameter=file_path>/src/config.py</parameter>
</function>
</tool_call>

Now let me also read the main module:

<tool_call>
<function=Read>
<parameter=file_path>/src/main.py</parameter>
</function>
</tool_call>"""


REASONING_NO_TOOLS = """\
I think the issue is in the loop condition. The variable `i` is incremented
before the check, so it skips the last element. Let me think about this more.
"""


# ===========================================================================
# Test fixtures – MALFORMED XML (fuzzy parser targets)
# ===========================================================================

# Pattern (a): merged <tool_call> + function tag, missing < before function
MALFORMED_MERGED_OPENING = """\
Thinking about it...

<tool_call>function=edit>
<parameter=parameters>
<parameter=filePath>
/home/user/projects/my-app/tests/test_end2end.py
</parameter>
</function>
</tool_call>"""

# Pattern (b): missing </function> closing tag
MALFORMED_NO_FUNC_CLOSE = """\
Let me read the file:

<tool_call>
<function=Read>
<parameter=file_path>/src/main.py</parameter>
</tool_call>"""

# Pattern (c): missing closing > on function tag
MALFORMED_MISSING_GT = """\
Checking...

<tool_call>
<function=bash
<parameter=command>ls -la /tmp</parameter>
</function>
</tool_call>"""

# Pattern (d): wrapper <parameter=parameters> nesting (your real-world example)
MALFORMED_NESTED_PARAMS = """\
I need to edit this file.

<tool_call>
<function=edit>
<parameter=parameters>
<parameter=filePath>/src/utils.py</parameter>
<parameter=oldString>def foo():</parameter>
<parameter=newString>def bar():</parameter>
</parameter>
</function>
</tool_call>"""

# Pattern (e): multiple malformed tool calls in one reasoning block
MALFORMED_TWO_CALLS = """\
First check then fix.

<tool_call>function=Read>
<parameter=file_path>/src/config.py</parameter>
</function>
</tool_call>

Now edit:

<tool_call>function=edit>
<parameter=parameters>
<parameter=filePath>/src/config.py</parameter>
<parameter=oldString>DEBUG = False</parameter>
<parameter=newString>DEBUG = True</parameter>
</parameter>
</function>
</tool_call>"""

# Pattern (f): no closing </tool_call> at all (truncated/end of string)
MALFORMED_UNCLOSED = """\
Let me check:

<tool_call>
<function=bash>
<parameter=command>echo hello</parameter>
</function>"""

# Pattern (g): extra whitespace and newlines everywhere
MALFORMED_EXTRA_WHITESPACE = """\
Thinking...

<tool_call>

  <function=bash>

    <parameter=command>  pwd  </parameter>

  </function>

</tool_call>"""

# Pattern (h): <tools> outer tag + bare function tag <read> instead of <function=read>
# Plus mismatched closers </function></tool_call>
MALFORMED_TOOLS_BARE_TAG = """\
Let me check the current state.
<tools>
<read>
<parameter=filePath>
/home/user/projects/my-app/web-ui/index.html
</parameter>
</function>
</tool_call>"""

# Pattern (i): <tools> with bare tag and proper </tools> closer
MALFORMED_TOOLS_PROPER_CLOSE = """\
Checking the file:
<tools>
<bash>
<parameter=command>ls -la /tmp</parameter>
</bash>
</tools>"""

# Pattern (j): completely degenerate — orphaned parameter fragments, no wrapper,
# no function name.  NOT recoverable, but must not crash.
MALFORMED_ORPHANED_FRAGMENTS = """\
Ah, I see the issue. The scenario format is different - it uses "turns" as a key in the turns array. Let me fix the populate script:
</parameter>
<parameter=update>
<task_id>ses_2967a806affe6V10DYcFwmZPJo</task_id>"""


# ===========================================================================
# Tests – strict (well-formed) extraction
# ===========================================================================


class TestStrictExtraction:
    def test_single_tool_call(self):
        cleaned, calls = extract_tool_calls_from_text(REASONING_WITH_SINGLE_TOOL)
        assert len(calls) == 1
        assert calls[0]["type"] == "function"
        assert calls[0]["function"]["name"] == "bash"
        args = json.loads(calls[0]["function"]["arguments"])
        assert "python3 -c" in args["command"]
        assert args["description"] == "Check graph structure"
        assert args["timeout"] == 30000
        assert "<tool_call>" not in cleaned
        assert "Let me examine" in cleaned

    def test_two_tool_calls(self):
        cleaned, calls = extract_tool_calls_from_text(REASONING_WITH_TWO_TOOLS)
        assert len(calls) == 2
        assert calls[0]["function"]["name"] == "Read"
        assert calls[1]["function"]["name"] == "Read"
        args0 = json.loads(calls[0]["function"]["arguments"])
        assert args0["file_path"] == "/src/config.py"
        args1 = json.loads(calls[1]["function"]["arguments"])
        assert args1["file_path"] == "/src/main.py"
        assert "<tool_call>" not in cleaned

    def test_no_tool_calls(self):
        cleaned, calls = extract_tool_calls_from_text(REASONING_NO_TOOLS)
        assert len(calls) == 0
        assert cleaned == REASONING_NO_TOOLS

    def test_empty_string(self):
        cleaned, calls = extract_tool_calls_from_text("")
        assert cleaned == ""
        assert calls == []

    def test_tool_call_ids_are_unique(self):
        _, calls = extract_tool_calls_from_text(REASONING_WITH_TWO_TOOLS)
        ids = [c["id"] for c in calls]
        assert len(set(ids)) == len(ids)

    def test_extra_whitespace_still_matches_strict(self):
        cleaned, calls = extract_tool_calls_from_text(MALFORMED_EXTRA_WHITESPACE)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "bash"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["command"] == "pwd"


# ===========================================================================
# Tests – fuzzy (malformed) extraction
# ===========================================================================


class TestFuzzyExtraction:
    def test_merged_opening_tag(self):
        """<tool_call>function=edit> (missing < before function)"""
        cleaned, calls = extract_tool_calls_from_text(MALFORMED_MERGED_OPENING)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "edit"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["filePath"] == "/home/user/projects/my-app/tests/test_end2end.py"
        assert "<tool_call>" not in cleaned
        assert "Thinking about it" in cleaned

    def test_nested_wrapper_params(self):
        """<parameter=parameters> wrapping real params"""
        cleaned, calls = extract_tool_calls_from_text(MALFORMED_NESTED_PARAMS)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "edit"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["filePath"] == "/src/utils.py"
        assert args["oldString"] == "def foo():"
        assert args["newString"] == "def bar():"
        # The wrapper param "parameters" should NOT be in arguments
        assert "parameters" not in args

    def test_missing_function_close(self):
        """Missing </function> but </tool_call> present"""
        cleaned, calls = extract_tool_calls_from_text(MALFORMED_NO_FUNC_CLOSE)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "Read"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["file_path"] == "/src/main.py"

    def test_multiple_malformed_calls(self):
        """Two malformed tool calls in one reasoning block"""
        cleaned, calls = extract_tool_calls_from_text(MALFORMED_TWO_CALLS)
        assert len(calls) == 2
        assert calls[0]["function"]["name"] == "Read"
        assert calls[1]["function"]["name"] == "edit"
        args1 = json.loads(calls[1]["function"]["arguments"])
        assert args1["filePath"] == "/src/config.py"
        assert args1["oldString"] == "DEBUG = False"
        assert args1["newString"] == "DEBUG = True"
        assert "<tool_call>" not in cleaned

    def test_unclosed_tool_call(self):
        """No </tool_call> at end of string"""
        cleaned, calls = extract_tool_calls_from_text(MALFORMED_UNCLOSED)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "bash"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["command"] == "echo hello"

    def test_tools_bare_tag_mismatched_closers(self):
        """<tools><read>…</function></tool_call> — wrong outer tag + bare func tag + mismatched closers"""
        cleaned, calls = extract_tool_calls_from_text(MALFORMED_TOOLS_BARE_TAG)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "read"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["filePath"] == "/home/user/projects/my-app/web-ui/index.html"
        assert "<tools>" not in cleaned
        assert "<tool_call" not in cleaned
        assert "Let me check" in cleaned

    def test_tools_bare_tag_proper_close(self):
        """<tools><bash>…</bash></tools> — wrong outer tag + bare func tag + proper closer"""
        cleaned, calls = extract_tool_calls_from_text(MALFORMED_TOOLS_PROPER_CLOSE)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "bash"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["command"] == "ls -la /tmp"
        assert "<tools>" not in cleaned

    def test_orphaned_fragments_not_recoverable(self):
        """Orphaned param fragments with no wrapper/function — must not crash, returns nothing"""
        # Ensure flag is off for this test
        original = middleware.EMIT_NOOP_ON_ORPHAN
        middleware.EMIT_NOOP_ON_ORPHAN = False
        try:
            cleaned, calls = extract_tool_calls_from_text(MALFORMED_ORPHANED_FRAGMENTS)
            assert len(calls) == 0
            assert cleaned == MALFORMED_ORPHANED_FRAGMENTS
        finally:
            middleware.EMIT_NOOP_ON_ORPHAN = original

    def test_orphaned_fragments_noop_when_enabled(self):
        """With EMIT_NOOP_ON_ORPHAN=true, orphaned fragments produce a synthetic bash noop"""
        original = middleware.EMIT_NOOP_ON_ORPHAN
        middleware.EMIT_NOOP_ON_ORPHAN = True
        try:
            cleaned, calls = extract_tool_calls_from_text(MALFORMED_ORPHANED_FRAGMENTS)
            assert len(calls) == 1
            assert calls[0]["function"]["name"] == "bash"
            args = json.loads(calls[0]["function"]["arguments"])
            assert "malformed" in args["command"].lower()
            # Reasoning text passes through unchanged (we can't identify what to strip)
            assert cleaned == MALFORMED_ORPHANED_FRAGMENTS
        finally:
            middleware.EMIT_NOOP_ON_ORPHAN = original

    def test_ids_unique_across_fuzzy(self):
        _, calls = extract_tool_calls_from_text(MALFORMED_TWO_CALLS)
        ids = [c["id"] for c in calls]
        assert len(set(ids)) == len(ids)


# ===========================================================================
# Tests – fix_completion_response (full response object)
# ===========================================================================


class TestFixCompletionResponse:
    def _make_response(self, reasoning="", content="", tool_calls=None):
        return {
            "id": "chatcmpl-test",
            "model": "test-model",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                        "reasoning_content": reasoning,
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    def test_normalize_empty_tool_calls_array(self):
        """tool_calls: [] → null (prevents OpenCode 'Expected function.name' error)"""
        resp = self._make_response(
            reasoning="Thinking about it...\n",
            content="\n\n",
            tool_calls=[],
        )
        fixed, was_fixed = fix_completion_response(resp)
        assert was_fixed
        msg = fixed["choices"][0]["message"]
        # Empty array should become None
        assert msg["tool_calls"] is None
        # Whitespace-only content should become ""
        # Newlines (\n) are preserved because they are meaningful formatting
        assert msg["content"] == "\n\n"

    def test_normalize_does_not_touch_valid_tool_calls(self):
        """Non-empty tool_calls array should pass through untouched"""
        existing = [
            {"id": "tc-1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
        ]
        resp = self._make_response(reasoning="", content="Running...", tool_calls=existing)
        fixed, was_fixed = fix_completion_response(resp)
        # No fix needed (tool_calls are valid, content is present)
        msg = fixed["choices"][0]["message"]
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["id"] == "tc-1"

    def test_normalize_null_tool_calls_untouched(self):
        """tool_calls: null stays null"""
        resp = self._make_response(reasoning="", content="Hello!", tool_calls=None)
        fixed, was_fixed = fix_completion_response(resp)
        assert not was_fixed
        assert fixed["choices"][0]["message"]["tool_calls"] is None

    def test_fix_moves_tool_calls(self):
        resp = self._make_response(reasoning=REASONING_WITH_SINGLE_TOOL, content="")
        fixed, was_fixed = fix_completion_response(resp)
        assert was_fixed
        msg = fixed["choices"][0]["message"]
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "bash"
        assert "<tool_call>" not in msg["reasoning_content"]
        assert fixed["choices"][0]["finish_reason"] == "tool_calls"

    def test_fix_malformed_response(self):
        resp = self._make_response(reasoning=MALFORMED_MERGED_OPENING, content="")
        fixed, was_fixed = fix_completion_response(resp)
        assert was_fixed
        msg = fixed["choices"][0]["message"]
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "edit"
        assert fixed["choices"][0]["finish_reason"] == "tool_calls"

    def test_no_fix_when_clean(self):
        resp = self._make_response(reasoning=REASONING_NO_TOOLS, content="All good")
        fixed, was_fixed = fix_completion_response(resp)
        assert not was_fixed
        assert fixed["choices"][0]["message"]["tool_calls"] is None

    def test_preserves_existing_tool_calls(self):
        existing = [
            {
                "id": "existing-1",
                "type": "function",
                "function": {"name": "ls", "arguments": "{}"},
            }
        ]
        resp = self._make_response(
            reasoning=REASONING_WITH_SINGLE_TOOL, tool_calls=existing
        )
        fixed, was_fixed = fix_completion_response(resp)
        assert was_fixed
        assert len(fixed["choices"][0]["message"]["tool_calls"]) == 2
        assert fixed["choices"][0]["message"]["tool_calls"][0]["id"] == "existing-1"

    def test_malformed_response_no_reasoning(self):
        resp = self._make_response(reasoning=None, content="hello")
        fixed, was_fixed = fix_completion_response(resp)
        assert not was_fixed

    def test_orphan_noop_in_full_response(self):
        """fix_completion_response emits noop for orphaned fragments when flag is on"""
        original = middleware.EMIT_NOOP_ON_ORPHAN
        middleware.EMIT_NOOP_ON_ORPHAN = True
        try:
            resp = self._make_response(reasoning=MALFORMED_ORPHANED_FRAGMENTS, content="")
            fixed, was_fixed = fix_completion_response(resp)
            assert was_fixed
            msg = fixed["choices"][0]["message"]
            assert len(msg["tool_calls"]) == 1
            assert msg["tool_calls"][0]["function"]["name"] == "bash"
            assert fixed["choices"][0]["finish_reason"] == "tool_calls"
        finally:
            middleware.EMIT_NOOP_ON_ORPHAN = original

    def test_reasoning_only_emits_noop(self):
        """Empty content + no tool_calls + reasoning → noop when flag is on"""
        original = middleware.EMIT_NOOP_ON_ORPHAN
        middleware.EMIT_NOOP_ON_ORPHAN = True
        try:
            resp = self._make_response(
                reasoning="Let me think about this... I should read the file and check.",
                content="",
            )
            fixed, was_fixed = fix_completion_response(resp)
            assert was_fixed
            msg = fixed["choices"][0]["message"]
            assert len(msg["tool_calls"]) == 1
            assert msg["tool_calls"][0]["function"]["name"] == "bash"
            assert fixed["choices"][0]["finish_reason"] == "tool_calls"
        finally:
            middleware.EMIT_NOOP_ON_ORPHAN = original

    def test_reasoning_only_no_noop_when_flag_off(self):
        """Empty content + no tool_calls + reasoning → no noop when flag is off"""
        original = middleware.EMIT_NOOP_ON_ORPHAN
        middleware.EMIT_NOOP_ON_ORPHAN = False
        try:
            resp = self._make_response(
                reasoning="Let me think about this...",
                content="",
            )
            fixed, was_fixed = fix_completion_response(resp)
            assert not was_fixed
        finally:
            middleware.EMIT_NOOP_ON_ORPHAN = original

    def test_reasoning_with_content_no_noop(self):
        """Reasoning + non-empty content → no noop even with flag on"""
        original = middleware.EMIT_NOOP_ON_ORPHAN
        middleware.EMIT_NOOP_ON_ORPHAN = True
        try:
            resp = self._make_response(
                reasoning="Thinking about it...",
                content="Here is my answer.",
            )
            fixed, was_fixed = fix_completion_response(resp)
            assert not was_fixed
        finally:
            middleware.EMIT_NOOP_ON_ORPHAN = original

    def test_reasoning_with_existing_tool_calls_no_noop(self):
        """Reasoning + existing tool_calls → no noop even with flag on"""
        original = middleware.EMIT_NOOP_ON_ORPHAN
        middleware.EMIT_NOOP_ON_ORPHAN = True
        try:
            existing = [
                {"id": "tc-1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
            ]
            resp = self._make_response(
                reasoning="Thinking...",
                content="",
                tool_calls=existing,
            )
            fixed, was_fixed = fix_completion_response(resp)
            assert not was_fixed
            # Original tool_calls untouched
            assert len(fixed["choices"][0]["message"]["tool_calls"]) == 1
            assert fixed["choices"][0]["message"]["tool_calls"][0]["id"] == "tc-1"
        finally:
            middleware.EMIT_NOOP_ON_ORPHAN = original


# ===========================================================================
# Tests – strip_reasoning_from_history (request preprocessor)
# ===========================================================================


class TestStripReasoningHistory:
    def test_strips_reasoning_from_assistant_messages(self):
        body = {
            "model": "qwen3.5",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi!"},
                {
                    "role": "assistant",
                    "content": "Hello! How can I help?",
                    "reasoning_content": "User greeted me, respond briefly.",
                },
                {"role": "user", "content": "Read my file"},
                {
                    "role": "assistant",
                    "content": "Sure, let me read it.",
                    "reasoning_content": "I should use the Read tool.",
                },
            ],
        }
        count = strip_reasoning_from_history(body)
        assert count == 2
        # Assistant messages should no longer have reasoning_content
        for msg in body["messages"]:
            assert "reasoning_content" not in msg
        # Content should be preserved
        assert body["messages"][2]["content"] == "Hello! How can I help?"
        assert body["messages"][4]["content"] == "Sure, let me read it."

    def test_does_not_touch_user_or_system_messages(self):
        body = {
            "messages": [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "Hello"},
            ],
        }
        count = strip_reasoning_from_history(body)
        assert count == 0
        assert len(body["messages"]) == 2

    def test_handles_assistant_without_reasoning(self):
        body = {
            "messages": [
                {"role": "assistant", "content": "No reasoning here"},
            ],
        }
        count = strip_reasoning_from_history(body)
        assert count == 0
        assert body["messages"][0]["content"] == "No reasoning here"

    def test_handles_empty_messages(self):
        body = {"messages": []}
        count = strip_reasoning_from_history(body)
        assert count == 0

    def test_handles_no_messages_key(self):
        body = {"model": "test"}
        count = strip_reasoning_from_history(body)
        assert count == 0

    def test_preserves_other_assistant_fields(self):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Let me run that.",
                    "reasoning_content": "Thinking...",
                    "tool_calls": [
                        {
                            "id": "tc-1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        }
                    ],
                },
            ],
        }
        strip_reasoning_from_history(body)
        msg = body["messages"][0]
        assert "reasoning_content" not in msg
        assert msg["content"] == "Let me run that."
        assert len(msg["tool_calls"]) == 1


class TestRenameReasoningHistory:
    def test_renames_reasoning_content_on_assistant_messages(self):
        body = {
            "messages": [
                {"role": "system", "content": "System prompt"},
                {
                    "role": "assistant",
                    "content": "Hello!",
                    "reasoning_content": "User greeted me.",
                },
                {"role": "user", "content": "Next request"},
                {
                    "role": "assistant",
                    "content": "Let me check.",
                    "reasoning_content": "Need to inspect the file.",
                },
            ],
        }

        count = rename_reasoning_in_history(body)

        assert count == 2
        assert "reasoning_content" not in body["messages"][1]
        assert body["messages"][1]["reasoning"] == "User greeted me."
        assert "reasoning_content" not in body["messages"][3]
        assert body["messages"][3]["reasoning"] == "Need to inspect the file."
        assert "reasoning" not in body["messages"][0]
        assert "reasoning" not in body["messages"][2]

    def test_does_not_touch_assistant_message_that_already_has_reasoning(self):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Done.",
                    "reasoning": "Existing reasoning.",
                    "reasoning_content": "Alternate reasoning.",
                },
            ],
        }

        count = rename_reasoning_in_history(body)

        assert count == 0
        assert body["messages"][0]["reasoning"] == "Existing reasoning."
        assert body["messages"][0]["reasoning_content"] == "Alternate reasoning."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
