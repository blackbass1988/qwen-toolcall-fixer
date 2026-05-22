"""
Test that newlines are preserved throughout the streaming and extraction pipeline.

The root cause was _normalize_streaming_chunk treating "\n" as whitespace-only
content and replacing it with "":  "\n".strip() == "" → True → content cleared.
All markdown table newlines were silently eaten in streaming mode.
"""

import json
import pytest
import middleware
from middleware import (
    _normalize_streaming_chunk,
    _normalize_message,
    extract_tool_calls_from_text,
)


class TestNormalizeStreamingChunk:
    """_normalize_streaming_chunk must preserve \\n in content deltas."""

    def test_preserves_single_newline(self):
        """A lone \\n in a delta must survive normalization."""
        chunk = {"choices": [{"delta": {"content": "\n"}}]}
        _normalize_streaming_chunk(chunk)
        assert chunk["choices"][0]["delta"]["content"] == "\n"

    def test_preserves_double_newline(self):
        """Double \\n\\n (paragraph break) must survive."""
        chunk = {"choices": [{"delta": {"content": "\n\n"}}]}
        _normalize_streaming_chunk(chunk)
        assert chunk["choices"][0]["delta"]["content"] == "\n\n"

    def test_preserves_newline_between_text(self):
        """Text with embedded newlines must be preserved."""
        chunk = {"choices": [{"delta": {"content": "line1\nline2\nline3"}}]}
        _normalize_streaming_chunk(chunk)
        assert chunk["choices"][0]["delta"]["content"] == "line1\nline2\nline3"

    def test_preserves_markdown_table_row(self):
        """Markdown table row separators (\\n) must survive."""
        chunk = {"choices": [{"delta": {"content": "\n"}}]}
        _normalize_streaming_chunk(chunk)
        assert chunk["choices"][0]["delta"]["content"] == "\n"

    def test_still_strips_spaces_only(self):
        """Pure spaces should still be stripped (unlike \\n)."""
        chunk = {"choices": [{"delta": {"content": "   "}}]}
        _normalize_streaming_chunk(chunk)
        assert chunk["choices"][0]["delta"]["content"] == ""

    def test_stills_strips_tabs_only(self):
        """Pure tabs should still be stripped."""
        chunk = {"choices": [{"delta": {"content": "\t\t"}}]}
        _normalize_streaming_chunk(chunk)
        assert chunk["choices"][0]["delta"]["content"] == ""

    def test_preserves_spaces_with_text(self):
        """Spaces within text must not be stripped."""
        chunk = {"choices": [{"delta": {"content": "  hello  "}}]}
        _normalize_streaming_chunk(chunk)
        assert chunk["choices"][0]["delta"]["content"] == "  hello  "

    def test_strips_empty_tool_calls(self):
        """tool_calls: [] must be removed (existing behaviour)."""
        chunk = {"choices": [{"delta": {"tool_calls": [], "content": "hi"}}]}
        _normalize_streaming_chunk(chunk)
        assert "tool_calls" not in chunk["choices"][0]["delta"]
        assert chunk["choices"][0]["delta"]["content"] == "hi"

    def test_preserves_newline_alongside_tool_calls(self):
        """Newline content adjacent to tool_calls: [] must survive."""
        chunk = {"choices": [{"delta": {"tool_calls": [], "content": "\n\n"}}]}
        _normalize_streaming_chunk(chunk)
        assert "tool_calls" not in chunk["choices"][0]["delta"]
        # The newline is preserved, not stripped to ""
        assert chunk["choices"][0]["delta"]["content"] == "\n\n"

    def test_no_content_key_unchanged(self):
        """Chunks without content key should be untouched."""
        chunk = {"choices": [{"delta": {"role": "assistant"}}]}
        _normalize_streaming_chunk(chunk)
        assert chunk["choices"][0]["delta"]["role"] == "assistant"
        assert "content" not in chunk["choices"][0]["delta"]

    def test_empty_content_unchanged(self):
        """Empty string content must stay empty."""
        chunk = {"choices": [{"delta": {"content": ""}}]}
        _normalize_streaming_chunk(chunk)
        assert chunk["choices"][0]["delta"]["content"] == ""


class TestNormalizeMessage:
    """_normalize_message must preserve \\n in content (non-streaming path)."""

    def test_preserves_newlines(self):
        """Newlines-only message content must survive."""
        msg = {"role": "assistant", "content": "\n\n", "tool_calls": []}
        _normalize_message(msg)
        assert msg["content"] == "\n\n"
        assert msg["tool_calls"] is None  # empty array still normalised

    def test_preserves_newlines_with_text(self):
        """Content with embedded newlines must survive."""
        msg = {"role": "assistant", "content": "line1\nline2\nline3"}
        _normalize_message(msg)
        assert msg["content"] == "line1\nline2\nline3"

    def test_strips_spaces_only(self):
        """Pure spaces should still be stripped."""
        msg = {"role": "assistant", "content": "   "}
        _normalize_message(msg)
        assert msg["content"] == ""

    def test_strips_tabs_only(self):
        """Pure tabs should still be stripped."""
        msg = {"role": "assistant", "content": "\t\t"}
        _normalize_message(msg)
        assert msg["content"] == ""

    def test_whitespace_and_newlines_preserved(self):
        """Mix of spaces and newlines: \\n is meaningful, so preserved."""
        msg = {"role": "assistant", "content": "  \n  \n  "}
        _normalize_message(msg)
        assert msg["content"] == "  \n  \n  "

    def test_empty_content_unchanged(self):
        """Empty string must stay empty."""
        msg = {"role": "assistant", "content": ""}
        _normalize_message(msg)
        assert msg["content"] == ""


class TestExtractionNewlinePreservation:
    """extract_tool_calls_from_text must preserve newlines around tool calls."""

    TABLE_WITH_TOOL_CALL = (
        "Here is your table:\n\n"
        "| Name | Age |\n| --- | --- |\n| Alice | 25 |\n\n"
        "<tool_call>\n"
        "<function=Read>\n"
        "<parameter=filePath>/tmp/test.txt</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )

    def test_table_newlines_preserved_after_extraction(self):
        """Newlines in the table before the tool call must survive extraction."""
        cleaned, calls = extract_tool_calls_from_text(self.TABLE_WITH_TOOL_CALL)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "Read"
        # The markdown table part must keep its newlines
        assert "| Name | Age |" in cleaned
        assert "| Alice | 25 |" in cleaned
        # The newlines between rows must be intact
        assert "\n" in cleaned
        # The tool call XML must be removed
        assert "<tool_call>" not in cleaned
        assert "<function=" not in cleaned

    def test_leading_newlines_preserved(self):
        """Newlines before a tool call must survive extraction."""
        text = "\n\nStart here\n<tool_call>\n<function=bash>\n<parameter=command>pwd</parameter>\n</function>\n</tool_call>"
        cleaned, calls = extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert "Start here" in cleaned
        # Leading newlines are preserved (only trailing stripped)
        assert cleaned.startswith("\n\n")

    def test_trailing_newlines_preserved(self):
        """Newlines after a tool call must survive extraction."""
        text = (
            "End here"
            "\n\n"
            "<tool_call>\n"
            "<function=Read>\n"
            "<parameter=path>/x</parameter>\n"
            "</function>\n"
            "</tool_call>"
            "\n\n\n\n"
        )
        cleaned, calls = extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert "End here" in cleaned


class TestRebuildStreamingChunks:
    """_rebuild_streaming_chunks output must have \\n inside JSON strings."""

    def _parse_sse_for_content(self, sse_lines: list[str]) -> str:
        """Helper: join content deltas from SSE-like output lines."""
        full = ""
        for line in sse_lines:
            line = line.strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                continue
            try:
                chunk = json.loads(payload)
                for c in chunk.get("choices", []):
                    ct = c.get("delta", {}).get("content", "")
                    full += ct
            except json.JSONDecodeError:
                pass
        return full

    def test_markdown_table_newlines_preserved(self):
        """A markdown table in content must keep its \\n through rebuild."""
        content = "| Name | Age | City |\n| --- | --- | --- |\n| Alice | 30 | NYC |\n| Bob | 25 | LA |"
        lines = middleware._rebuild_streaming_chunks(
            "test-id", "test-model", 100,
            "",  # no reasoning
            content,
            [],  # no tool calls
        )
        result = self._parse_sse_for_content(lines)
        assert result == content
        assert "\n" in result
        assert result.count("\n") == 3  # header + 2 data rows

    def test_reasoning_and_content_newlines(self):
        """Newlines in both reasoning and content must survive rebuild."""
        reasoning = "Step 1\nStep 2\nStep 3"
        content = "Result:\n| A | B |\n| - | - |\n| 1 | 2 |"
        lines = middleware._rebuild_streaming_chunks(
            "test-id", "test-model", 100,
            reasoning,
            content,
            [],
        )
        result = self._parse_sse_for_content(lines)
        assert result == content
        assert "\n" in result

    def test_sse_json_is_single_line(self):
        """Each SSE data: line must be a single line (no embedded newlines)."""
        content = "Hello\nWorld\n\nFoo\nBar"
        lines = middleware._rebuild_streaming_chunks(
            "test-id", "test-model", 100,
            "thinking...",
            content,
            [],
        )
        for line in lines:
            # Every line should end with \n\n and be a single JSON line
            assert line.endswith("\n\n"), f"Line does not end with \\n\\n: {repr(line)}"
            data_part = line.rstrip("\n")
            if data_part.startswith("data: "):
                payload = data_part[6:]
                if payload == "[DONE]":
                    continue
                # Verify the JSON is parseable (no embedded newlines broke it)
                try:
                    json.loads(payload)
                except json.JSONDecodeError as e:
                    pytest.fail(f"SSE JSON not parseable: {e}\n  line={repr(line)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
