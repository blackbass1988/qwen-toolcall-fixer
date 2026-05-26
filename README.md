# qwen-toolcall-fixer

A transparent OpenAI-compatible proxy that fixes multiple Qwen3.5 bugs where tool calls end up malformed or misplaced, causing agentic coding tools to stall.

Sits between your clients (Claude Code, OpenCode, or any OpenAI-compatible client) and an upstream OpenAI-compatible API (LiteLLM, vLLM, etc.).

## The Problem

Qwen3.5 on vLLM (and other backends) intermittently produces broken tool-call responses in several ways:

**1. Tool calls inside reasoning block** — the model emits `<tool_call>` XML inside `reasoning_content` instead of the proper `tool_calls` field:

```json
{
  "reasoning_content": "Let me check...\n<tool_call>\n<function=bash>...",
  "tool_calls": null
}
```

**2. Malformed XML** — the tool-call XML itself is broken in various ways: merged tags (`<tool_call>function=edit>`), wrong outer tags (`<tools>` instead of `<tool_call>`), bare function tags (`<read>` instead of `<function=read>`), nested wrapper params, missing closing tags, etc.

**3. Empty tool_calls arrays** — the model returns `tool_calls: []` (empty array) instead of `null`, causing clients like OpenCode to crash with "Expected 'function.name' to be a string".

**4. Reasoning-only responses** — the model produces `reasoning_content` but no `content` or `tool_calls`, silently stalling the agentic loop.

## What It Does

| Feature | Default | Description |
|---|---|---|
| **Tool-call extraction** | Always on | Two-tier parser (strict + fuzzy) moves tool calls from `reasoning_content` to `tool_calls` |
| **Response normalization** | Always on | Cleans `tool_calls: []` to `null`, whitespace-only content to `""` |
| **Strip reasoning history** | `true` | Removes `reasoning_content` from assistant messages in request history to save context window |
| **Rename reasoning history** | `false` | Renames assistant `reasoning_content` to `reasoning` before forwarding upstream |
| **Noop fallback** | `false` | Emits synthetic bash tool call on unrecoverable responses to keep the agent loop alive |

## Architecture

```
  Any OpenAI-compatible client
  (Claude Code, OpenCode, etc.)
            │
            ▼
  ┌──────────────────────┐
  │  qwen-toolcall-fixer │  :4001
  │    (this middleware)  │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │   Upstream API       │  :4000
  │ (LiteLLM, vLLM, etc)│
  └──────────────────────┘
```

## Quick Start

### Docker Compose (recommended)

```bash
git clone https://github.com/czerwiakowskim/qwen-toolcall-fixer.git
cd qwen-toolcall-fixer

# Set your upstream API URL (default: http://host.docker.internal:4000)
export LITELLM_BASE_URL=http://your-api-host:4000

docker compose up -d
docker compose logs -f
```

### Bare metal

```bash
pip install -r requirements.txt
LITELLM_BASE_URL=http://localhost:4000 python middleware.py
```

### With uvicorn (multiple workers)

```bash
LITELLM_BASE_URL=http://localhost:4000 uvicorn middleware:app \
  --host 0.0.0.0 --port 4001 --workers 2
```

### systemd service

Create `/etc/systemd/system/qwen-toolcall-fixer.service`:

```ini
[Unit]
Description=Qwen Tool-Call Fixer Middleware
After=network.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/opt/qwen-toolcall-fixer
Environment=LITELLM_BASE_URL=http://localhost:4000
Environment=MIDDLEWARE_PORT=4001
Environment=LOG_LEVEL=INFO
ExecStart=/usr/bin/python3 middleware.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now qwen-toolcall-fixer
```

### Same Docker network as upstream

If your upstream API runs in Docker, uncomment the network section in `docker-compose.yml` and set:

```bash
LITELLM_BASE_URL=http://litellm:4000  # use container name
```

## Configuration

All via environment variables:

| Variable | Default | Description |
|---|---|---|
| `LITELLM_BASE_URL` | `http://localhost:4000` | Upstream OpenAI-compatible API URL |
| `MIDDLEWARE_PORT` | `4001` | Port this middleware listens on |
| `REQUEST_TIMEOUT` | `600` | Upstream request timeout in seconds |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `STRIP_REASONING_HISTORY` | `true` | Strip `reasoning_content` from assistant messages in request history to save context window. Set to `false` to disable |
| `RENAME_REASONING_HISTORY` | `false` | Rename assistant `reasoning_content` to `reasoning` before forwarding upstream. When enabled, this takes precedence over stripping |
| `EMIT_NOOP_ON_ORPHAN` | `false` | Emit synthetic bash noop tool call when model produces reasoning-only responses or unrecoverable fragments. Keeps agent loop alive. Set to `true` to enable |

## Client Configuration

Point your client at the middleware instead of your upstream API.

### Claude Code

In `~/.claude/settings.json` or project-level `.claude/settings.json`:

```json
{
  "env": {
    "OPENAI_BASE_URL": "http://localhost:4001/v1",
    "OPENAI_API_KEY": "your-api-key"
  }
}
```

Or via environment:

```bash
export OPENAI_BASE_URL=http://localhost:4001/v1
export OPENAI_API_KEY=your-api-key
claude --model your-qwen-model
```

### OpenCode

Set the API base in your OpenCode config to:

```
http://localhost:4001/v1
```

### OpenAI SDK (generic)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4001/v1",
    api_key="your-api-key",
)
```

## How It Works

### Tool-call extraction (two-tier)

**Tier 1 — Strict regex** for well-formed `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>` XML. Fast path, zero overhead on clean responses.

**Tier 2 — Fuzzy parser** kicks in when strict finds nothing but tool-call markers are detected. Handles:
- Merged tags: `<tool_call>function=edit>` (missing `<` before function)
- Wrong outer tags: `<tools>` instead of `<tool_call>`
- Bare function tags: `<read>` instead of `<function=read>`
- Wrapper params: `<parameter=parameters>` nesting around real params
- Missing closing tags: `</function>`, `</tool_call>`
- Mismatched closers: `<tools>...</tool_call>`
- Unclosed blocks at end of string

Uses a leaf-only parameter regex with negative lookahead to skip wrapper params and extract only the innermost (real) parameters.

### Response normalization (always on)

- `tool_calls: []` (empty array) → `null` — fixes OpenCode crash
- Whitespace-only `content: "\n\n"` → `""` — clean up noise

### Streaming

Streaming responses are buffered, fixed if needed, then re-emitted. The buffering adds latency equal to generation time, which is acceptable for local models and guarantees the response is clean.

### Noop fallback

When `EMIT_NOOP_ON_ORPHAN=true`, if the model produces:
- Orphaned `<parameter=` fragments without any wrapper/function (unrecoverable)
- Reasoning-only responses with empty `content` and no `tool_calls`

...the middleware injects a synthetic `bash` tool call that echoes a diagnostic message. The agent executes it, sees the warning, and gets a chance to self-correct on the next turn instead of silently stalling.

## Health Check

```bash
curl http://localhost:4001/health
# {"status":"ok","upstream":"http://localhost:4000"}
```

## Testing

```bash
pip install -r requirements.txt
python -m pytest test_extraction.py -v
```

59 tests covering all known malformation patterns, normalization, history stripping/renaming, and noop emission.

## Pass-through

All endpoints besides `/v1/chat/completions` and `/chat/completions` are proxied transparently to the upstream API (models listing, embeddings, etc.).

## Troubleshooting

**Middleware starts but clients get connection errors:** Check that `LITELLM_BASE_URL` is reachable from where the middleware runs. Inside Docker, use `host.docker.internal` for host services.

**Tool calls still not detected:** Set `LOG_LEVEL=DEBUG` and check logs. If Qwen produces a new malformation pattern not yet covered, open an issue with the raw response JSON.

**High latency on streaming:** Expected — streaming responses are buffered to allow fixing reasoning content. The added latency is the model's generation time.

## License

MIT
