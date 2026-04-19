# OpenAI Agents SDK — Maple Guard AI Gateway Integration

## Overview

The [OpenAI Agents Python SDK](https://github.com/openai/openai-agents-python) can be routed through the Maple Guard AI (MGAI) gateway with minimal configuration. MGAI implements the OpenAI Responses API contract, so no changes to agent logic are required — only the client setup needs updating.

---

## Authentication

MGAI uses two separate credentials on every request:

| Header | Purpose |
|--------|---------|
| `Authorization: Bearer <token>` | Your OpenAI API key — MGAI passes this through to the upstream provider |
| `X-MG-API-Key: mg-...` | Your MGAI-issued tenant key — identifies and validates your organisation at the gateway |

---

## Client Setup

Create an `AsyncOpenAI` client pointed at the MGAI base URL and register it as the global default. This means all agents and runner calls route through MGAI without any per-agent configuration.

```python
import os
from openai import AsyncOpenAI
from agents import set_default_openai_client, set_tracing_disabled

client = AsyncOpenAI(
    base_url="https://api.lab.mapleguarddemo.com/api/openai/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
    default_headers={
        "X-MG-API-Key":          os.getenv("MGAI_API_KEY"),
        "X-MG-SOURCE":           "openai-agents-python",
        "X-MG-Refs":             '{"tenant.workflow_id": "<uuid>"}',
        "X-MG-Provider-API-Key": "",
        "X-MG-Fail-Open":        "true",
        "X-MG-Store-Content":    "true",
    },
)
set_default_openai_client(client=client, use_for_tracing=False)
set_tracing_disabled(disabled=True)
```

> **Note:** Do not call `set_default_openai_api("chat_completions")` — MGAI supports the Responses API natively and this is the preferred mode.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MGAI_BASE_URL` | MGAI gateway base URL (e.g. `https://api.lab.mapleguarddemo.com/api/openai/v1`) |
| `OPENAI_API_KEY` | Your OpenAI API key — passed through by MGAI to the upstream provider |
| `MGAI_API_KEY` | Your MGAI-issued tenant key (`mg-...`) |
| `MGAI_MODEL` | Model name as configured in MGAI (e.g. `gpt-5.2`) |
| `MGAI_REFS` | JSON string of routing/tracking references (e.g. `{"tenant.workflow_id": "<uuid>"}`) |

---

## Run Correlation (X-MG-Run-ID)

Multi-step agent runs generate multiple separate API calls to MGAI — one per tool call. To correlate all steps in MGAI logs, inject a fresh UUID into `X-MG-Run-ID` before each run.

**Important:** `client.default_headers` is a read-only property that returns a new dict on every access — writes to it are silently discarded. The correct internal storage to update is `client._custom_headers`:

```python
import uuid

async def run_agent(agent, prompt):
    client._custom_headers["X-MG-Run-ID"] = str(uuid.uuid4())
    return await Runner.run(agent, prompt, auto_previous_response_id=True)
```

All MGAI events within that run will share the same `X-MG-Run-ID`, making them trivially joinable in audit logs.

---

## Performance: auto_previous_response_id

By default the SDK bundles the full conversation history (all prior messages and tool results) into the `input` array of every API call. Passing `auto_previous_response_id=True` to `Runner.run()` enables server-side context management — subsequent calls only send new tool results, with OpenAI referencing prior context by response ID. This reduces input token usage significantly on multi-step runs.

| Mode | Call 1 input tokens | Call 2 input tokens |
|------|--------------------|--------------------|
| Default (full input array) | 60 | 96 (resends full history) |
| `auto_previous_response_id=True` | 60 | ~30 (tool result only) |

---

## How Multi-Step Runs Appear in MGAI Logs

Each function tool call produces a distinct MGAI request/response pair. A 4-tool agent run generates 5 MGAI events:

| MGAI Event | Input contains | Output contains |
|------------|---------------|-----------------|
| Step 1 | User message | Tool call decision |
| Step 2 | Tool result | Next tool call decision |
| Step 3 | Tool result | Next tool call decision |
| Step 4 | Tool result | Next tool call decision |
| Step 5 | Tool result | Final answer |

All 5 events share the same `X-MG-Run-ID` header, linking them as a single logical run.

---

## Hosted Tools vs Function Tools

The SDK supports two categories of tools with importantly different gateway behaviour:

### Function Tools (`@function_tool`)
- Execute **locally in your infrastructure**
- Each tool call produces a **distinct MGAI round trip**
- MGAI has full visibility and can apply policy at every step
- Tool execution and results never leave your environment

### Hosted Tools (`WebSearchTool`, `FileSearchTool`, `CodeInterpreterTool`)
- Execute **inside OpenAI's infrastructure**
- All tool calls and the final response are returned in a **single MGAI round trip**
- MGAI captures the tool call records and results in the response body but cannot intercept execution between steps
- Tool execution and any data passed to tools crosses OpenAI's servers

**Recommendation:** For enterprise use cases where auditability, data sovereignty, and per-step policy enforcement matter, function tools are the preferred pattern — tool execution stays entirely within your own infrastructure and every reasoning step is individually visible to MGAI.

---

## Example Files

All examples are in `examples/model_providers/` in the fork at [`mapleguard-matt/openai-agents-python`](https://github.com/mapleguard-matt/openai-agents-python):

| File | Purpose |
|------|---------|
| `maple_guard_global.py` | Minimal smoke test — single tool call, confirms end-to-end connectivity |
| `maple_guard_multistep.py` | 4-step sequential function tool chain — verifies run correlation across steps |
| `maple_guard_pe_analysis.py` | PE/IPO readiness analysis — realistic demo for financial services audiences |
| `maple_guard_web_search.py` | Hosted `WebSearchTool` — demonstrates the hosted vs function tool distinction |

### Running the Examples

```powershell
# Set environment variables
$env:MGAI_BASE_URL   = "https://api.lab.mapleguarddemo.com/api/openai/v1"
$env:OPENAI_API_KEY  = "sk-..."
$env:MGAI_API_KEY    = "mg-..."
$env:MGAI_MODEL      = "gpt-5.2"
$env:MGAI_REFS       = '{"tenant.workflow_id": "<uuid>"}'

# Run an example
uv run examples/model_providers/maple_guard_pe_analysis.py
```
