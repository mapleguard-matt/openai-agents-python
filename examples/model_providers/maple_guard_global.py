import asyncio
import os
import uuid

from openai import AsyncOpenAI

from agents import (
    Agent,
    Runner,
    function_tool,
    set_default_openai_client,
    set_tracing_disabled,
)

MGAI_BASE_URL = os.getenv("MGAI_BASE_URL") or ""
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""  # passed through by MGAI to the upstream provider
MGAI_API_KEY = os.getenv("MGAI_API_KEY") or ""
MGAI_MODEL = os.getenv("MGAI_MODEL") or ""

if not MGAI_BASE_URL or not OPENAI_API_KEY or not MGAI_API_KEY or not MGAI_MODEL:
    raise ValueError(
        "Please set MGAI_BASE_URL, OPENAI_API_KEY, MGAI_API_KEY, MGAI_MODEL via env vars.\n"
        "  MGAI_BASE_URL   — e.g. https://api.lab.mapleguarddemo.com/api/openai/v1\n"
        "  OPENAI_API_KEY  — your OpenAI key; sent as Authorization bearer, passed through by MGAI\n"
        "  MGAI_API_KEY    — X-MG-API-Key value (mg-...); identifies your tenant to Maple Guard\n"
        "  MGAI_MODEL      — model name as configured in MGAI (e.g. gpt-5.2)\n"
        "Optional:\n"
        "  MGAI_SOURCE     — X-MG-SOURCE value (default: openai-agents-python)\n"
        "  MGAI_REFS       — X-MG-Refs JSON string (e.g. {\"tenant.workflow_id\": \"<uuid>\"})\n"
        "  MGAI_FAIL_OPEN  — X-MG-Fail-Open value (default: true)\n"
        "  MGAI_STORE      — X-MG-Store-Content value (default: true)\n"
    )

"""Routes all agent requests through the Maple Guard AI Gateway.

MGAI implements the OpenAI Responses API contract, so no chat_completions
fallback is needed. Auth uses two separate credentials: your OpenAI API key
sent as the standard Authorization bearer (MGAI passes it through to the
upstream provider), and an MGAI-issued key in X-MG-API-Key that identifies
and validates your tenant at the gateway. X-MG-Provider-API-Key is left
empty since the OpenAI key is already supplied via Authorization.

Steps:
1. Build an AsyncOpenAI client pointed at the MGAI base URL with MGAI headers.
2. Set it as the global default client (not used for tracing).
3. Disable tracing since we're not connecting to platform.openai.com.
"""

client = AsyncOpenAI(
    base_url=MGAI_BASE_URL,
    api_key=OPENAI_API_KEY,
    default_headers={
        "X-MG-API-Key":          MGAI_API_KEY,
        "X-MG-SOURCE":           os.getenv("MGAI_SOURCE", "openai-agents-python"),
        "X-MG-Refs":             os.getenv("MGAI_REFS", "{}"),
        "X-MG-Provider-API-Key": "",
        "X-MG-Fail-Open":        os.getenv("MGAI_FAIL_OPEN", "true"),
        "X-MG-Store-Content":    os.getenv("MGAI_STORE", "true"),
    },
)
set_default_openai_client(client=client, use_for_tracing=False)
set_tracing_disabled(disabled=True)


@function_tool
def get_weather(city: str):
    print(f"[debug] getting weather for {city}")
    return f"The weather in {city} is sunny."


async def run_agent(agent, prompt):
    client._custom_headers["X-MG-Run-ID"] = str(uuid.uuid4())
    return await Runner.run(agent, prompt, auto_previous_response_id=True)


async def main():
    agent = Agent(
        name="Assistant",
        instructions="You only respond in haikus.",
        model=MGAI_MODEL,
        tools=[get_weather],
    )

    result = await run_agent(agent, "What's the weather in Tokyo?")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
