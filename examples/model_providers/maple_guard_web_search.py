import asyncio
import os
import uuid

from openai import AsyncOpenAI

from agents import (
    Agent,
    ModelSettings,
    Runner,
    WebSearchTool,
    set_default_openai_client,
    set_tracing_disabled,
)

MGAI_BASE_URL = os.getenv("MGAI_BASE_URL") or ""
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
MGAI_API_KEY = os.getenv("MGAI_API_KEY") or ""
MGAI_MODEL = os.getenv("MGAI_MODEL") or ""

if not MGAI_BASE_URL or not OPENAI_API_KEY or not MGAI_API_KEY or not MGAI_MODEL:
    raise ValueError(
        "Please set MGAI_BASE_URL, OPENAI_API_KEY, MGAI_API_KEY, MGAI_MODEL via env vars."
    )

"""Two-step web search using OpenAI's hosted WebSearchTool routed through MGAI.

Unlike function tools (which execute locally and generate one MGAI round trip per
tool call), hosted tools like WebSearchTool execute server-side inside OpenAI's
infrastructure. This means tool execution itself is invisible to MGAI — it sees
only the LLM inference calls. Contrast this with maple_guard_pe_analysis.py where
each local function tool produces a distinct MGAI round trip.

The two searches are sequenced so the second query is informed by the first result,
demonstrating that the model reasons across multiple search steps before answering.

Expected MGAI sequence (all sharing the same X-MG-Run-ID):
  Step 1 — model searches for recent IPO market conditions, synthesises findings
  Step 2 — model searches for sector-specific comparable transactions, synthesises
  Step 3 — model delivers final research brief
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


async def run_agent(agent, prompt):
    run_id = str(uuid.uuid4())
    client._custom_headers["X-MG-Run-ID"] = run_id
    print(f"[run] X-MG-Run-ID: {run_id}\n")
    return await Runner.run(agent, prompt, auto_previous_response_id=True)


async def main():
    agent = Agent(
        name="IPOMarketResearcher",
        instructions=(
            "You are a senior equity capital markets analyst. When asked to research IPO "
            "conditions for a sector, you MUST perform exactly two web searches in sequence:\n"
            "1. Search for the current IPO market outlook and recent listings in the given sector.\n"
            "2. Using a specific company or trend identified in the first search, perform a second "
            "targeted search to find valuation multiples or comparable transaction data.\n\n"
            "After both searches, deliver a concise two-section research brief:\n"
            "  Market Conditions — key findings from search 1\n"
            "  Comparable Valuations — key findings from search 2\n"
            "End with a one-sentence IPO window assessment. "
            "Be definitive — do not ask follow-up questions."
        ),
        model=MGAI_MODEL,
        tools=[WebSearchTool()],
        model_settings=ModelSettings(parallel_tool_calls=False),
    )

    result = await run_agent(
        agent,
        "Research current IPO market conditions and comparable valuations for "
        "enterprise SaaS companies.",
    )
    print(f"\n{'='*60}\nIPO MARKET RESEARCH BRIEF\n{'='*60}\n{result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
