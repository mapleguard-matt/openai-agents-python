import asyncio
import os
import uuid

from openai import AsyncOpenAI

from agents import (
    Agent,
    ModelSettings,
    Runner,
    function_tool,
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

"""Multi-step tool chain to verify X-MG-Run-ID correlation across 4 MGAI round trips.

Each tool depends on the previous tool's output, forcing the model to call them
sequentially (not in parallel). With auto_previous_response_id=True, each step
only sends the new tool result rather than the full conversation history.

Expected MGAI sequence per run:
  Step 1 — input: user message             → output: call get_weather
  Step 2 — input: weather result           → output: call get_forecast
  Step 3 — input: forecast result          → output: call get_available_items
  Step 4 — input: full item catalog        → output: curated packing list based on weather
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
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    print(f"[step 1] get_weather({city!r})")
    return f"The current weather in {city} is 5°C with light rain."


@function_tool
def get_forecast(city: str, current_weather: str) -> str:
    """Get the 3-day forecast for a city given the current weather."""
    print(f"[step 2] get_forecast({city!r}, {current_weather!r})")
    return f"3-day forecast for {city}: rain today, clearing tomorrow, sunny by day 3."


@function_tool
def get_available_items() -> str:
    """Returns the full catalog of travel items available to pack."""
    print("[step 3] get_available_items()")
    return """Available items:
- Clothing: t-shirts, shorts, jeans, thermal underlayer, light fleece, heavy winter coat, waterproof jacket, formal shirt, swimwear
- Footwear: sandals, trainers, waterproof boots, formal shoes, flip flops
- Accessories: umbrella, sunglasses, sunhat, wool hat, scarf, gloves, belt, watch
- Toiletries: sunscreen SPF50, sunscreen SPF15, lip balm, moisturiser, insect repellent
- Electronics: phone charger, universal adapter, portable battery pack, laptop, e-reader
- Misc: travel pillow, reusable water bottle, snacks, first aid kit, travel insurance docs, guidebook"""


async def run_agent(agent, prompt):
    run_id = str(uuid.uuid4())
    client._custom_headers["X-MG-Run-ID"] = run_id
    print(f"\n[run] X-MG-Run-ID: {run_id}")
    return await Runner.run(agent, prompt, auto_previous_response_id=True)


async def main():
    agent = Agent(
        name="TravelAssistant",
        instructions=(
            "You are a travel assistant. When asked about travel to a city, you MUST call these "
            "tools in order: first get_weather, then get_forecast using the weather result, then "
            "get_available_items to retrieve the full item catalog. Once you have the catalog and "
            "the forecast, select only the items appropriate for the weather and trip and present "
            "a curated packing list with a brief reason for each item. "
            "Always give a complete, definitive answer based on the information available — "
            "never ask follow-up questions or request more details from the user."
        ),
        model=MGAI_MODEL,
        tools=[get_weather, get_forecast, get_available_items],
        model_settings=ModelSettings(parallel_tool_calls=False),
    )

    result = await run_agent(agent, "I'm travelling to London tomorrow, what should I pack?")
    print(f"\n[result] {result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
