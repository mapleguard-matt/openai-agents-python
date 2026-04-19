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

"""PE/IPO readiness analysis — 4-step sequential tool chain.

Simulates a private equity analyst evaluating a portfolio company for IPO.
Each tool depends on the previous result, forcing 4 sequential MGAI round trips
before the model synthesises a final recommendation.

Expected MGAI sequence per run (all sharing the same X-MG-Run-ID):
  Step 1 — input: company name         → output: call get_financial_performance
  Step 2 — input: financials            → output: call get_valuation_analysis
  Step 3 — input: valuation             → output: call get_ipo_market_conditions
  Step 4 — input: market conditions     → output: call assess_ipo_readiness_criteria
  Step 5 — input: readiness assessment  → output: final IPO recommendation
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
def get_financial_performance(company_name: str) -> str:
    """Retrieve historical revenue, growth, and profitability metrics for a company."""
    print(f"[step 1] get_financial_performance({company_name!r})")
    return f"""Financial performance for {company_name} (last 3 fiscal years):

Revenue:
  FY2022: $48.2M  |  FY2023: $71.6M  |  FY2024: $103.4M
  3-year CAGR: 46.5%  |  YoY growth FY2024: 44.4%

ARR (Annual Recurring Revenue): $118.2M (as of Q4 FY2024)
Net Revenue Retention: 124%
Gross Margin: 74.2%
EBITDA: -$4.1M (FY2024) — improving from -$18.3M (FY2022)
EBITDA Margin: -3.9% (FY2024), trending toward breakeven
Rule of 40 Score: 40.5 (growth rate + EBITDA margin)

Customer metrics:
  Total customers: 1,840
  Enterprise customers (>$100K ARR): 214
  Average contract value (enterprise): $312K
  Churn rate: 4.1% gross annual"""


@function_tool
def get_valuation_analysis(company_name: str, arr: str, gross_margin: str) -> str:
    """Run a valuation analysis using ARR and margin profile against comparable public companies."""
    print(f"[step 2] get_valuation_analysis({company_name!r}, arr={arr!r}, margin={gross_margin!r})")
    return f"""Valuation analysis for {company_name}:

Comparable public SaaS companies (enterprise workflow segment):
  Median EV/ARR multiple: 8.2x  |  Range: 5.4x – 14.1x
  High-growth cohort (>40% YoY) median: 11.4x
  Margin-adjusted (Rule of 40 > 40) premium: +1.8x

Implied valuation range for {company_name}:
  Base case  (8.2x ARR):   $969M   (~$1.0B)
  Bull case  (11.4x ARR):  $1.35B
  Bear case  (5.4x ARR):   $638M

Last private round (Series D, 18 months ago): $780M post-money
  → Bull case represents a 73% step-up from last round
  → Bear case represents an 18% step-down (down round risk)

Key valuation risks:
  - EBITDA still negative; public markets penalise unprofitable SaaS at current rates
  - Net new ARR adds decelerating slightly (Q3/Q4 FY2024 vs prior year)
  - Two large competitors raised at 6–7x ARR in recent quarters"""


@function_tool
def get_ipo_market_conditions(sector: str) -> str:
    """Retrieve current IPO market conditions, investor appetite, and recent comparable listings."""
    print(f"[step 3] get_ipo_market_conditions({sector!r})")
    return f"""IPO market conditions — {sector} (as of Q1 2026):

Market window: CAUTIOUSLY OPEN
  - US IPO volumes up 34% YoY in Q1 2026 vs depressed 2025 levels
  - Tech/SaaS IPO sentiment: neutral-to-positive; investors selective on profitability
  - S&P 500 within 3% of all-time highs; VIX at 17 (low volatility)

Recent comparable SaaS IPOs (last 12 months):
  DataStream Inc.   — Listed at 9.1x ARR, +22% on day 1, now +8% vs issue price
  CloudAxis Corp.   — Listed at 7.4x ARR, flat on day 1, now -11% vs issue price (EBITDA -ve)
  NexaFlow Ltd.     — Withdrew IPO filing citing market conditions (EBITDA -ve, slowing growth)

Investor appetite signals:
  - Funds actively looking for Rule of 40 stories with clear path to profitability
  - Negative EBITDA accepted only with >40% growth AND credible 18-month breakeven plan
  - Lock-up sensitivity high: several recent IPOs saw 15–20% drops at 180-day lock-up expiry

Recommended float size for this valuation range: 15–20% of fully diluted shares
Estimated IPO execution timeline: 6–9 months from go decision to listing"""


@function_tool
def assess_ipo_readiness_criteria(
    company_name: str,
    financial_summary: str,
    valuation_summary: str,
    market_summary: str,
) -> str:
    """Score the company against standard IPO readiness criteria."""
    print(f"[step 4] assess_ipo_readiness_criteria({company_name!r})")
    return f"""IPO readiness assessment for {company_name}:

Criteria scoring (1–5, where 5 = fully ready):

Financial readiness:
  Revenue scale & growth trajectory    5/5  — $103M ARR, 46% CAGR, strong NRR
  Profitability / path to breakeven    3/5  — EBITDA negative but improving; needs clear plan
  Financial reporting infrastructure   4/5  — Big 4 auditor in place; SOX prep 70% complete
  Predictability of revenue            4/5  — 89% recurring; some concentration in top 20 accounts

Business readiness:
  Management team depth                3/5  — CFO hired 8 months ago; IR function not yet built
  Corporate governance                 3/5  — Board refresh needed; two independent directors required
  Legal & compliance                   4/5  — Clean cap table; minor IP dispute (immaterial, counsel advises)
  ESG / sustainability reporting       2/5  — Basic disclosure only; institutional investors expect more

Market readiness:
  Brand & analyst awareness            3/5  — Known in sector; limited tier-1 analyst coverage
  Comparable IPO performance           3/5  — Mixed; profitability narrative critical
  Timing vs. market window             4/5  — Window open but selective; 6-month execution risk

Overall readiness score: 35/55 (64%) — CONDITIONALLY READY
Key blockers: CFO/IR team maturity, board composition, EBITDA breakeven roadmap, ESG baseline"""


async def run_agent(agent, prompt):
    run_id = str(uuid.uuid4())
    client._custom_headers["X-MG-Run-ID"] = run_id
    print(f"[run] X-MG-Run-ID: {run_id}\n")
    return await Runner.run(agent, prompt, auto_previous_response_id=True)


async def main():
    agent = Agent(
        name="IPOAdvisor",
        instructions=(
            "You are a senior investment banking analyst specialising in IPO advisory for "
            "private equity-backed companies. When asked to evaluate a company for IPO, you "
            "MUST call these tools in order:\n"
            "1. get_financial_performance — to retrieve revenue and profitability data\n"
            "2. get_valuation_analysis — using the ARR and gross margin from step 1\n"
            "3. get_ipo_market_conditions — using the company's sector\n"
            "4. assess_ipo_readiness_criteria — passing summaries from all three prior steps\n\n"
            "After all four tools have been called, deliver a structured recommendation covering: "
            "IPO readiness verdict, key strengths, critical risks, suggested timing, and any "
            "pre-IPO actions required. Be direct and definitive — do not ask follow-up questions."
        ),
        model=MGAI_MODEL,
        tools=[
            get_financial_performance,
            get_valuation_analysis,
            get_ipo_market_conditions,
            assess_ipo_readiness_criteria,
        ],
        model_settings=ModelSettings(parallel_tool_calls=False),
    )

    result = await run_agent(
        agent,
        "Evaluate NorthStar Analytics for IPO readiness. They are an enterprise SaaS company "
        "in the workflow automation sector.",
    )
    print(f"\n{'='*60}\nIPO ADVISORY RECOMMENDATION\n{'='*60}\n{result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
