# Kanuni on AgentSociety 2: design (draft)

OASIS models the **online** reaction to a policy: what people see on X/Twitter
and how they respond. AgentSociety 2 (AS2) models the **offline** life of the
general public: how people hear about a policy through radio, church or the
market, how it changes their spending, work and debt, and whether they
protest. Most of a policy's economic, social and cultural effects happen
offline, so this is where AS2 adds value. It is not built to copy OASIS.

## Policies

| Bill | Kind of run | Status | Notes |
|---|---|---|---|
| Finance Bill 2024 | Backtest, plus "what if it had passed" scenarios | Withdrawn 26 Jun 2024 | Protest dates and counties can be checked against reality. Spending effects never happened, so those are counterfactual and must be labelled that way. |
| Artificial Intelligence Bill 2026 (Senate Bill No. 4 of 2026, Sen. Karen Nyamu) | Forward-looking scenarios | First reading 2 Apr 2026, with the Senate ICT committee | No outcome to backtest. Our 30 hand-written personas were built for this bill (section references in `personas_v1.csv`). |

Each bill is a **scenario pack**: a dated event timeline, plus the economic
levers each clause would pull if enacted (for example, `vat_bread`: bread
+16%). A new scenario is a new data file, with no code changes.

## What each agent has

1. **Persona**: from `personas.json`, the same file OASIS uses.
2. **Household account**, kept by code and not by the model. Its contents come from `countries/ke/livelihood_profiles.json`:
   - income on that livelihood's rhythm (daily, monthly, seasonal or irregular)
   - obligations with due dates (rent on the 5th, fees at term start, daily bike payment)
   - cash, savings and debt
   - consequences when a payment is missed
3. **Information diet**: the channels the persona actually uses (radio, TV, WhatsApp, church/mosque, matatu/market talk, social media), plus word of mouth from family, neighbours, chama and workmates.
4. **Memory**: AS2's own memory notes, so consequences carry forward.

## Environment modules (ours, no AS2 core edits)

- **Information**: delivers each day's events through each persona's channels and spreads news by word of mouth. It records every exposure as a `channel_exposures` item.
- **Household economy**: applies income, prices and obligations, and enforces budgets (an agent cannot spend money it does not have). It applies the policy levers when a clause takes effect. Agent choices: cut back, switch to cheaper goods, work more, start a side hustle, borrow, or skip a payment.
- **Civic**: protest (where and when; costs the day's earnings), contacting the MP, and discussing the bill at church, chama or work. Turnout is added up by county and day.

## Time

The step length can change during a run: daily around key dates, weekly or
monthly in between. Bookkeeping costs no model calls; only decisions do.

| Run | Horizon | Agents | Step | Purpose |
|---|---|---|---|---|
| Smoke test | 2-4 days | 5-10 | daily | Check the pipeline |
| **Baseline (report)** | **1 month** | ~100 | daily around key dates, otherwise weekly | First reportable baseline per bill |
| Paper | 2 and 5 years | ~20-30 | monthly, daily around shocks | Long-run economic and social effects |

Long runs need assumptions about the wider economy over the period (inflation,
fuel prices). These come from the scenario file, not from the model, and are
reported as assumptions.

## Outputs

- `output.jsonl`: the shared format, unchanged, with `simulator: "agentsociety"`. `post_text` is usually null.
- `household_days.jsonl`: AS2 only. One row per agent per step with income, spending by category, obligations paid or missed, debt, work days and protest attendance.
- `summary.json`: stance by step, failure counts (model vs network), and protest turnout by county.

## Run rules (same as OASIS)

- **Our own day loop.** One agent's failure is counted and does not stop the step. AS2's built-in step aborts on any failure.
- **Our strict stance survey.** It is not saved to the agent's memory. AS2's built-in questionnaire repairs and retries answers, so we don't use it.
- **AS2 corrects malformed model output within a step.** This cannot be turned off without editing AS2's core, so it is counted from the traces and reported.
- **Leakage.**
  - The Finance Bill backtest needs a model with a training cutoff before June 2024.
  - For the AI Bill, a model trained after April 2026 may already know the public debate. Record the cutoff with every run.

## Other countries later

Code is country-neutral. Everything country-specific lives in `countries/<iso>/`:

```
countries/ke/
  livelihood_profiles.json   # incomes, obligations, prices, calendar, levies
  channels.json              # media and offline channels (to do)
  personas.json              # or a generator plus census weights (to do)
countries/tz/ ...            # e.g. Tanzania AI bill, later
scenarios/ke_fb24.json       # bill timeline plus economic levers (to do)
scenarios/ke_ai_bill_2026.json
```

A new country needs a country pack and a scenario. This is deferred until the
Kenya baseline is solid.

## Open items

- Replace "assumed" figures in `livelihood_profiles.json` with KNBS or Qhala data.
- Get the AI Bill 2026 text to map its clauses to levers (compliance costs, fines, sector effects).
- Measure AS2's model calls per agent-step on the VM (5 agents, 2 days) to size the 1-month and multi-year runs.
