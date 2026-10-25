# Kanuni experiments log

## E1. Information environment: news only vs news plus activists (Finance Bill 2024)

### Why
In the first clean OASIS run (`smoke_100_seed0_vm_bounded`, 74 heavy-online
agents, 17-20 Jun 2024) agents saw only three neutral news accounts. The
population did not escalate the way Kenya did:

| Day | Stance mean | Share opposed | Agents intending to protest |
|---|---|---|---|
| 17 Jun | -1.58 | 93% | 4 |
| 18 Jun (amendments) | -0.82 | 82% | 0 |
| 19 Jun | -0.50 | 70% | 0 |
| 20 Jun (second reading) | -0.47 | 63% | 0 |

Real world: opposition stayed high (Infotrak: about 80% opposed in late May)
and protests spread from Nairobi (18 Jun) to 19 counties (20 Jun) to about
35 counties (25 Jun). Validation similarity for this run: 47% (stance 0.87,
geography 0.50 and 0.50, escalation 0.0). Model `qwen3.5:27b` (knows the
outcome), so a smoke-test result, not a finding.

Working reading: a population that only hears official news reacts to the
concession (bread VAT dropped) and calms down. The 2024 mobilisation was
carried by activists and influencers online: explainers, MPs' phone numbers,
#RejectFinanceBill2024 and #OccupyParliament calls. Leaving them out may be
why the simulation under-reacts. This is worth documenting either way: it
tests whether mobilisation, not the policy alone, drives escalation.

### Design
Two information environments, everything else identical (personas, model,
seeds, platform settings, survey):

| Condition | OASIS | AgentSociety |
|---|---|---|
| `news_only` | `configs/smoke_100.json` | scenario `ke_fb24_backtest.json` |
| `news_plus_activists` | `configs/smoke_100_activists.json` | scenario `ke_fb24_backtest_activists.json` |

The activist layer (`shared/activist_posts.json`) adds 13 dated posts from
four fictional accounts: a youth activist collective, a Swahili/Sheng tax
explainer, a comedy/music creator, and a pro-government account. Posts
paraphrase what circulated publicly on each date; they never say which
counties protested, so geography stays a validation target. In OASIS
personas follow the activist and influencer accounts with probability 0.6
and the pro-government account with 0.3 (assumed). In AgentSociety the
posts reach people only through social media they use and WhatsApp
forwards, never radio or TV.

### What to compare
- Stance trajectory per day, overall and by connectivity, livelihood, age.
- Intent and action: protest, contact MP, posting.
- Exposure: share of agents who saw any activist post (`channel_exposures`
  sources `activist`, `influencer`, `pro_government`).
- Validation scorecard (`shared/validation/score.py`) for both conditions.
- At least 5 seeds per condition before drawing conclusions.

### Results
| Run | Condition | Similarity | Notes |
|---|---|---|---|
| `smoke_100_seed0_vm_bounded` (OASIS) | news_only | 47% | above |
| `smoke_100_activists_seed0` (OASIS) | news_plus_activists | 78% | stance 0.91, geography 0.60 / 0.63, escalation 1.0 |
| `as2_fb24_month_seed0` (AgentSociety) | news_only | pending | 1-month baseline |

Caveat for every result here: `qwen3.5:27b` was trained after June 2024 and
knows what happened. Reportable runs need a model with an earlier cutoff.

### First result (OASIS, seed 0, 14 Oct 2026)

| Day | news_only: mean / opposed / intend to protest / contact MP | news_plus_activists: mean / opposed / intend to protest / contact MP |
|---|---|---|
| 17 Jun | -1.58 / 93% / 4 / 9 | -1.53 / 89% / 16 / 30 |
| 18 Jun | -0.82 / 82% / 0 / 1 | -1.57 / 89% / 31 / 25 |
| 19 Jun | -0.50 / 70% / 0 / 1 | -1.65 / 91% / 34 / 21 |
| 20 Jun | -0.47 / 63% / 0 / 1 | -1.59 / 89% / 32 / 19 |

Failures: 3 of 296 agent-days in each run (model action rejected or one
unparsed answer); no network failures or context overflows.

Observations:
- With activists in the feed, opposition did not soften after the 18 Jun
  concession and protest intent rose instead of disappearing. This matches
  the real direction of events much better.
- Exposure was near-universal: every agent saw activist or influencer posts
  on 17 Jun (OASIS's random recommendations spread them beyond followers).
  So this run cannot separate exposed from unexposed agents; a lower follow
  rate or a recommender that favours followed accounts would be needed.
- The swing is now too uniform. Agents in 13 of 16 counties intended to
  protest on 18 Jun, when only Nairobi did (geography correct-rejection rate
  0.2). News-only under-reacts; news-plus-activists over-reacts on where.
- The escalation score (1.0) is generous: 13 to 14 counties counts as
  growth. Real growth was 1 to 19 to ~35. The score should weigh the size
  of growth, not only its direction.
- OASIS protest "geography" here is stated intent, not action.

Next: 5 seeds per condition; a middle condition with lower follow rates
(e.g. 0.2) to see whether the effect is dose-dependent; refine the
escalation metric; repeat in AgentSociety, where activist posts reach only
social-media and WhatsApp users.

### Series 2: vLLM backend (Qwen3.5-27B BF16), 15 Oct 2026

Same personas, events, settings and seed as series 1, but the model is
served by vLLM in full precision (BF16) instead of Ollama's 4-bit build
(Q4_K_M). Each 4-day, 74-agent run took about 12-14 minutes instead of
about 40-70.

| Day | news_only: mean / opposed / protest / contact MP | news_plus_activists: mean / opposed / protest / contact MP |
|---|---|---|
| 17 Jun | -1.68 / 91% / 19 / 8 | -1.53 / 86% / 21 / 38 |
| 18 Jun | -1.27 / 86% / 21 / 3 | -1.54 / 86% / 33 / 24 |
| 19 Jun | -1.34 / 85% / 25 / 1 | -1.58 / 89% / 34 / 26 |
| 20 Jun | -1.23 / 83% / 19 / 1 | -1.66 / 90% / 34 / 29 |
| Similarity | 52% | 52% |

Failures: news_only 7 of 296 agent-days (3 actions rejected, 4 unparsed
answers); news_plus_activists 10 (7 rejected, 2 unparsed, 1 network).

Observations:
- The model build matters as much as the information environment. With the
  4-bit Ollama build, news_only collapsed (-0.47, no protest intent by day
  2); with the full-precision build it stays opposed (-1.23) with steady
  protest intent. Results from different builds must not be pooled, and
  the build must be recorded with every run (configs now carry `backend`).
- Activists still shift behaviour in the same direction: stance holds or
  hardens after the concession instead of softening, protest intent is
  about 1.5x higher and intent to contact the MP is 10-30x higher.
- Both conditions now protest "everywhere from day one": 12-15 of 16
  counties already on 18 Jun, so escalation scores 0 and geography stays
  near chance. This, not overall stance, is the main gap to reality.

### Series 2, five seeds per condition (OASIS), 18-19 Oct 2026

Seeds 0-4, otherwise identical to the series 2 runs above (the seed sets
who follows whom and the order agents act in). Mean ± standard deviation
across the five runs; all 8 new runs exited cleanly.

| Day | news_only: mean / protest / contact MP | news_plus_activists: mean / protest / contact MP |
|---|---|---|
| 17 Jun | -1.61 ± 0.05 / 23 ± 5 / 10 ± 6 | -1.54 ± 0.04 / 23 ± 3 / 36 ± 4 |
| 18 Jun | -1.26 ± 0.07 / 23 ± 3 / 4 ± 2 | -1.56 ± 0.04 / 35 ± 7 / 26 ± 7 |
| 19 Jun | -1.29 ± 0.04 / 23 ± 2 / 2 ± 1 | -1.59 ± 0.03 / 36 ± 8 / 25 ± 6 |
| 20 Jun | -1.21 ± 0.08 / 17 ± 2 / 3 ± 3 | -1.62 ± 0.04 / 37 ± 5 / 26 ± 4 |
| Similarity | 59% ± 11 (52, 78, 54, 52, 57) | 61% ± 12 (52, 52, 74, 54, 74) |

- The activist effect is robust: on every day from 18 Jun the gap between
  conditions is several times the seed-to-seed spread.
- The similarity score is not: single runs range from 52% to 78% and the
  two conditions overlap. Report it only as a mean over seeds.

## E2. Counterfactuals: what if the bill had passed? (AgentSociety, 100 agents)

### Design

Same personas, channels and household model as the 1-month baseline, one
seed each, run to 28 Jul (daily to 1 Jul, then weekly; 20 steps).

- `ke_fb24_passed` ("passed as amended"): the real timeline to 25 Jun, but
  the 26 Jun withdrawal is replaced by invented news that the President
  signed the amended bill (26 Jun) and it took effect (1 Jul)
  (`counterfactual_news_fb24_amended.json`). Levers from 1 Jul: excise on
  mobile-money fees and the narrowed eco levy.
- `ke_fb24_as_published`: the 18 Jun amendments never happen, the bill is
  signed as first published. Levers add 16% VAT on bread, a wider eco
  levy, motor-vehicle tax and VAT on financial services.

The two counterfactual news files are invented and flagged as such in
their `_note`. Levers marked `assumed` are placeholders until KNBS data
replaces them.

### Results

| | Withdrawn (news) | Withdrawn (+activists) | Passed as amended | Passed as published |
|---|---|---|---|---|
| Stance 18 Jun / 1 Jul / 22 Jul | -1.69 / -1.84 / - | -1.76 / -1.78 / - | -1.55 / -1.70 / -1.75 | -1.87 / -1.85 / -1.85 |
| Protest intent 20 Jun / 22 Jul | 20 / - | 18 / - | 25 / 5 | 34 / 18 |
| Protest joins, total | 103 | 195 | 131 | 160 |
| Contact MP actions, total | 71 | 113 | 50 | 118 |
| In debt 1 Jul / 22 Jul | 47 / - | 49 / - | 43 / 49 | 44 / 55 |
| Behind on a bill 1 Jul / 22 Jul | 16 / - | 18 / - | 18 / 42 | 16 / 48 |
| Cutting back 1 Jul / 28 Jul | 50 / - | 49 / - | 51 / 57 | 54 / 58 |
| Model failures (agent-steps) | 10 / 1700 | 6 / 1700 | 8 / 2000 | 18 / 2000 |
| Similarity (to 25 Jun) | 75% | 70% | 84% | n/a (timeline differs from 18 Jun) |

July spending per agent, published vs amended (medians): +2-3% for boda
riders, formal employees, matatu crews and farmers; +8% for high-income
professionals; but -16% for casual labourers and -7% for students and
job seekers, who have no cash buffer: they buy less rather than pay more.

### Reading it

- **Noise floor.** "Passed as amended" sees exactly the same events as the
  withdrawn baseline up to 25 Jun, so the gap between them before 26 Jun is
  pure run-to-run noise: up to 0.15 in mean stance, about 5 households on
  each money measure, and 75% vs 84% similarity. Any single-run difference
  smaller than that means nothing.
- **Published vs amended** is the clearest effect: without the 18 Jun
  concession, stance never softens (-1.87 vs -1.55) and protest intent
  stays high into late July (18 vs 5). Contact-MP actions more than double.
- **Household strain** in July rises in both counterfactuals, but most of
  the rise (behind on a bill 18 to 42-48) comes from monthly rent and fees
  falling due, not from the bill. There is no withdrawn run through July to
  separate the two. The published-vs-amended gap (+6 in debt, +6 behind)
  is near the noise floor.
- **Withdrawn + activists** in AgentSociety: protest joins nearly double
  (195 vs 103), as in OASIS, but similarity falls (70% vs 75%) because
  protests spread to more counties too early.

Next: a withdrawn control run through 28 Jul; 3+ seeds per counterfactual;
replace the assumed levers with KNBS figures; a pre-June-2024 model.

## E3. Policy simulations from uploaded drafts (six seeds, OASIS and AgentSociety)

### Design

Two drafts were run through the backend's policy-simulation pipeline
(stakeholder framework v1.1): extract measures, map stakeholder groups,
generate personas, compile inputs. No persona was edited by hand.

- **Transport:** e-mobility in the Nairobi CBD (county policy). 59 personas;
  55 are online, so OASIS gets 55.
- **Education:** AI in Education policy framework (national). 52 personas,
  48 online.

Each run covers 7 days (21-27 Oct 2026, daily), with 11 events: the
publication plus 10 news items, each on one of the draft's measures. There
are no price levers and no scripted protest calls. Seeds 0-5 per draft per
simulator, 24 runs, all exited cleanly. Seed 0 ran on vLLM with 16
concurrent sequences, seeds 1-5 with 128. The model and inputs are
identical.

### Results (mean over six seeds; spread in brackets)

| | Opposition day 1 → day 7 | Mean stance day 1 → day 7 |
|---|---|---|
| Education, OASIS | 23% → 40% (sd 14) | +0.54 → +0.17 |
| Education, AgentSociety | 22% → 9% (sd 3) | +0.51 → +1.01 |
| Transport, OASIS | 12% → 9% (sd 11) | +0.78 → +0.79 |
| Transport, AgentSociety | 26% → 25% (sd 5) | +0.48 → +0.60 |

- **Education:** in OASIS, opposition rises in 5 of 6 seeds. School
  administrators (average -0.54) and teachers (-0.38) end negative in 5 of 6
  seeds. The argument that spreads is "basics first": fix water,
  classrooms, textbooks and teacher pay before AI, and remember that "free
  education" ended up costing parents money. Such posts are 33-58% of posts
  on days 4-7 in OASIS, against about 10% in AgentSociety. AgentSociety
  spreads information mostly through WhatsApp, has no shared public feed
  and shows no backlash: every group ends positive.
- **Transport:** support is stable in both simulators. The bottom three
  groups are the same in both: small matatu owners (negative in 5 of 6
  AgentSociety seeds), conductors and drivers. Charging developers,
  financial institutions and battery importers are the most supportive.
  Protest intent appears only in AgentSociety (5-26 agent-days per run).
  There is far more MP contact than for education (11-36 per OASIS run,
  against 1-4).

### Failures

AgentSociety: no interview failures; at most 1 action failure per run.
OASIS: 7-17% of interviews are counted as parse failures, and almost all
of them (650 of 651 across E3 education and E4) are the same fault. The
model writes valid JSON with a stance but leaves off the closing `}`. The
raw text is kept, so the analysis below recovers these answers. Failures
lean slightly towards agents whose previous answer was negative, so the
raw summaries understate opposition a little.

## E4. Counterfactual: the education draft with funded basics (OASIS and AgentSociety)

### Design

Personas, seeds and the 11 E3 events are unchanged. Two things are added
(`events_education_amended.json`, marked invented in its `_note`):

1. One sentence in the publication summary: ring-fenced annual funding for
   school electricity, connectivity, classrooms and teacher training, and
   no AI-related fees or levies on learners or parents.
2. A news item on 22 Oct, "section 18, amended". It replaces "subject to
   budgetary approval" with a ring-fenced budget line. Underserved schools
   get infrastructure before any AI tool. Teacher AI training is free, held
   in working hours and counts towards promotion. There are no AI fees for
   parents. The costed plan is tabled in Parliament before Phase II.

The amendment gives no amounts. Persona exposure text still describes the
original draft.

### Results (OASIS, six seeds, truncated interviews recovered)

| | Opposition day 1 | Opposition day 7 | Mean stance day 7 |
|---|---|---|---|
| Draft as written | 23% | 38% (21-54% by seed) | +0.18 |
| Amended | 11% | 10% (0-29% by seed) | +0.77 |

- Opposition no longer rises over the week. One seed (seed 4) still reaches 29%; every
  baseline seed but one ends above that.
- Teachers move from -0.38 to +0.61 on day 7 and are negative in 0 of 6
  seeds, against 5 of 6. School administrators move from -0.54 to +0.08.
  These group averages use only the answers that parsed.
- The "basics first" argument does not go away (15-37 posts per run on days
  4-7), but its authors have changed. Where the author's stance is known,
  68% of these posts come from supporters (85 of 125), against 27% at
  baseline (59 of 218), and 14% from opponents, against 44%.
- Without recovery, interview failures double in the amended runs (46-94
  per run, against 25-48). The cause is the same missing `}`. With the
  answers recovered, all 48 agents are measured in both conditions and the
  effect is unchanged (carry-forward imputation gives 38% → 9%).

### Results (AgentSociety, six seeds)

| | Opposition day 1 → day 7 | Mean stance day 1 → day 7 |
|---|---|---|
| Draft as written | 22% → 9% (sd 3) | +0.51 → +1.01 |
| Amended | 5% → 3% (sd 3) | +1.24 → +1.42 |

There was no backlash here to remove. Instead, the amendment raises
support from the first day, and every group ends positive in every seed.
Teachers end at +1.42 (against +0.92), school administrators at +1.21
(against +0.79). Interview failures stay at 0.

### Reading it

The two simulators agree that the amendment helps. They differ on what
it does. In OASIS it stops a public-debate backlash led by teachers and
school heads. In AgentSociety, where no backlash forms, it shifts support
upwards by about 0.4 throughout. The finding for the draft's author: the
risk is teachers' doubt that AI comes with real money for basics, and
writing that money into the policy answers it in both simulators.

### Parser fix (23 Oct 2026)

From 23 Oct, `stance.py` in both simulators (still identical) repairs this
one fault. When the text after the last `{` has no `}`, it adds one. If
that then parses, the answer counts as ok, is marked
`interview_repaired` and is counted in `totals.interviews_repaired`.
Answers cut off any other way still fail. A replay over all 6,342 saved
OASIS policy-run interviews gives:

- all 5,346 that parsed before parse the same way now;
- 993 of the 996 failures are repaired;
- 3 remain failures (genuinely malformed).

Runs before 23 Oct have the raw text saved and can be re-scored. The E4
OASIS table already uses recovered answers. The E3 tables do not: the E3
OASIS education figure for day 7 (40%) comes out at 38% when recovered.

Next: test single parts of the amendment (fees only, infrastructure only)
to see which one carries the effect; give the amendment real amounts.
