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
