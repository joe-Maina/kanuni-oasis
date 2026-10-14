"""Step 1 of the smoke test: OASIS's own example, unchanged except for the
model backend and the number of agents.

Mirrors examples/twitter_simulation_vllm.py (default Twitter platform with the
TwHIN-BERT recommender and the example's own user CSV), but uses the first 10
users and our Ollama model through kanuni/ollama_shim.py. Afterwards it counts
how many LLM activations produced a valid action in the trace table.

Usage (from the repo root, with the shim running):
    python -m kanuni.upstream_example_10
"""
import asyncio
import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
from camel.models import ModelFactory
from camel.types import ModelPlatformType

REPO = Path(__file__).resolve().parent.parent
RUN_DIR = REPO / "kanuni" / "runs" / "upstream_example_10"
SRC_CSV = (REPO / "data" / "twitter_dataset" / "anonymous_topic_200_1h" /
           "False_Business_0.csv")
N_AGENTS = 10
MODEL = os.environ.get("KANUNI_MODEL", "qwen3.5:27b")
URL = os.environ.get("KANUNI_MODEL_URL", "http://127.0.0.1:8765/v1")


def write_profile_slice() -> Path:
    df = pd.read_csv(SRC_CSV).head(N_AGENTS)
    # The example's follow lists point at agents outside the slice; drop them.
    df["following_agentid_list"] = "[]"
    out = RUN_DIR / "profiles_10.csv"
    df.to_csv(out, index=False)
    return out


async def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = write_profile_slice()
    db_path = RUN_DIR / "twitter_simulation.db"
    if db_path.exists():
        db_path.unlink()
    os.environ["OASIS_DB_PATH"] = str(db_path)
    os.chdir(RUN_DIR)  # OASIS writes ./log relative to the cwd

    import oasis
    from oasis import (ActionType, LLMAction, ManualAction,
                       generate_twitter_agent_graph)

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type=MODEL, url=URL, api_key="unused",
        model_config_dict={"temperature": 0.7}, timeout=600, max_retries=0)

    agent_graph = await generate_twitter_agent_graph(
        profile_path=str(profile_path),
        model=model,
        available_actions=ActionType.get_default_twitter_actions(),
    )
    env = oasis.make(agent_graph=agent_graph,
                     platform=oasis.DefaultPlatformType.TWITTER,
                     database_path=str(db_path))
    await env.reset()

    await env.step({env.agent_graph.get_agent(0): ManualAction(
        action_type=ActionType.CREATE_POST,
        action_args={"content": "Earth is flat."})})
    await env.step({agent: LLMAction() for _, agent in
                    env.agent_graph.get_agents([1, 3, 5, 7, 9])})
    await env.step({env.agent_graph.get_agent(1): ManualAction(
        action_type=ActionType.CREATE_POST,
        action_args={"content": "Earth is not flat."})})
    await env.step({agent: LLMAction()
                    for _, agent in env.agent_graph.get_agents()})
    await env.close()

    # Each LLM activation calls refresh first, so count refreshes as
    # activations and non-bookkeeping trace rows at the same step as actions.
    con = sqlite3.connect(db_path)
    rows = con.execute("SELECT user_id, created_at, action FROM trace "
                       "ORDER BY created_at").fetchall()
    con.close()
    activations = {(u, t) for u, t, a in rows if a == "refresh"}
    acted = {(u, t) for u, t, a in rows
             if a not in ("refresh", "sign_up") and (u, t) in activations}
    actions = [a for u, t, a in rows if (u, t) in acted
               and a not in ("refresh", "sign_up")]
    summary = {
        "llm_activations": len(activations),
        "activations_with_valid_action": len(acted),
        "activations_without_action": len(activations - acted),
        "action_counts": pd.Series(actions).value_counts().to_dict(),
        "db": str(db_path),
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
