"""Check that a CAMEL agent can make a tool call through the configured model.

Usage: python -m kanuni.check_model [--url http://127.0.0.1:8765/v1]
"""
import argparse
import asyncio

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import FunctionTool
from camel.types import ModelPlatformType

calls = []


def like_post(post_id: int) -> dict:
    r"""Like a post.

    Args:
        post_id (int): The ID of the post to like.

    Returns:
        dict: Whether the like succeeded.
    """
    calls.append(("like_post", post_id))
    return {"success": True}


def create_post(content: str) -> dict:
    r"""Create a new post.

    Args:
        content (str): The text of the post.

    Returns:
        dict: Whether the post was created.
    """
    calls.append(("create_post", content))
    return {"success": True}


async def main(url: str, model: str) -> None:
    backend = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type=model, url=url, api_key="unused",
        model_config_dict={"temperature": 0.7}, timeout=600, max_retries=0)
    agent = ChatAgent("You are a Kenyan social media user.", model=backend,
                      tools=[FunctionTool(like_post),
                             FunctionTool(create_post)])
    resp = await agent.astep(
        'Post 1: "Finance Bill proposes 16% VAT on bread." '
        "Perform one action by tool calling.")
    print("tool calls:", [(t.tool_name, t.args) for t in
                          resp.info["tool_calls"]])
    print("executed:", calls)
    print("final text:", resp.msgs[0].content if resp.msgs else None)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8765/v1")
    p.add_argument("--model", default="qwen3.5:27b")
    a = p.parse_args()
    asyncio.run(main(a.url, a.model))
