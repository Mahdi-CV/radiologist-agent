import asyncio
import json
import textwrap

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.messages import (
    PartDeltaEvent, TextPartDelta,          # model‑token events
    FunctionToolCallEvent, FunctionToolResultEvent,  # tool events
)


# your Qwen-3 planning model
from config import orch_model

# any other tools you need
from tools_orchestrator import analyse_image_base64 #, to_base64, get_test_image_b64

async def build_orchestrator():

    fs_server = MCPServerStdio(
        # uses your existing mcp_config.json entry:
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "/home/mahdi/git/pydantic-ai-agents-tutorial/radiology_async/data",
        ],
    )

    gmail_server = MCPServerStdio(
        command="npx",
        args=["-y", "@gongrzhe/server-gmail-autoauth-mcp"],
    )

    # 3) clear, deterministic instructions
    system_prompt = """
    You have two tools:

    • analyse_image_base64(path: str = "data/image.jpg") -> RadiologyReport
    • send_email_gmail(
        to: List[str],
        subject: str,
        body: str,
        cc: List[str] | None = [],
        bcc: List[str] | None = []
    ) -> "Gmail message ID"

    Rules
    1. If the user has **not** provided an image path, ask:
        “Which image file should I analyse? (default = data/image.jpg)”
    2. When you have a path, call the tool. Your reply must be ONLY the JSON:
        {"name": "analyse_image_base64",
        "arguments": {"path": "<PATH_HERE>"}}

    Example
    User: default
    Assistant:
    {"name": "analyse_image_base64",
    "arguments": {"path": "data/image.jpg"}}

    3. Summarise the findings for the user.
    4. If the user asks to email / share / peer‑review, draft an email
    *in your own response*, show it **and ask for confirmation**:
        “Would you like me to send this e‑mail?”
    5. If the user confirms, call send_email_gmail with the draft.
    Do **not** send without explicit confirmation.
    """.strip()



    # 4) build your Agent
    orchestrator = Agent(
        model=orch_model,
        mcp_servers=[fs_server,  gmail_server],
        tools=[analyse_image_base64],  
        system_prompt=system_prompt,
        instrument=True,
    )

    return orchestrator

async def chat_loop() -> None:
    orchestrator = await build_orchestrator()          # await the agent

    async with orchestrator.run_mcp_servers():         # keep MCP running
        message_history: list[  # previous turns, starts empty
            pydantic_ai.messages.ModelMessage
        ] = []

        while True:
            user = input("\n🧑  You: ").strip()
            if user.lower() in {"quit", "exit"}:
                break

            # ─── stream one turn ──────────────────────────────────────────
            async with orchestrator.iter(
                user_prompt=user,               # ① this turn’s prompt
                message_history=message_history # ② prior context
            ) as run:

                async for node in run:
                    if Agent.is_call_tools_node(node):
                        async with node.stream(run.ctx) as s:
                            async for ev in s:
                                if isinstance(ev, FunctionToolCallEvent):
                                    print(
                                        f"\n▶️  Tool call → {ev.part.tool_name}{ev.part.args}"
                                    )
                                elif isinstance(ev, FunctionToolResultEvent):
                                    print("✅ Tool result received\n")

                    elif Agent.is_model_request_node(node):
                        async with node.stream(run.ctx) as s:
                            async for ev in s:
                                if (
                                    isinstance(ev, PartDeltaEvent)
                                    and isinstance(ev.delta, TextPartDelta)
                                ):
                                    print(ev.delta.content_delta, end="", flush=True)

            # keep full, typed history for the next turn
            message_history = run.result.all_messages()

if __name__ == "__main__":
    asyncio.run(chat_loop())
