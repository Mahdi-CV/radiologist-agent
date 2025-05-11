import asyncio
import json
import textwrap

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

# your Qwen-3 planning model
from config import orch_model

# any other tools you need
from tools_orchestrator import analyse_image_base64, to_base64

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

    # 3) clear, deterministic instructions
    system_prompt = textwrap.dedent("""\
    You are a deterministic orchestrator. You have exactly two tools:
    • list_directory(path: str) -> List[str]
    • read_file(path: str) -> bytes

    When the user says “start fs test”, you MUST do exactly **one** call, and **only** emit this `<tool_call>` block—no extra text:

    <tool_call>
    {"name":"list_directory","arguments":{"path":"data"}}
    </tool_call>

    End immediately after that `</tool_call>`. Do not emit anything else.
    """)

    # 4) build your Agent
    orchestrator = Agent(
        model=orch_model,
        mcp_servers=[fs_server],
        tools=[to_base64, analyse_image_base64],  
        system_prompt=system_prompt,
        instrument=True,
    )

    return orchestrator

async def main():
    orchestrator = await build_orchestrator()

    # this context manager will launch your FS server and connect to it
    async with orchestrator.run_mcp_servers():
        run = await orchestrator.run(
            messages=[{"role": "user", "content": "start fs test"}]
        )
        print("FINAL OUTPUT:", json.dumps(run.output, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
