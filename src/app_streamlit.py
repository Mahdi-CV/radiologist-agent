import asyncio
import streamlit as st
from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.messages import (
    PartDeltaEvent, TextPartDelta,
    FunctionToolCallEvent, FunctionToolResultEvent,
)
from config import orch_model
from tools_orchestrator import analyse_image_base64, show_reference_images_tool, draft_report_html
import re

# ── Constants ─────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

def is_html_content(text: str) -> bool:
    return bool(re.search(r"</?[a-z][\s\S]*>", text))

def show_html_report(html: str):
    report_container = st.container()
    with st.expander("📄 Radiology Report", expanded=True):
        st.components.v1.html(html, height=600)
    st.session_state.setdefault("reports", []).append(html)
    st.download_button(
    label="⬇️ Download Report as HTML",
    data=html,
    file_name="radiology_report.html",
    mime="text/html"
    )
    st.session_state.setdefault("reports", []).append(html)

SYSTEM_PROMPT = """
ROLE
You are a radiology assistant.

TOOLS
1. analyse_image_base64(path:str)            -> RadiologyReport
2. show_reference_images_tool(confirm:str)   -> dict
3. draft_report_html(diagnosis:str, recs:str,
                    critical:bool, img_path:str|None) -> {
                        subject, mimeType, body }

You have 3 main responsibilities:
1. Once you receive the result of "analyse_image_base64", you MUST output it using a markdown table with the following format:

| Field           | Value                          |
|----------------|----------------------------------|
| Diagnosis       | <diagnosis_description>         |
| Recommendations | <clinical_recommendations>      |
| Critical        | <critical>                      |


2. If the user asks for reference images or similar images: 
• IMMEDIATELY call:
    {"name":"show_reference_images_tool","arguments":{"confirm":"yes"}}

3. If the user asks for the report in a non-English language (e.g., Simplified Chinese, French, Spanish, etc.), you MUST:

a. Translate the following fields into the requested language:
   • diagnosis_description → diagnosis
   • clinical_recommendations → recs
   • critical → true/false (keep this boolean as-is, do not translate it)
   
b. Then call the tool:
   {
     "name": "draft_report_html",
     "arguments": {
       "diagnosis": "<translated diagnosis>",
       "recs": "<translated recommendations>",
       "critical": <critical>,
       "img_path": "<PATH>"
     }
   }


RULES
• Never skip a step.
• Never invent clinical data.
""".strip()

async def build_orchestrator():
    gmail_server = MCPServerStdio(
        command="npx",
        args=["-y", "@gongrzhe/server-gmail-autoauth-mcp"],
    )

    return Agent(
        model=orch_model,
        # mcp_servers=[gmail_server],
        tools=[analyse_image_base64, show_reference_images_tool, draft_report_html],
        system_prompt=SYSTEM_PROMPT,
        instrument=True,
    )


class StreamlitChatUI:
    def __init__(self, show_tool_calls: bool = False):
        self.orchestrator = None
        self.show_tool_calls = show_tool_calls
        st.session_state.setdefault("messages", [])
        st.session_state.setdefault("internal_history", [])

    async def initialize(self):
        if self.orchestrator is None:
            with st.spinner("⏳ Initializing assistant..."):
                self.orchestrator = await build_orchestrator()
        return self.orchestrator

    async def process_message(self, message: str):
        await self.initialize()

        st.session_state.messages.append({"role": "user", "content": message})
        assistant_placeholder = st.empty()
        full_response = ""

        async with self.orchestrator.run_mcp_servers():
            async with self.orchestrator.iter(
                user_prompt=message,
                message_history=st.session_state.internal_history,
            ) as run:
                async for node in run:
                    if Agent.is_call_tools_node(node):
                        async with node.stream(run.ctx) as s:
                            async for ev in s:
                                if isinstance(ev, FunctionToolCallEvent) and self.show_tool_calls:
                                    call = f"\n▶️ Tool call → {ev.part.tool_name}{ev.part.args}\n"
                                    full_response += call
                                    assistant_placeholder.markdown(full_response)
                                elif isinstance(ev, FunctionToolResultEvent):
                                    if self.show_tool_calls:
                                        result_str = f"✅ Tool Result:\n{ev.result}\n\n"
                                        full_response += result_str
                                        assistant_placeholder.markdown(full_response)

                                    if ev.result.tool_name == "show_reference_images_tool":
                                        self.show_reference_images()

                                    elif ev.result.tool_name == "draft_report_html":
                                        report = ev.result.content  
                                        report_html = report.get("body", "")
                                        show_html_report(report_html)

                                        return  # skip streaming full_response
                    elif Agent.is_model_request_node(node):
                        async with node.stream(run.ctx) as s:
                            async for ev in s:
                                if isinstance(ev, PartDeltaEvent) and isinstance(ev.delta, TextPartDelta):
                                    delta = ev.delta.content_delta
                                    if not is_html_content(delta):
                                        full_response += delta
                                        assistant_placeholder.markdown(full_response)

            st.session_state.internal_history.extend(run.result.all_messages())
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        assistant_placeholder.markdown(full_response)

    def show_reference_images(self):
        st.markdown("### 🩻 Reference Cases (Confirmed Diagnoses)")
        image_paths = [
            "data/cancer/sample_1.jpg",
            "data/cancer/sample_2.jpg",
            "data/cancer/sample_3.jpg",
            "data/cancer/sample_4.jpg",
        ]
        cols = st.columns(len(image_paths))
        for col, path in zip(cols, image_paths):
            with col:
                st.image(path, use_container_width=True, caption="Case: " + path.split("/")[-1].split(".")[0].replace("_", " ").title())

    async def process_upload_message(self, message: str, placeholder_container):
        await self.initialize()

        full_response = ""
        async with self.orchestrator.run_mcp_servers():
            async with self.orchestrator.iter(
                user_prompt=message,
                message_history=st.session_state.internal_history,
            ) as run:
                async for node in run:
                    if Agent.is_call_tools_node(node):
                        async with node.stream(run.ctx) as s:
                            async for ev in s:
                                if isinstance(ev, FunctionToolResultEvent):
                                    if ev.result.tool_name == "show_reference_images_tool":
                                        self.show_reference_images()
                                    if ev.result.tool_name == "draft_report_html":
                                        report = ev.result.content  
                                        report_html = report.get("body", "")
                                        show_html_report(report_html)

                    elif Agent.is_model_request_node(node):
                        async with node.stream(run.ctx) as s:
                            async for ev in s:
                                if isinstance(ev, PartDeltaEvent) and isinstance(ev.delta, TextPartDelta):
                                    full_response += ev.delta.content_delta

        # Save to state for chat history
        st.session_state.internal_history.extend(run.result.all_messages())
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # ✅ Show once, cleanly rendered
        # with placeholder_container:
        #     st.markdown("#### 📋 Radiology Report")
        #     st.markdown(full_response.strip())


def main():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("internal_history", [])
    st.session_state.setdefault("image_uploaded", False)

    st.set_page_config(page_title="Radiology Assistant", page_icon="🏥", layout="wide")

    with st.sidebar:
        st.header("🧾 Session Info")
        st.markdown("• Image uploaded: ✅" if st.session_state["image_uploaded"] else "• Awaiting image...")
        st.markdown(f"• Messages: {len(st.session_state['messages'])}")
        st.markdown("• Role: Assistant with tools")
        st.markdown("---")
        show_tool_calls = st.checkbox("Show Tool Call Logs", value=False)

    st.title("🏥 Radiology Assistant")
    st.markdown("Analyze radiology images and manage critical follow-up workflows using AI tools.")

    ui = StreamlitChatUI(show_tool_calls=show_tool_calls)

    with st.expander("📤 Upload Medical Image", expanded=True):
        uploaded_file = st.file_uploader("Choose a medical image (.jpg or .png)", type=["jpg", "png"])
        if uploaded_file and not st.session_state["image_uploaded"]:
            save_path = DATA_DIR / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            st.session_state["image_uploaded"] = True

            st.session_state["uploaded_image_path"] = str(save_path.resolve())
            response_container = st.container()
            with st.spinner("Analyzing uploaded image..."):
                asyncio.run(ui.process_upload_message(f"Analyze this image: {save_path}", response_container))


    st.divider()
    avatar_map = {"user": "🧑", "assistant": "🤖"}
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"], avatar=avatar_map.get(message["role"], "❓")):
            st.markdown(message["content"])

    if prompt := st.chat_input("💬 Ask about the diagnosis, treatment, or say 'show reference images'..."):
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("💭 Thinking..."):
                asyncio.run(ui.process_message(prompt))

if __name__ == "__main__":
    main()