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
from tools_orchestrator import analyse_image_base64, show_reference_images_tool

# ── Constants ─────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """
ROLE
You are a radiology assistant.

TOOLS
1. analyse_image_base64(path:str)            -> RadiologyReport
2. show_reference_images_tool(path:str)      -> dict
3. send_email(to:list[str], subject:str,
              body:str, cc:list[str]|None=[],
              bcc:list[str]|None=[])         -> Gmail message ID

FLOW
A) Get image
• If no image path: ask “Which image file should I analyse? (default = data/image.jpg)”

B) Analyse image
• Once you have <PATH> call: {"name":"analyse_image_base64","arguments":{"path":"<PATH>"}}

C) Present result
• Tool returns {critical, diagnosis_description, clinical_recommendations}
• Show this summary:
    Diagnosis: <diagnosis_description>
    Recommendations: <clinical_recommendations>
    Critical: Yes/No

D) Offer reference images ── ALWAYS happens first
• Ask: “Would you like to view reference images from similar confirmed cases?”
• If user replies yes/yep/“show them”… IMMEDIATELY call:
    {"name":"show_reference_images_tool","arguments":{"confirm":"yes"}}
• After the tool finishes —or if the user said “no” —continue to step E.

E) Handle critical cases ── ONLY if critical == true
• Ask: “This case is marked as critical. Would you like to send it for peer review?”
• If user agrees:
    1. Draft a plain-text email (from Dr. Mahdi Ghodsi) with the summary.
    2. Show the draft and ask: “Would you like me to send this email?”
    3. If user confirms sending:
        Ask for recipient address.
        Call IMMEDIATELY:
        {"name":"send_email","arguments":{"to":["<EMAIL>"],"subject":"<SUBJECT>","body":"<BODY>"}}

RULES
• Never skip a step.
• D must finish (or be declined) before E begins.
• Never send an email without explicit confirmation.
• Never invent clinical data.
""".strip()


async def build_orchestrator():
    gmail_server = MCPServerStdio(
        command="npx",
        args=["-y", "@gongrzhe/server-gmail-autoauth-mcp"],
    )

    return Agent(
        model=orch_model,
        mcp_servers=[gmail_server],
        tools=[analyse_image_base64, show_reference_images_tool],
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
                    elif Agent.is_model_request_node(node):
                        async with node.stream(run.ctx) as s:
                            async for ev in s:
                                if isinstance(ev, PartDeltaEvent) and isinstance(ev.delta, TextPartDelta):
                                    full_response += ev.delta.content_delta
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
                    elif Agent.is_model_request_node(node):
                        # 👇 DO NOT stream updates line by line
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