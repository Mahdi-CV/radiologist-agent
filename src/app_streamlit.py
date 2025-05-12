import asyncio
import streamlit as st
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.messages import (
    PartDeltaEvent, TextPartDelta,
    FunctionToolCallEvent, FunctionToolResultEvent,
)
from config import orch_model
from tools_orchestrator import analyse_image_base64, show_reference_images_tool
from pathlib import Path

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
   • If no image path: ask
     “Which image file should I analyse? (default = data/image.jpg)”

B) Analyse image
   • Once you have <PATH> call:
     {"name":"analyse_image_base64","arguments":{"path":"<PATH>"}}

C) Present result
   • Tool returns {critical, diagnosis_description, clinical_recommendations}
   • Show this summary:
     Diagnosis: <diagnosis_description>
     Recommendations: <clinical_recommendations>
     Critical: Yes/No

D) Offer reference images  ── ALWAYS happens first
   • Ask exactly:
     “Would you like to view reference images from similar confirmed cases?”
   • If user replies yes/yep/“show them”… IMMEDIATELY call:
     {"name":"show_reference_images_tool","arguments":{"confirm":"yes"}}
     Say nothing else.
   • After the tool finishes —or if the user said “no” —continue to step E.

E) Handle critical cases  ── ONLY if critical == true
   • Ask:
     “This case is marked as critical. Would you like to send it for peer review?”
   • If user agrees:
       1. Draft a plain-text email (from Dr. Mahdi Ghodsi) with the summary.
       2. Show the draft and ask:
          “Would you like me to send this email?”
   • If user confirms sending:
       1. Ask for recipient address.
       2. Call IMMEDIATELY:
          {"name":"send_email",
           "arguments":{"to":["<EMAIL>"],"subject":"<SUBJECT>","body":"<BODY>"}}

RULES
• Never skip a step.
• D must finish (or be declined) before E begins.
• Never send an email without explicit confirmation.
• Never invent clinical data.
"""

async def build_orchestrator():
    # fs_server = MCPServerStdio(
    #     command="npx",
    #     args=[
    #         "-y",
    #         "@modelcontextprotocol/server-filesystem",
    #         "/home/mahdi/git/pydantic-ai-agents-tutorial/radiology_async/data",
    #     ],
    # )

    gmail_server = MCPServerStdio(
        command="npx",
        args=["-y", "@gongrzhe/server-gmail-autoauth-mcp"],
    )

    system_prompt = SYSTEM_PROMPT

    return Agent(
        model=orch_model,
        mcp_servers=[gmail_server],
        tools=[analyse_image_base64, show_reference_images_tool],
        system_prompt=system_prompt,
        instrument=True,
    )
        
class StreamlitChatUI:
    def __init__(self):
        self.orchestrator = None
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "internal_history" not in st.session_state:
            st.session_state.internal_history = []
        if "uploaded_image" not in st.session_state:
            st.session_state.uploaded_image = None
        if "analysis_triggered" not in st.session_state:
            st.session_state.analysis_triggered = False
        if "show_references" not in st.session_state:
            st.session_state.show_references = False
        if "processing" not in st.session_state:
            st.session_state.processing = False
        if "last_message" not in st.session_state:
            st.session_state.last_message = None

    async def initialize(self):
        if self.orchestrator is None:
            print("⏳ Bootstrapping orchestrator...")
            with st.spinner("Initializing orchestrator..."):
                self.orchestrator = await build_orchestrator()
            print("✅ Orchestrator ready!")
        return self.orchestrator
    
    async def process_message(self, message: str):
        if st.session_state.processing:
            return
            
        st.session_state.processing = True
        try:
            await self.initialize()

            # Record the user's turn in the UI
            st.session_state.messages.append({"role": "user", "content": message})
            st.session_state.last_message = message

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
                                    if isinstance(ev, FunctionToolCallEvent):
                                        call = f"\n▶️ Tool call → {ev.part.tool_name}{ev.part.args}\n"
                                        full_response += call
                                        assistant_placeholder.markdown(full_response)
                                        print(call.strip())  
                                    elif isinstance(ev, FunctionToolResultEvent):
                                        result_str = f"✅ Tool Result:\n{ev.result}\n\n"
                                        full_response += result_str
                                        assistant_placeholder.markdown(full_response)
                                        print(result_str.strip())
                                        if ev.result.tool_name == "show_reference_images_tool":
                                            print("✅ show_reference_images_tool was triggered!")
                                            st.session_state.show_references = True
                                            # Don't rerun here - we'll handle it in the main loop

                        elif Agent.is_model_request_node(node):
                            async with node.stream(run.ctx) as s:
                                async for ev in s:
                                    if isinstance(ev, PartDeltaEvent) and isinstance(ev.delta, TextPartDelta):
                                        full_response += ev.delta.content_delta
                                        assistant_placeholder.markdown(full_response)
                
                new_messages = run.result.all_messages()
                st.session_state.internal_history.extend(new_messages) 

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            assistant_placeholder.markdown(full_response)
        finally:
            st.session_state.processing = False

def main():
    st.set_page_config(layout="wide")
    st.title("🏥 Radiology Assistant")
    
    # Initialize UI if not already done
    if "ui" not in st.session_state:
        st.session_state.ui = StreamlitChatUI()
    
    # Create two columns
    col1, col2 = st.columns([1, 2], gap="medium")
    
    # Left column - Image upload and preview only
    with col1:
        st.header("Image Upload")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload medical image", type=["jpg", "png"], 
                                       key="file_uploader")
        
        if uploaded_file:
            # Save the image
            save_path = DATA_DIR / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Store and display the image
            st.session_state.uploaded_image = uploaded_file
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            
            # Set flag to trigger analysis but don't process here
            if not st.session_state.get("image_uploaded"):
                st.session_state.image_uploaded = True
                st.session_state.pending_analysis = f"Analyze this image: data/{uploaded_file.name}"
        elif st.session_state.get("uploaded_image"):
            st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_container_width=True)
        else:
            st.info("Please upload a medical image to begin analysis")
            
        # Display reference images if they exist
        if st.session_state.get("show_references"):
            st.markdown("### 🩻 Reference Cases")
            st.markdown("These are sample cases with confirmed diagnoses:")
            
            cols = st.columns(4)
            image_paths = [
                "data/cancer/sample_1.jpg",
                "data/cancer/sample_2.jpg",
                "data/cancer/sample_3.jpg",
                "data/cancer/sample_4.jpg",
            ]
            for col, path in zip(cols, image_paths):
                with col:
                    try:
                        st.image(path, use_container_width=True, caption=path.split("/")[-1].replace("_", " ").title())
                    except:
                        st.warning(f"Could not load image: {path}")
    
    # Right column - Model chat and all responses
    with col2:
        st.header("Diagnosis Chat")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Process pending analysis if exists
        if st.session_state.get("pending_analysis"):
            analysis_prompt = st.session_state.pop("pending_analysis")
            with st.chat_message("user"):
                st.markdown(analysis_prompt.split(": ")[1])  # Show just the image name
            with st.chat_message("assistant"):
                asyncio.run(st.session_state.ui.process_message(analysis_prompt))
        
        # Chat input - always at the bottom
        if prompt := st.chat_input("Enter your message...", key="chat_input"):
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                asyncio.run(st.session_state.ui.process_message(prompt))
        
        # If we have a last message but it's not in the history, add it
        if st.session_state.last_message and (
            not st.session_state.messages or 
            st.session_state.messages[-1]["content"] != st.session_state.last_message
        ):
            st.session_state.messages.append({"role": "user", "content": st.session_state.last_message})
            st.rerun()

if __name__ == "__main__":
    main()