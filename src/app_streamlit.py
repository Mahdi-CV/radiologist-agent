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
    You are a medical assistant that helps analyze radiology images and share critical findings with peers.

    You have three tools:
    • analyse_image_base64(path: str) -> RadiologyReport
    • send_email(
        to: List[str],
        subject: str,
        body: str,
        cc: List[str] | None = [],
        bcc: List[str] | None = []
    ) -> "Gmail message ID"
    • show_reference_images_tool(path: str) -> dict  

    Workflow Rules:
    1. If the user has not specified an image path, ask:
    “Which image file should I analyse? (default = data/image.jpg)”

    2. Once you have the image path, call:
    {"name": "analyse_image_base64", "arguments": {"path": "<PATH_HERE>"}}

    3. The response from the tool will be a JSON containing:
    • critical: bool
    • diagnosis_description: str
    • clinical_recommendations: str

    4. After getting the response, show a **prettified summary**:
    - Diagnosis
    - Clinical Recommendations
    - Critical: Yes/No

    5. After displaying the diagnosis, ask EXACTLY this phrase:
    “Would you like to view reference images from similar confirmed cases?”
    
    6. If the user responds with ANY affirmative answer (e.g., "yes", "yep", "show them"), 
    call IMMEDIATELY:
    {"name": "show_reference_images_tool", "arguments": {"confirm": "yes"}}
    Examples of valid user confirmations:
    - User: "Yes"
    - User: "Sure, show them"
    - User: "Yep"
    
    → DO NOT say anything else after this. Just call the tool.

    7. If the result from the diagnosis report is critical (i.e., `critical: true`), ask:
    “This case is marked as critical. Would you like to send it for peer review?”

    8. If the user says yes or requests to share/email, generate a draft email (sender name is Dr. Mahdi Ghodsi) with the report contents. Show it in plain text and ask:
    “Would you like me to send this email?” 
    Once the draft is shown, wait for confirmation. If the user agrees, proceed to Step 9 to send the email.

    9. If the user confirms that they would like the email to be sent, ask for the recipient’s email address. 
    Then IMMEDIATELY call the following tool to send the email:
    {"name": "send_email", "arguments": {"to": ["<RECIPIENT_EMAIL>"], "subject": "<EMAIL_SUBJECT>", "body": "<EMAIL_BODY>"}}

    REMINDER: The tool is available and must be called in response to user confirmation.
    
    - Never move to sending email step before checking with the user if they want to see reference images.
    - Never send an email without explicit confirmation from the user.
    - Never guess or invent clinical data.
    Available tools:
    - analyse_image_base64
    - send_email
    - show_reference_images_tool

    """.strip()

async def build_orchestrator():
    fs_server = MCPServerStdio(
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

    system_prompt = SYSTEM_PROMPT

    return Agent(
        model=orch_model,
        mcp_servers=[fs_server, gmail_server],
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

    async def initialize(self):
        if self.orchestrator is None:
            print("⏳ Bootstrapping orchestrator...")
            with st.spinner("Initializing orchestrator..."):
                self.orchestrator = await build_orchestrator()
            print("✅ Orchestrator ready!")
        return self.orchestrator
    
    async def process_message(self, message: str):
        await self.initialize()

        # Record the user’s turn in the UI (you already do this)
        st.session_state.messages.append({"role": "user", "content": message})

        assistant_placeholder = st.empty()
        full_response = ""

        async with self.orchestrator.run_mcp_servers():
            # Pass in the prior turns (ModelMessage list) exactly as your CLI does
            async with self.orchestrator.iter(
                user_prompt=message,
                message_history=st.session_state.internal_history,  # this holds ModelMessage objects
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
                                        self.show_reference_images()


                    elif Agent.is_model_request_node(node):
                        async with node.stream(run.ctx) as s:
                            async for ev in s:

                                if isinstance(ev, PartDeltaEvent) and isinstance(ev.delta, TextPartDelta):
                                    #print(ev.delta.content_delta, end="", flush=True)
                                    full_response += ev.delta.content_delta
                                    assistant_placeholder.markdown(full_response)

            
            new_messages = run.result.all_messages()
            st.session_state.internal_history.extend(new_messages) 

        # And update the UI chat log
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        assistant_placeholder.markdown(full_response)
    
    def show_reference_images(self):
        print("we got here!!!!")
        # st.markdown("### 🩻 Reference Cases")
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
                st.image(path, use_container_width=True, caption=path.split("/")[-1].replace("_", " ").title())



def main():
    st.title("🏥 Radiology Assistant")
    st.markdown("Analyze medical images and manage communications")
    
    ui = StreamlitChatUI()
    
    # File upload
    uploaded_file = st.file_uploader("Upload medical image", type=["jpg", "png"])
    if uploaded_file and not st.session_state.get("image_uploaded"):
        # 1⃣  Persist inside DATA_DIR (radiology_async/data/)
        save_path = DATA_DIR / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # 2⃣  Analyse it – pass the *relative* path visible to the MCP FS server
        st.session_state.image_uploaded = True
        asyncio.run(
            ui.process_message(f"Analyze this image: data/{uploaded_file.name}")
        )

    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # Chat input
    if prompt := st.chat_input("Enter your message..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            asyncio.run(ui.process_message(prompt))

if __name__ == "__main__":
    main()