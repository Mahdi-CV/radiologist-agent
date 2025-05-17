import os
from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pathlib import Path

# Load environment variables from .env
load_dotenv()
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

# ───────────────────────────────────────────────
# Radiology Agent Config
# ───────────────────────────────────────────────
rad_provider = OpenAIProvider(
    base_url=os.getenv("RAD_API_BASE"),
    api_key=os.getenv("RAD_API_KEY"),
)
vlm_model = OpenAIModel(os.getenv("RAD_MODEL_NAME"), provider=rad_provider)

# ───────────────────────────────────────────────
# Orchestrator Agent Config
# ───────────────────────────────────────────────
orc_provider = OpenAIProvider(
    base_url=os.getenv("ORCH_API_BASE"),
    api_key=os.getenv("ORCH_API_KEY"),
)
orch_model = OpenAIModel(os.getenv("ORCH_MODEL_NAME"), provider=orc_provider)

