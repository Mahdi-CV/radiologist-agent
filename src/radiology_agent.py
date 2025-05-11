# radiology_agent.py (concise + echo patient ID)
from pydantic_ai import Agent
from config import vlm_model
from schema import RadiologyReport
import textwrap


system_prompt = ("""
    You are an **AI radiologist**. Analyze the image and create a structured report in the exact JSON schema provided.

    Guidelines:
    • **diagnosis_description** – abnormal findings + anatomical structures visualized  
    • **clinical_recommendations** – clinical‑correlation suggestions  
    • Set `"critical": true` for malignancy, acute PE, aortic dissection, active bleed, tension pneumothorax, or any life‑threatening finding.

    Respond **only** with the single JSON tool‑call.

    Format reference:
    ```json
    { "...": "See RadiologyReport Pydantic schema" }"""
)


radiology_agent = Agent(
    model=vlm_model,
    output_type=RadiologyReport,
    system_prompt=system_prompt,
    output_retries=3,
    temperature=0.0,
)
