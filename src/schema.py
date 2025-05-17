from pydantic import BaseModel, Field
from typing import Optional

class RadiologyReport(BaseModel):
    """Structured output produced by the vision model."""
    critical: bool = Field(
        description="True if immediate clinical action is required"
    )
    diagnosis_description: str = Field(
        description="In-depth description of the diagnosis and its implications"
    )
    clinical_recommendations: str = Field(
        description="Concrete next steps (imaging, biopsy, labs, referral)"
    )