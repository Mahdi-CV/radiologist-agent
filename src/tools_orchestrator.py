from pydantic_ai import Tool
import base64, pathlib
from radiology_agent import radiology_agent

DEFAULT_IMAGE = pathlib.Path("data/image.jpg").resolve()

@Tool
async def analyse_image_base64(path: str | None = None) -> dict:
    """
    Analyse a JPEG with the radiology VLM and return a structured report.

    Parameters
    ----------
    path : str | None
        Path to a JPEG.  If omitted ― or equal to a known placeholder like
        'str' ― we fall back to data/image.jpg.
    """
    if path in (None, "", "str"):
        path = DEFAULT_IMAGE
    else:
        path = pathlib.Path(path).expanduser().resolve()

    b64 = base64.b64encode(path.read_bytes()).decode()

    run = await radiology_agent.run(
        messages=[{
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            }],
        }]
    )
    return run.output.model_dump()

@Tool
def show_reference_images_tool(confirm: str = "yes") -> dict:
    """
    Display reference medical images from similar confirmed cases. 
    Call this ONLY after the user explicitly asks to see reference images.
    """
    return {"action": "show_reference_images", "status": "success"}
