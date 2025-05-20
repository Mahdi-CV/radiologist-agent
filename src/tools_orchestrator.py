from pydantic_ai import Tool
import base64, pathlib
from radiology_agent import radiology_agent
from jinja2 import Template

DEFAULT_SUBJECT   = "Peer Review Request"

EMAIL_TMPL = Template("""
<!DOCTYPE html>
<html>
  <body style="font-family:sans-serif; line-height:1.4; max-width:600px; margin:0 auto;">
                
    <div style="background:#f8f9fa; padding:1rem; border-radius:4px;">
      <p><strong>Diagnosis:</strong> {{ diagnosis }}</p>
      <p><strong>Recommendations:</strong> {{ recs }}</p>
      <p><strong>Critical Finding:</strong> {{ critical }}</p>
    </div>

    {% if img_tag %}
    <div style="margin:2rem 0; text-align:center">
      {{ img_tag | safe }}
      <p style="color:#666; font-size:0.9em">Attached Imaging</p>
    </div>
    {% endif %}

    <p>敬祝 平安順心，<br>張醫師 敬上</p>
    <p>Regards，<br>Dr. Chang</p>
                      
                      
  </body>
</html>
""")

@Tool
def draft_report_html(
    diagnosis: str,
    recs: str,
    critical: bool,
    img_path: str | None = None,
) -> dict:
    """
    Produce the JSON payload that send_email expects:
      {to, subject, mimeType, body}
    Does NOT embed base64 image anymore.
    """
    img_tag = ""
    if img_path and pathlib.Path(img_path).exists():
        b64 = base64.b64encode(pathlib.Path(img_path).read_bytes()).decode()
        img_tag = f'<div style="margin:1rem 0"><img src="data:image/jpeg;base64,{b64}" style="max-width:100%; height:auto; border-radius:4px;"></div>'


    body_html = EMAIL_TMPL.render(
        diagnosis=diagnosis,
        recs=recs,
        critical=critical,
        img_tag=img_tag
    )

    return {
        "subject":  DEFAULT_SUBJECT,
        "mimeType": "text/html",
        "body":     body_html,
    }

@Tool
async def analyse_image_base64(path: str | None = None) -> dict:
    """
    Analyse a JPEG with the radiology VLM and return a structured report.

    Parameters
    ----------
    path : str | None
        Path to a JPEG. 
    """

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
