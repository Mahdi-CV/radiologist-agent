# Radiologist Agent

This project analyzes radiology images using an orchestrator agent and a fine-tuned radiologist vision-language model. If the case is flagged as critical, the system prompts the user to optionally send a peer-review email through Gmail.

## Requirements

- Python 3.11+
- `venv`-based virtual environment (recommended)
- `.env` file with model server configuration
- Access to two OpenAI-compatible inference servers:
  - One hosting the fine-tuned radiology model
  - One hosting the orchestrator model (e.g., Qwen3)
- Gmail account with enabled OAuth credentials for email notifications

## Quickstart

Clone the repository and install dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/radiologist-agent.git
cd radiologist-agent

python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 1. Set up environment variables

Copy the example `.env` file and fill in your model server details:

```bash
cp .env.template .env
```

Edit `.env`:

```env
RAD_API_BASE=http://<radiology-model-server>:<port>/v1
RAD_API_KEY=your_key
RAD_MODEL_NAME=your_model_name

ORCH_API_BASE=http://<orchestrator-server>:<port>/v1
ORCH_API_KEY=your_key
ORCH_MODEL_NAME=your_model_name
```

### 2. Authenticate Gmail for peer-review email feature

#### Step A: Set up Gmail OAuth Credentials
1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select a project
3. Enable the `Gmail API` (under `APIs & Services` > `Library`)
4. Click Create `Credentials` > `OAuth client ID`
5. Choose `Desktop` app
6. Give it a name and click `Create`, then Download the `JSON` key file
7. Save the downloaded file as:

```bash
src/gcp-oauth.keys.json
```
8. Under project `Overview` -> `Audeince` -> `Test users` add your Google account that is going to be used in the next step.

#### Step B: Install the MCP Gmail Server and Authenticate your account
If you don't have npx installed, first install Node.js (which includes npx) from [nodejs.org](https://nodejs.org).

Once Node.js is installed, you can install the Gmail MCP server:

```bash
mkdir -p ~/.gmail-mcp
cd src
npx @gongrzhe/server-gmail-autoauth-mcp auth 
```
You will be prompted to follow a link and authenticate your account. This will store the Gmail credentials globally in `~/.gmail-mcp/`.

### 3. Launch the UI

```bash
streamlit run src/app_streamlit.py
```

## Project Structure

```text
radiologist-agent/
├── data/                        # Sample input images
├── src/
│   ├── config.py                # Loads model/server config from .env
│   ├── orchestrator_agent.py   # Qwen3 orchestrator logic
│   ├── radiology_agent.py      # Radiology VLM agent
│   ├── schema.py               # Pydantic schema for report
│   ├── tools_orchestrator.py   # Tool definitions (image analysis, email)
│   ├── app_streamlit.py        # Streamlit UI entrypoint
│   ├── app_test_cli.py         # Standalone email test
│   └── gcp-oauth.keys.json     # OAuth key (local only)
├── .env.template               # Env config template
├── requirements.txt
├── LICENSE
└── README.md
```

## Notes

- The Gmail MCP server must be authenticated before email functionality can work.
- This repo does not include the fine-tuned models; you must deploy them separately and configure their endpoints in `.env`.
