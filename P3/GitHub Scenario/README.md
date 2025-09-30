# Scenario C — GitHub Repository Analytics (Starter)

This folder contains a minimal, **token-safe** setup for calling the GitHub API.

## Files
- `github_test.py` — quick smoke test; prints your current API rate limits.
- `github_analytics.py` — searches top repositories by stars for a language/topic and summarizes activity.
- `config.example.json` — template config with a **fake** placeholder token.
- `.env.example` — example env file you can copy to `.env` (if you use a dotenv loader) or source manually.
- `README.md` — this guide.

## Getting a token
1. Create a GitHub account (if needed).
2. Go to **Settings → Developer settings → Personal access tokens → Fine-grained tokens** (or classic).
3. Generate a token. For public data, **no special scopes are required**.
4. Copy the token and store it as an environment variable or in a local `config.json`.

## Never hardcode tokens
Prefer environment variables:
```bash
# macOS/Linux
export GITHUB_TOKEN="ghp_your_real_token_here"
# Windows (PowerShell)
$env:GITHUB_TOKEN="ghp_your_real_token_here"
```

Or a config file `config.json` (not committed to Git):
```json
{"github_token":"ghp_your_real_token_here"}
```

## Install dependencies
```bash
python -m venv .venv
# Activate the venv:
#   macOS/Linux: source .venv/bin/activate
#   Windows PowerShell: .\.venv\Scripts\Activate.ps1

pip install requests python-dotenv
```

## Run the test
```bash
python github_test.py                 # uses env var
python github_test.py --config config.json
```

## Run the analytics
Examples:
```bash
python github_analytics.py --language python --topN 25
python github_analytics.py --topic machine-learning --topN 30 --config config.json
python github_analytics.py --language javascript --topic visualization --topN 20
```

Outputs:
- `github_repo_summary.json`
- `github_repo_summary.csv`

> Note: `/stats` endpoints sometimes return HTTP 202 while GitHub prepares the data. The script retries a few times and then continues if not ready.
