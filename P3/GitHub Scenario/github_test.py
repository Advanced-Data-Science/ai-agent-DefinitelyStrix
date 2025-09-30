"""
github_test.py — Scenario C quick test (auto-load .env)
------------------------------------------------------
- Loads .env automatically using python-dotenv (if present).
- Token sources checked in order:
    1) Environment variable GITHUB_TOKEN (including .env loaded)
    2) Config JSON passed via --config
- Prints a masked preview of token (safe) and calls /rate_limit
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

# load .env if present
load_dotenv()

GITHUB_API = "https://api.github.com"

def load_token(config_path: str | None) -> str | None:
    token = os.getenv("GITHUB_TOKEN")
    if token:
        return token.strip()

    if config_path and Path(config_path).is_file():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                token = data.get("github_token")
                if token:
                    return token.strip()
        except Exception as e:
            print(f"[ERROR] Could not read config file: {e}", file=sys.stderr)
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config.json containing {'github_token':'...'}")
    args = parser.parse_args()

    token = load_token(args.config)
    if not token:
        print("[ERROR] No token found. Set env var GITHUB_TOKEN, use a .env file, or provide --config config.json.", file=sys.stderr)
        sys.exit(1)

    # Safe masked preview
    preview = f"{token[:6]}...{token[-4:]}" if len(token) > 10 else token
    print(f"[INFO] Using token preview: {preview} (length={len(token)})")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "UVM-API-Exercise/1.0"
    }

    url = f"{GITHUB_API}/rate_limit"
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        core = data.get("resources", {}).get("core", {})
        search = data.get("resources", {}).get("search", {})
        print("✅ GitHub API reachable.")
        print(f"Core remaining: {core.get('remaining')} / {core.get('limit')} (resets at {core.get('reset')})")
        print(f"Search remaining: {search.get('remaining')} / {search.get('limit')} (resets at {search.get('reset')})")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}", file=sys.stderr)
        if hasattr(e, 'response') and e.response is not None:
            try:
                print(f"[DEBUG] Status: {e.response.status_code} Body: {e.response.text}", file=sys.stderr)
            except Exception:
                pass
        sys.exit(2)

if __name__ == "__main__":
    main()
