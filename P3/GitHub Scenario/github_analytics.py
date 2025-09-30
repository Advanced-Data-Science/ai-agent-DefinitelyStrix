"""
github_analytics.py — Scenario C: GitHub Repository Analytics (auto-load .env)
-----------------------------------------------------------------------------
- Loads .env automatically using python-dotenv (if present).
- Token sources checked in order:
    1) Environment variable GITHUB_TOKEN (including .env loaded)
    2) Config JSON passed via --config
- Usage examples:
    python github_analytics.py --language python --topN 25
    python github_analytics.py --topic machine-learning --topN 30 --config config.json
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

# load .env if present
load_dotenv()

GITHUB_API = "https://api.github.com"

def load_token(config_path: str | None) -> Optional[str]:
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
        except Exception:
            pass
    return None

def make_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "UVM-API-Exercise/1.0"
    }

def github_get(url: str, headers: Dict[str, str], params: Dict[str, Any] | None = None, timeout: int = 20) -> requests.Response:
    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp

def search_top_repos(language: Optional[str], topic: Optional[str], top_n: int, headers: Dict[str, str]) -> List[Dict[str, Any]]:
    assert language or topic, "Provide at least --language or --topic"
    q_parts = []
    if language:
        q_parts.append(f"language:{language}")
    if topic:
        q_parts.append(f"topic:{topic}")
    q = " ".join(q_parts) if q_parts else "stars:>100"

    per_page = min(100, top_n)
    params = {"q": q, "sort": "stars", "order": "desc", "per_page": per_page, "page": 1}

    results: List[Dict[str, Any]] = []
    while len(results) < top_n:
        r = github_get(f"{GITHUB_API}/search/repositories", headers, params)
        data = r.json()
        items = data.get("items", [])
        if not items:
            break
        results.extend(items)
        if len(items) < per_page:
            break
        params["page"] += 1

    return results[:top_n]

def get_basic_repo_info(full_name: str, headers: Dict[str, str]) -> Dict[str, Any]:
    r = github_get(f"{GITHUB_API}/repos/{full_name}", headers)
    j = r.json()
    return {
        "full_name": j.get("full_name"),
        "html_url": j.get("html_url"),
        "description": j.get("description"),
        "language": j.get("language"),
        "stargazers_count": j.get("stargazers_count"),
        "forks_count": j.get("forks_count"),
        "open_issues_count": j.get("open_issues_count"),
        "watchers_count": j.get("watchers_count"),
        "created_at": j.get("created_at"),
        "updated_at": j.get("updated_at"),
        "pushed_at": j.get("pushed_at"),
    }

def get_commit_activity(full_name: str, headers: Dict[str, str], retries: int = 3, delay: float = 1.5) -> Optional[List[Dict[str, Any]]]:
    url = f"{GITHUB_API}/repos/{full_name}/stats/commit_activity"
    for i in range(retries):
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code == 202:
            time.sleep(delay)
            continue
        r.raise_for_status()
        return r.json()
    return None

def get_contributors(full_name: str, headers: Dict[str, str], retries: int = 3, delay: float = 1.5) -> Optional[List[Dict[str, Any]]]:
    url = f"{GITHUB_API}/repos/{full_name}/stats/contributors"
    for i in range(retries):
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code == 202:
            time.sleep(delay)
            continue
        r.raise_for_status()
        return r.json()
    return None

def summarize_stats(commit_activity: Optional[List[Dict[str, Any]]], contributors: Optional[List[Dict[str, Any]]]) -> Tuple[int, int]:
    total_commits = 0
    if commit_activity:
        total_commits = sum(week.get("total", 0) for week in commit_activity)
    contributor_count = 0
    if contributors:
        contributor_count = len(contributors)
    return total_commits, contributor_count

def main():
    parser = argparse.ArgumentParser(description="Analyze top GitHub repositories by language/topic.")
    parser.add_argument("--language", type=str, default=None, help="Programming language to filter by (e.g., python)")
    parser.add_argument("--topic", type=str, default=None, help="Topic to filter by (e.g., machine-learning)")
    parser.add_argument("--topN", type=int, default=25, help="How many repositories to include (<=50 recommended)")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json containing {'github_token': '...'}")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory for results")
    args = parser.parse_args()

    token = load_token(args.config)
    if not token:
        print("[ERROR] No token found. Set env var GITHUB_TOKEN, use a .env file, or provide --config config.json with {'github_token': '...'}")
        raise SystemExit(1)

    headers = make_headers(token)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    repos = search_top_repos(args.language, args.topic, args.topN, headers)
    print(f"Found {len(repos)} repositories. Fetching details...")

    rows: List[Dict[str, Any]] = []
    for i, item in enumerate(repos, 1):
        full_name = item["full_name"]
        basic = get_basic_repo_info(full_name, headers)
        commit_activity = get_commit_activity(full_name, headers)
        contributors = get_contributors(full_name, headers)
        total_commits, contributor_count = summarize_stats(commit_activity, contributors)

        row = {
            **basic,
            "commits_last_year": total_commits,
            "contributors_count": contributor_count,
        }
        rows.append(row)
        print(f"[{i}/{len(repos)}] {full_name} ★{basic['stargazers_count']} commits(1y)={total_commits} contributors={contributor_count}")

    json_path = outdir / "github_repo_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    csv_path = outdir / "github_repo_summary.csv"
    fieldnames = [
        "full_name","html_url","description","language",
        "stargazers_count","forks_count","open_issues_count","watchers_count",
        "created_at","updated_at","pushed_at",
        "commits_last_year","contributors_count"
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\nSaved:\n- {json_path.resolve()}\n- {csv_path.resolve()}")
    print("Done.")

if __name__ == "__main__":
    main()
