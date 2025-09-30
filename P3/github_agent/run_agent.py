"""
run_agent.py — Example runner that starts the GitHubDataAgent using a config file.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from github_agent import GitHubDataAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json", help="Path to agent config JSON")
    args = parser.parse_args()

    agent = GitHubDataAgent(args.config)
    agent.collect_data()

if __name__ == "__main__":
    main()
