"""
Exercise 2.2 — Your First API Call
----------------------------------
- Fetch 5 different cat facts from https://catfact.ninja/fact
- Add robust error handling and logging
- Save the facts to a JSON file
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Set

import requests


# -----------------------------
# Logging setup
# -----------------------------
LOG_LEVEL = logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("catfacts")


def get_cat_fact(timeout: float = 10.0) -> Optional[str]:
    """
    Make a single GET request to catfact.ninja to retrieve a random fact.

    Returns:
        A cat fact string on success, or None if anything went wrong.
    """
    url = "https://catfact.ninja/fact"
    try:
        headers = {"Accept": "application/json", "User-Agent": "UVM-API-Exercise/1.0"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()

        # Validate JSON structure
        data = resp.json()
        if not isinstance(data, dict) or "fact" not in data:
            logger.error("Unexpected JSON format: %s", data)
            return None

        fact = data["fact"]
        if not isinstance(fact, str) or not fact.strip():
            logger.error("Empty/invalid 'fact' field: %r", fact)
            return None

        return fact.strip()

    except requests.exceptions.Timeout:
        logger.exception("Request timed out")
    except requests.exceptions.RequestException as e:
        logger.exception("Network/HTTP error: %s", e)
    except ValueError:
        logger.exception("Response was not valid JSON")
    except Exception as e:  # Catch-all to avoid crashing the script
        logger.exception("Unexpected error: %s", e)

    return None


def get_multiple_cat_facts(
    n: int = 5,
    ensure_unique: bool = True,
    max_attempts: int = 20,
    backoff_seconds: float = 0.5,
) -> List[str]:
    """
    Collect n cat facts, optionally enforcing uniqueness.

    Args:
        n: number of facts desired.
        ensure_unique: if True, avoid duplicates observed during this run.
        max_attempts: safety cap to avoid infinite loops when API repeats results.
        backoff_seconds: delay between calls to be polite and reduce rate limiting.

    Returns:
        List of facts (length may be < n if API keeps failing/repeating).
    """
    results: List[str] = []
    seen: Set[str] = set()

    attempts = 0
    while len(results) < n and attempts < max_attempts:
        attempts += 1
        fact = get_cat_fact()
        if fact is None:
            logger.warning("Skipping due to previous error (attempt %d)", attempts)
        else:
            if ensure_unique and fact in seen:
                logger.info("Duplicate fact received; trying again.")
            else:
                results.append(fact)
                seen.add(fact)
                logger.info("Collected fact %d/%d", len(results), n)

        time.sleep(backoff_seconds)

    if len(results) < n:
        logger.warning(
            "Requested %d facts but collected %d after %d attempts.",
            n,
            len(results),
            attempts,
        )
    return results


def save_facts_to_json(facts: List[str], path: Path) -> None:
    """
    Save the list of facts to a JSON file with a simple schema.
    """
    payload = {
        "source": "https://catfact.ninja/fact",
        "count": len(facts),
        "facts": facts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d facts to %s", len(facts), path.resolve())


def main():
    facts = get_multiple_cat_facts(n=5, ensure_unique=True)
    for i, fact in enumerate(facts, 1):
        print(f"{i}. {fact}")

    out_path = Path("cat_facts.json")
    save_facts_to_json(facts, out_path)


if __name__ == "__main__":
    main()
