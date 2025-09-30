"""
Exercise 2.3 — API with Parameters
----------------------------------
- Use Nager.Date public API to fetch holidays with parameters
- Print holiday names and dates for multiple countries
- Create and save a summary comparing holiday counts
API docs: https://date.nager.at/swagger/index.html
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import requests

# -----------------------------
# Logging setup
# -----------------------------
LOG_LEVEL = logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("holidays")


def get_public_holidays(country_code: str = "US", year: int = 2024, timeout: float = 15.0) -> Optional[List[dict]]:
    """
    Get public holidays for a specific country and year using Nager.Date API.
    Returns a list of holiday dicts or None on error.
    """
    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}"
    try:
        headers = {"Accept": "application/json", "User-Agent": "UVM-API-Exercise/1.0"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        holidays = resp.json()
        if not isinstance(holidays, list):
            logger.error("Unexpected JSON format for holidays: %s", holidays)
            return None
        return holidays
    except requests.exceptions.RequestException as e:
        logger.exception("Request failed for %s %s: %s", country_code, year, e)
        return None
    except ValueError:
        logger.exception("Response was not valid JSON for %s %s", country_code, year)
        return None


def extract_names_and_dates(holidays: List[dict]) -> List[dict]:
    """
    Return a simplified list of {date, name} from the API payload.
    Prefer 'localName' when present; fall back to 'name'.
    """
    simple: List[dict] = []
    for h in holidays:
        date = h.get("date")
        name = h.get("localName") or h.get("name")
        if date and name:
            simple.append({"date": date, "name": name})
    return simple


def compare_holiday_counts(countries: List[str], year: int) -> Dict[str, int]:
    """
    Return a mapping of country_code -> number of holidays.
    """
    counts: Dict[str, int] = {}
    for c in countries:
        holidays = get_public_holidays(c, year=year) or []
        counts[c] = len(holidays)
        logger.info("%s has %d public holidays in %d", c, counts[c], year)
    return counts


def save_summary(counts: Dict[str, int], path: Path, year: int) -> None:
    """
    Save the country counts summary as JSON.
    """
    payload = {"year": year, "summary": counts}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Saved summary to %s", path.resolve())


def main():
    year = 2024
    countries = ["US", "CA", "GB"]  # Test with 3 different countries

    # Fetch and print names/dates
    for code in countries:
        holidays = get_public_holidays(code, year) or []
        simple = extract_names_and_dates(holidays)
        print(f"\n=== {code} Holidays ({year}) — {len(simple)} total ===")
        for item in simple:
            print(f"{item['date']} — {item['name']}")

    # Compare counts and save summary
    counts = compare_holiday_counts(countries, year)
    save_summary(counts, Path("holidays_summary.json"), year)


if __name__ == "__main__":
    main()
