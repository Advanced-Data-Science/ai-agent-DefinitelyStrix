"""GitHubDataAgent — Data collection agent for Scenario C (fixed)

This agent implements:
- Configuration management (loads .env and config.json)
- Intelligent collection strategy with simple adaptation
- Data quality assessment (basic heuristics)
- Respectful collection with rate-limit awareness and jittered delays
- Logging and final report generation
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
import os
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

GITHUB_API = "https://api.github.com"


class GitHubDataAgent:
    def __init__(self, config_file: str):
        """Initialize the agent with a configuration JSON file."""
        self.config = self.load_config(config_file)
        self.setup_logging()
        self.data_store: List[Dict[str, Any]] = []
        self.collection_stats = {
            "start_time": datetime.utcnow().isoformat() + "Z",
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "apis_used": set(),
            "quality_scores": []
        }
        self.delay_multiplier = 1.0
        self.session = requests.Session()

        token = self.config.get("github_token") or os.getenv("GITHUB_TOKEN")
        if token:
            self.session.headers.update({
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "UVM-API-Exercise-GitHubDataAgent/1.0"
            })
        else:
            # logger isn't set up until setup_logging() runs, so set a temporary print fallback
            print("Warning: No GitHub token found in .env or config; unauthenticated requests will be heavily rate-limited.")

        outdir = Path(self.config.get("outdir", "."))
        outdir.mkdir(parents=True, exist_ok=True)
        self.outdir = outdir

        # Keep a simple in-memory rate-limit tracker (populate on each response if available)
        self.rate_limits: Dict[str, Any] = {}

    # ---------------- Configuration & logging ----------------
    def load_config(self, config_file: str) -> Dict[str, Any]:
        p = Path(config_file)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        with p.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        # Apply sensible defaults
        cfg.setdefault("language", None)
        cfg.setdefault("topic", None)
        cfg.setdefault("topN", 25)
        cfg.setdefault("base_delay", 1.0)
        cfg.setdefault("max_requests", 500)
        cfg.setdefault("outdir", "results")
        cfg.setdefault("use_fallback_api", False)
        return cfg

    def setup_logging(self) -> None:
        log_path = Path("data_collection.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_path, encoding="utf-8"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("GitHubDataAgent")

    # ---------------- Collection control ----------------
    def collection_complete(self) -> bool:
        desired = int(self.config.get("topN", 25))
        if len(self.data_store) >= desired:
            return True
        if self.collection_stats["total_requests"] >= int(self.config.get("max_requests", 500)):
            return True
        return False

    def get_success_rate(self) -> float:
        total = max(1, self.collection_stats["total_requests"])
        return self.collection_stats["successful_requests"] / total

    # ---------------- Core collection loop ----------------
    def collect_data(self) -> None:
        self.logger.info("Starting collection: language=%s topic=%s topN=%s",
                         self.config.get("language"), self.config.get("topic"), self.config.get("topN"))

        try:
            repos = self.search_top_repos(self.config.get("language"), self.config.get("topic"), int(self.config.get("topN", 25)))
            self.logger.info("Search returned %d candidates", len(repos))
            for item in repos:
                if self.collection_complete():
                    break

                # Adaptive check
                if self.get_success_rate() < 0.8:
                    self.adjust_strategy()

                full_name = item.get("full_name")
                if not full_name:
                    continue

                # Respect rate limits before performing extra requests
                self.check_rate_limits()

                # Request detailed repo info
                data = self.make_api_request(f"{GITHUB_API}/repos/{full_name}")
                self.collection_stats["total_requests"] += 1

                if data is not None:
                    self.collection_stats["successful_requests"] += 1
                    processed = self.process_data(data)
                    if self.validate_data(processed):
                        self.store_data(processed)
                else:
                    self.collection_stats["failed_requests"] += 1

                # Respectful delay between calls
                self.respectful_delay()
        except Exception as e:
            self.logger.exception("Unexpected error during collection: %s", e)

        # After loop finishes, compute final quality score and save outputs
        final_quality = self.assess_data_quality()
        self.collection_stats["data_quality_score"] = final_quality
        self.generate_final_report()

    # ---------------- API helpers ----------------
    def search_top_repos(self, language: Optional[str], topic: Optional[str], top_n: int) -> List[Dict[str, Any]]:
        q_parts = []
        if language:
            q_parts.append(f"language:{language}")
        if topic:
            q_parts.append(f"topic:{topic}")
        q = " ".join(q_parts) if q_parts else "stars:>100"
        params = {"q": q, "sort": "stars", "order": "desc", "per_page": min(100, top_n), "page": 1}
        results: List[Dict[str, Any]] = []
        while len(results) < top_n:
            try:
                r = self.session.get(f"{GITHUB_API}/search/repositories", params=params, timeout=20)
                self.collection_stats["total_requests"] += 1
                r.raise_for_status()
                data = r.json()
                items = data.get("items", [])
                if not items:
                    break
                results.extend(items)
                if len(items) < params["per_page"]:
                    break
                params["page"] += 1
            except requests.exceptions.RequestException as e:
                self.logger.warning("Search request failed: %s", e)
                break
        return results[:top_n]

    def make_api_request(self, url: str, params: Optional[Dict[str, Any]] = None, retries: int = 3) -> Optional[Dict[str, Any]]:
        for attempt in range(1, retries + 1):
            try:
                r = self.session.get(url, params=params, timeout=25)
                # Update rate limit snapshot when available
                if "X-RateLimit-Limit" in r.headers:
                    try:
                        self.rate_limits = {
                            "limit": int(r.headers.get("X-RateLimit-Limit", 0)),
                            "remaining": int(r.headers.get("X-RateLimit-Remaining", 0)),
                            "reset": int(r.headers.get("X-RateLimit-Reset", 0))
                        }
                    except Exception:
                        pass
                if r.status_code == 202:
                    self.logger.info("Received 202 (processing). Sleeping briefly before retry.")
                    time.sleep(1.5 * attempt)
                    continue
                r.raise_for_status()
                return r.json()
            except requests.exceptions.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                body = getattr(e.response, "text", "")
                self.logger.warning("HTTP error (%s) on %s: %s", status, url, (body or "")[:200])
                if status == 401:
                    self.logger.error("Authentication failed (401). Check token and permissions.")
                    return None
                if status in (403, 429):
                    backoff = 2 ** attempt
                    self.logger.warning("Rate limited, backing off for %ds", backoff)
                    time.sleep(backoff)
                    continue
                return None
            except requests.exceptions.RequestException as e:
                self.logger.warning("Request exception on attempt %d: %s", attempt, e)
                time.sleep(0.5 * attempt)
                continue
        return None

    # ---------------- Data handling ----------------
    def process_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "full_name": raw.get("full_name"),
            "html_url": raw.get("html_url"),
            "description": raw.get("description"),
            "language": raw.get("language"),
            "stargazers_count": raw.get("stargazers_count"),
            "forks_count": raw.get("forks_count"),
            "open_issues_count": raw.get("open_issues_count"),
            "watchers_count": raw.get("watchers_count"),
            "created_at": raw.get("created_at"),
            "pushed_at": raw.get("pushed_at"),
        }

    def validate_data(self, item: Dict[str, Any]) -> bool:
        if not item.get("full_name") or not item.get("html_url"):
            return False
        if item.get("stargazers_count") is None:
            return False
        return True

    def store_data(self, item: Dict[str, Any]) -> None:
        self.data_store.append(item)
        if len(self.data_store) % 10 == 0:
            self._flush_to_disk()

    def _flush_to_disk(self) -> None:
        json_path = self.outdir / "github_repo_summary.json"
        csv_path = self.outdir / "github_repo_summary.csv"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(self.data_store, f, indent=2, ensure_ascii=False)
        fieldnames = ["full_name","html_url","description","language","stargazers_count","forks_count","open_issues_count","watchers_count","created_at","pushed_at"]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in self.data_store:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        self.logger.info("Flushed %d records to disk (%s, %s)", len(self.data_store), json_path, csv_path)

    # ---------------- Data quality ----------------
    def assess_data_quality(self) -> float:
        if not self.data_store:
            return 0.0
        completeness = self.check_completeness()
        timeliness = self.check_timeliness()
        consistency = self.check_consistency()
        accuracy = self.check_accuracy()
        score = (completeness + timeliness + consistency + accuracy) / 4.0
        self.collection_stats["quality_scores"].append(score)
        return score

    def check_completeness(self) -> float:
        if not self.data_store:
            return 0.0
        required = ["full_name", "html_url", "stargazers_count"]
        ok = sum(1 for r in self.data_store if all(r.get(k) is not None for k in required))
        return ok / len(self.data_store)

    def check_timeliness(self) -> float:
        now = datetime.utcnow()
        count_recent = 0
        total = 0
        for r in self.data_store:
            pushed = r.get("pushed_at")
            total += 1
            if not pushed:
                continue
            try:
                pushed_dt = datetime.fromisoformat(pushed.replace("Z", "+00:00"))
                age_days = (now - pushed_dt).days
                if age_days <= 365:
                    count_recent += 1
            except Exception:
                continue
        return count_recent / max(1, total)

    def check_consistency(self) -> float:
        if not self.data_store:
            return 0.0
        languages = [r.get("language") for r in self.data_store if r.get("language")]
        if not languages:
            return 0.0
        top = Counter(languages).most_common(1)[0][1]
        return top / len(languages)

    def check_accuracy(self) -> float:
        # Placeholder: no external verification performed
        return 1.0

    # ---------------- Adaptive strategy ----------------
    def adjust_strategy(self) -> None:
        success_rate = self.get_success_rate()
        if success_rate < 0.5:
            self.delay_multiplier *= 2.0
            if self.config.get("use_fallback_api"):
                self.try_fallback_api()
        elif success_rate > 0.9:
            self.delay_multiplier = max(0.1, self.delay_multiplier * 0.8)
        self.delay_multiplier = min(max(self.delay_multiplier, 0.1), 16.0)
        self.logger.info("Adjusted strategy: delay_multiplier=%.2f success_rate=%.3f", self.delay_multiplier, success_rate)

    def try_fallback_api(self) -> None:
        self.logger.info("Fallback API requested but not configured. Skipping fallback.")

    # ---------------- Respectful collection ----------------
    def respectful_delay(self) -> None:
        base_delay = float(self.config.get("base_delay", 1.0))
        jitter = random.uniform(0.5, 1.5)
        delay = base_delay * float(self.delay_multiplier) * jitter
        if delay > 0:
            self.logger.debug("Sleeping for %.2fs (base=%.2f mult=%.2f jitter=%.2f)",
                              delay, base_delay, self.delay_multiplier, jitter)
            time.sleep(delay)

    def check_rate_limits(self) -> None:
        rl = self.rate_limits or {}
        try:
            remaining = int(rl.get("remaining", -1))
            limit = int(rl.get("limit", -1))
            reset = int(rl.get("reset", 0))
        except Exception:
            remaining = -1
            limit = -1
            reset = 0
        if remaining >= 0 and limit > 0:
            frac = remaining / float(limit)
            if frac < 0.1:
                wait = max(5, reset - int(time.time()))
                self.logger.warning("Approaching rate limit (%.2f%% remaining). Waiting %ds until reset.", frac*100.0, wait)
                time.sleep(wait + 1)
            elif frac < 0.3:
                self.logger.info("Rate limit low (%.2f%%). Increasing delay multiplier.", frac*100.0)
                self.delay_multiplier = min(self.delay_multiplier * 1.5, 16.0)

    # --------- Documentation & Quality Assurance helpers ---------

    def get_sources_used(self) -> List[str]:
        """Return the APIs/sources used during collection as a list."""
        apis = self.collection_stats.get("apis_used", [])
        # convert set to list if needed
        if isinstance(apis, (set, tuple)):
            return list(apis)
        if isinstance(apis, list):
            return apis
        return [str(apis)] if apis else []

    def generate_data_dictionary(self) -> Dict[str, Dict[str, str]]:
        """Generate a simple data dictionary from the first record (field -> type + description)."""
        if not self.data_store:
            return {}
        sample = self.data_store[0]
        dd: Dict[str, Dict[str, str]] = {}
        for k, v in sample.items():
            dtype = type(v).__name__
            # short human-friendly descriptions for common fields (customize if you like)
            desc_map = {
                "full_name": "Repository full name (owner/repo)",
                "html_url": "Repository web URL",
                "description": "Repository description",
                "language": "Primary language",
                "stargazers_count": "Number of stars",
                "forks_count": "Number of forks",
                "open_issues_count": "Open issues count",
                "watchers_count": "Watchers count",
                "created_at": "Repository creation timestamp",
                "pushed_at": "Last push timestamp"
            }
            dd[k] = {
                "type": dtype,
                "description": desc_map.get(k, "")
            }
        return dd

    def get_processing_log(self) -> List[str]:
        """Produce a small processing history from collection stats (events/timestamps)."""
        history = []
        start = self.collection_stats.get("start_time")
        if start:
            history.append(f"Collection started at {start}")
        total = self.collection_stats.get("total_requests", 0)
        succ = self.collection_stats.get("successful_requests", 0)
        fail = self.collection_stats.get("failed_requests", 0)
        history.append(f"Total requests: {total} (successful: {succ}, failed: {fail})")
        q_scores = self.collection_stats.get("quality_scores", [])
        if q_scores:
            history.append(f"Quality score samples (last 5): {q_scores[-5:]}")
        # include rate limit snapshot if available
        if self.rate_limits:
            rl = self.rate_limits
            history.append(f"Rate limit snapshot: remaining={rl.get('remaining')} limit={rl.get('limit')}")
        return history

    def calculate_final_quality_metrics(self) -> Dict[str, float]:
        """Return quality metrics computed from existing helper methods."""
        completeness = self.check_completeness()
        timeliness = self.check_timeliness()
        consistency = self.check_consistency()
        accuracy = self.check_accuracy()
        # average and also include the components
        overall = (completeness + timeliness + consistency + accuracy) / 4.0
        return {
            "completeness": round(completeness, 4),
            "timeliness": round(timeliness, 4),
            "consistency": round(consistency, 4),
            "accuracy": round(accuracy, 4),
            "overall_quality_score": round(overall, 4)
        }

    def analyze_completeness(self) -> Dict[str, Any]:
        """Per-field completeness analysis: percent of records with that field present."""
        if not self.data_store:
            return {}
        fields = set().union(*(r.keys() for r in self.data_store))
        totals = {f: 0 for f in fields}
        for r in self.data_store:
            for f in fields:
                if r.get(f) is not None:
                    totals[f] += 1
        n = len(self.data_store)
        return {f: {"present": totals[f], "pct_present": round(totals[f] / n, 4)} for f in sorted(fields)}

    def analyze_distribution(self) -> Dict[str, Any]:
        """Basic numeric distribution stats for numeric fields like stargazers_count, forks_count."""
        import statistics
        numeric_fields = ["stargazers_count", "forks_count", "open_issues_count", "watchers_count"]
        out = {}
        for f in numeric_fields:
            vals = [r.get(f) for r in self.data_store if isinstance(r.get(f), (int, float))]
            if not vals:
                continue
            out[f] = {
                "count": len(vals),
                "min": min(vals),
                "max": max(vals),
                "mean": round(statistics.mean(vals), 2),
                "median": round(statistics.median(vals), 2),
                "stdev": round(statistics.pstdev(vals), 2) if len(vals) > 1 else 0.0
            }
        return out

    def detect_anomalies(self) -> Dict[str, Any]:
        """Simple anomaly detection using mean +/- 3*stdev for numeric fields."""
        import statistics
        dist = self.analyze_distribution()
        anomalies = {}
        for f, stats in dist.items():
            if stats["count"] < 2:
                continue
            mean = stats["mean"]
            stdev = stats["stdev"]
            if stdev == 0:
                continue
            lower = mean - 3 * stdev
            upper = mean + 3 * stdev
            outliers = [r for r in self.data_store if
                        isinstance(r.get(f), (int, float)) and (r.get(f) < lower or r.get(f) > upper)]
            if outliers:
                anomalies[f] = {"outlier_count": len(outliers), "examples": outliers[:5]}
        return anomalies

    def generate_recommendations(self) -> List[str]:
        """Produce simple recommendations based on quality metrics."""
        metrics = self.calculate_final_quality_metrics()
        recs = []
        if metrics["completeness"] < 0.9:
            recs.append("Consider adding alternative data sources or increasing retries to improve completeness.")
        if metrics["timeliness"] < 0.8:
            recs.append(
                "Focus on actively maintained repositories or increase sampling frequency to improve timeliness.")
        if metrics["consistency"] < 0.75:
            recs.append("Narrow your search criteria (language/topic) to improve consistency of collected records.")
        if metrics["overall_quality_score"] < 0.8:
            recs.append("Run a longer collection or cross-validate records with another source to boost data quality.")
        if not recs:
            recs.append("Data quality looks good. Continue with current configuration.")
        return recs

    def create_readable_report(self, report: Dict[str, Any], path: Path) -> None:
        """Write a human-readable markdown report from the report dict."""
        md_lines = []
        md_lines.append(f"# Quality Report\n\nGenerated: {datetime.utcnow().isoformat()}Z\n\n")
        md_lines.append("## Summary\n")
        summary = report.get("summary", {})
        for k, v in summary.items():
            md_lines.append(f"- **{k}**: {v}\n")
        md_lines.append("\n## Completeness Analysis\n")
        comp = report.get("completeness_analysis", {})
        for field, info in comp.items():
            md_lines.append(f"- **{field}**: {info['present']} present ({info['pct_present'] * 100:.2f}%)\n")
        md_lines.append("\n## Distribution\n")
        dist = report.get("data_distribution", {})
        for f, s in dist.items():
            md_lines.append(f"- **{f}**: count={s['count']} mean={s['mean']} min={s['min']} max={s['max']}\n")
        md_lines.append("\n## Anomalies\n")
        anomalies = report.get("anomaly_detection", {})
        if not anomalies:
            md_lines.append("No major anomalies detected.\n")
        else:
            for f, a in anomalies.items():
                md_lines.append(f"- **{f}**: {a['outlier_count']} outliers, examples:\n")
                for ex in a.get("examples", [])[:3]:
                    md_lines.append(f"  - {ex.get('full_name', '<unknown>')} -> {ex.get(f)}\n")
        md_lines.append("\n## Recommendations\n")
        for r in report.get("recommendations", []):
            md_lines.append(f"- {r}\n")
        # write file
        path.write_text("".join(md_lines), encoding="utf-8")

    def generate_quality_report(self) -> Dict[str, Any]:
        """Create detailed quality assessment report (JSON + human-readable)."""
        report = {
            "summary": {
                "total_records": len(self.data_store),
                "collection_success_rate": round(self.get_success_rate(), 4),
                "overall_quality_score": self.collection_stats.get("data_quality_score", None)
            },
            "completeness_analysis": self.analyze_completeness(),
            "data_distribution": self.analyze_distribution(),
            "anomaly_detection": self.detect_anomalies(),
            "recommendations": self.generate_recommendations()
        }
        # Save JSON and markdown to outdir
        qr_json = self.outdir / "quality_report.json"
        qr_md = self.outdir / "quality_report.md"
        with qr_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        # human-readable
        self.create_readable_report(report, qr_md)
        self.logger.info("Wrote quality report: %s and %s", qr_json, qr_md)
        return report

    def generate_metadata(self) -> Dict[str, Any]:
        """Create comprehensive metadata for the collected dataset and save to outdir."""
        metadata = {
            "collection_info": {
                "collection_date": datetime.utcnow().isoformat() + "Z",
                "agent_version": "1.0",
                "collector": os.getenv("USER") or os.getenv("USERNAME") or "unknown",
                "total_records": len(self.data_store),
                "config": {k: v for k, v in self.config.items() if k != "github_token"}
            },
            "data_sources": self.get_sources_used(),
            "quality_metrics": self.calculate_final_quality_metrics(),
            "processing_history": self.get_processing_log(),
            "variables": self.generate_data_dictionary()
        }
        md_path = self.outdir / "dataset_metadata.json"
        with md_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        self.logger.info("Wrote dataset metadata: %s", md_path)
        return metadata

    def generate_collection_summary(self) -> Dict[str, Any]:
        """Compose and save a short text summary of the collection."""
        total = len(self.data_store)
        total_requests = self.collection_stats.get("total_requests", 0)
        successful = self.collection_stats.get("successful_requests", 0)
        failed = self.collection_stats.get("failed_requests", 0)
        quality = self.collection_stats.get("data_quality_score", None)
        lines = [
            f"Collection Summary - {datetime.utcnow().isoformat()}Z\n",
            f"Total records collected: {total}\n",
            f"Total requests: {total_requests} (successful: {successful}, failed: {failed})\n",
            f"Overall quality score: {quality}\n",
            "Sources used: " + ", ".join(self.get_sources_used()) + "\n",
            "\nRecommendations:\n"
        ]
        for r in self.generate_recommendations():
            lines.append(f"- {r}\n")
        txt_path = self.outdir / "collection_summary.txt"
        json_path = self.outdir / "collection_summary.json"
        txt_path.write_text("".join(lines), encoding="utf-8")
        json_path.write_text(json.dumps({
            "total_records": total,
            "total_requests": total_requests,
            "successful_requests": successful,
            "failed_requests": failed,
            "quality_score": quality,
            "recommendations": self.generate_recommendations()
        }, indent=2), encoding="utf-8")
        self.logger.info("Wrote collection summary: %s and %s", txt_path, json_path)
        return {"txt": str(txt_path), "json": str(json_path)}

    # JSON-safe helper used when writing collection_stats (keeps previous behaviour)
    def _make_json_safe(self, obj):
        """Recursively convert sets/tuples/datetimes to json-serializable types."""
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._make_json_safe(v) for v in obj]
        if isinstance(obj, set):
            return [self._make_json_safe(v) for v in obj]
        try:
            import datetime as _dt
            if isinstance(obj, _dt.datetime):
                return obj.isoformat()
        except Exception:
            pass
        return obj

    # Replace or update the existing generate_final_report to call the docs generation
    def generate_final_report(self) -> None:
        """Write final outputs, collection stats, metadata, quality report, and a collection summary."""
        # Final flush in case any last items are unwritten
        self._flush_to_disk()

        # Save stats (JSON-safe)
        stats_path = self.outdir / "collection_stats.json"
        safe_stats = self._make_json_safe(self.collection_stats)
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(safe_stats, f, indent=2, ensure_ascii=False)
        self.logger.info("Saved collection stats: %s", stats_path)

        # Generate dataset metadata
        try:
            self.generate_metadata()
        except Exception as e:
            self.logger.warning("Failed to generate metadata: %s", e)

        # Generate quality report (JSON + human readable)
        try:
            self.generate_quality_report()
        except Exception as e:
            self.logger.warning("Failed to generate quality report: %s", e)

        # Generate a short collection summary
        try:
            self.generate_collection_summary()
        except Exception as e:
            self.logger.warning("Failed to generate collection summary: %s", e)

        self.logger.info("Final reporting complete.")
