"""
load_jmdict_db.py — Phase 3.1 data pipeline
Loads the merged CSV produced by build_jmdict_db.py into the
Supabase `jmdict_entries` table (Tokyo region).

Run from repo root after build_jmdict_db.py has completed:
    python scripts/load_jmdict_db.py \
        --csv data/jmdict_merged.csv \
        --supabase-url https://your-project.supabase.co \
        --supabase-key your-service-role-key

Required Supabase table (create this first — see SQL below):
    create table jmdict_entries (
      id text primary key,
      text text not null,
      reading text not null,
      jlpt_level text,
      pitch_accent text,
      meanings text[] not null,
      parts_of_speech text[] not null
    );
    create index on jmdict_entries (text);
    create index on jmdict_entries (reading);
    alter table jmdict_entries disable row level security;

Strategy:
    Upsert in batches of BATCH_SIZE rows.
    Supabase upsert (on_conflict=id) handles re-runs safely —
    re-running this script after a partial load will fill in missing rows
    without duplicating completed ones.

Dependencies:
    pip install supabase python-dotenv
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 500        # rows per upsert call — safe for Supabase free tier
RETRY_LIMIT = 3         # max retries per batch on transient failure
RETRY_DELAY_SEC = 2.0   # seconds to wait between retries

# ---------------------------------------------------------------------------
# CSV → row conversion
# ---------------------------------------------------------------------------


def csv_row_to_supabase_row(row: dict[str, str]) -> dict:
    """
    Convert a CSV row (all strings) to a Supabase-ready dict.
    meanings and parts_of_speech are stored as text[] in Postgres —
    pass them as Python lists; the supabase-py client serialises correctly.
    jlpt_level and pitch_accent are nullable — convert empty string to None.
    """
    return {
        "id": row["id"],
        "text": row["text"],
        "reading": row["reading"],
        "jlpt_level": row["jlpt_level"] or None,
        "pitch_accent": row["pitch_accent"] or None,
        "meanings": json.loads(row["meanings"]),
        "parts_of_speech": json.loads(row["parts_of_speech"]),
    }


def count_csv_rows(csv_path: Path) -> int:
    """Count data rows in CSV (excluding header). Used for progress display."""
    with csv_path.open(encoding="utf-8", newline="") as fh:
        return sum(1 for _ in fh) - 1  # subtract header


def iter_csv_batches(
    csv_path: Path,
    batch_size: int,
) -> list[list[dict]]:
    """
    Read the CSV and yield lists of Supabase-ready row dicts,
    each list up to batch_size in length.
    """
    batch: list[dict] = []
    with csv_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for csv_row in reader:
            batch.append(csv_row_to_supabase_row(csv_row))
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Supabase upsert with retry
# ---------------------------------------------------------------------------


def upsert_batch_with_retry(
    client,
    batch: list[dict],
    batch_num: int,
    retry_limit: int,
    retry_delay: float,
) -> None:
    """
    Upsert a single batch into jmdict_entries.
    Retries on failure up to retry_limit times.
    Raises RuntimeError if all retries are exhausted — caller decides whether to abort.
    """
    last_error: Exception | None = None
    for attempt in range(1, retry_limit + 1):
        try:
            client.table("jmdict_entries").upsert(batch, on_conflict="id").execute()
            return
        except Exception as exc:
            last_error = exc
            log.warning(
                "Batch %d: attempt %d/%d failed: %s",
                batch_num,
                attempt,
                retry_limit,
                exc,
            )
            if attempt < retry_limit:
                time.sleep(retry_delay)

    raise RuntimeError(
        f"Batch {batch_num}: all {retry_limit} attempts failed. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Load orchestrator
# ---------------------------------------------------------------------------


def load_csv_to_supabase(
    csv_path: Path,
    supabase_url: str,
    supabase_key: str,
    batch_size: int = BATCH_SIZE,
    dry_run: bool = False,
) -> None:
    """
    Main load function. Reads CSV, upserts in batches.

    dry_run=True: reads and validates CSV rows but does not write to Supabase.
    Useful to confirm the CSV is well-formed before a full load.
    """
    # Lazy import — keeps startup fast if user just runs --help
    try:
        from supabase import create_client
    except ImportError as exc:
        raise RuntimeError(
            "supabase-py is not installed. Run: pip install supabase"
        ) from exc

    log.info("=== load_jmdict_db.py ===")
    log.info("CSV: %s", csv_path)
    log.info("Supabase URL: %s", supabase_url)
    log.info("Batch size: %d | Dry run: %s", batch_size, dry_run)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    total_rows = count_csv_rows(csv_path)
    log.info("CSV row count: %d", total_rows)

    client = create_client(supabase_url, supabase_key) if not dry_run else None

    batches_loaded = 0
    rows_loaded = 0
    start_time = time.monotonic()

    for batch in iter_csv_batches(csv_path, batch_size):
        batches_loaded += 1
        rows_in_batch = len(batch)

        if dry_run:
            # Validate structure only — confirm serialisation succeeds
            _ = [json.dumps(row) for row in batch]
            log.info(
                "  [DRY RUN] Batch %d: %d rows validated", batches_loaded, rows_in_batch
            )
        else:
            upsert_batch_with_retry(
                client=client,
                batch=batch,
                batch_num=batches_loaded,
                retry_limit=RETRY_LIMIT,
                retry_delay=RETRY_DELAY_SEC,
            )
            rows_loaded += rows_in_batch
            pct = 100 * rows_loaded / max(total_rows, 1)
            elapsed = time.monotonic() - start_time
            rate = rows_loaded / max(elapsed, 0.001)
            remaining_sec = (total_rows - rows_loaded) / max(rate, 1)
            log.info(
                "  Batch %d done — %d/%d rows (%.1f%%) — %.0f rows/s — ~%.0fs remaining",
                batches_loaded,
                rows_loaded,
                total_rows,
                pct,
                rate,
                remaining_sec,
            )

    elapsed_total = time.monotonic() - start_time
    if dry_run:
        log.info("Dry run complete: %d batches, %d rows validated in %.1fs", batches_loaded, total_rows, elapsed_total)
    else:
        log.info(
            "Load complete: %d rows upserted in %d batches in %.1fs",
            rows_loaded,
            batches_loaded,
            elapsed_total,
        )
        log.info("Confirm in Supabase: SELECT COUNT(*) FROM jmdict_entries;")


# ---------------------------------------------------------------------------
# Post-load verification
# ---------------------------------------------------------------------------


def verify_spot_check(client, spot_terms: list[str]) -> None:
    """
    Quick smoke test: look up a handful of known words to confirm
    the load landed correctly. Logs PASS / FAIL per term.
    """
    log.info("--- Spot check ---")
    for term in spot_terms:
        result = client.table("jmdict_entries").select("id, text, jlpt_level, pitch_accent").eq("text", term).limit(1).execute()
        if result.data:
            row = result.data[0]
            log.info("  ✓ %s → jlpt=%s pitch=%s", row["text"], row["jlpt_level"], row["pitch_accent"])
        else:
            log.warning("  ✗ %s → not found", term)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Default spot-check terms — common N4/N5 words we expect to find
DEFAULT_SPOT_TERMS = ["食べる", "飲む", "電車", "勉強", "友達", "会社"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load JMdict merged CSV into Supabase jmdict_entries table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        default=Path("data/jmdict_merged.csv"),
        type=Path,
        metavar="PATH",
        help="Path to merged CSV from build_jmdict_db.py (default: data/jmdict_merged.csv)",
    )
    parser.add_argument(
        "--supabase-url",
        type=str,
        metavar="URL",
        help="Supabase project URL. Can also be set via SUPABASE_URL env var.",
    )
    parser.add_argument(
        "--supabase-key",
        type=str,
        metavar="KEY",
        help="Supabase service role key. Can also be set via SUPABASE_SERVICE_KEY env var. "
             "Use the SERVICE ROLE key (not anon), required for upsert on a table with RLS disabled.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        metavar="N",
        help=f"Rows per upsert batch (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate CSV rows without writing to Supabase.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After load, run a spot check on common words to confirm rows landed.",
    )
    return parser


def resolve_credentials(args: argparse.Namespace) -> tuple[str, str]:
    """
    Resolve Supabase URL and key from CLI args or environment.
    .env is loaded if present so this works when called from the repo root.
    """
    load_dotenv()  # no-op if .env not present

    url = args.supabase_url or os.environ.get("SUPABASE_URL", "")
    key = args.supabase_key or os.environ.get("SUPABASE_SERVICE_KEY", "")

    missing = []
    if not url:
        missing.append("--supabase-url / SUPABASE_URL")
    if not key:
        missing.append("--supabase-key / SUPABASE_SERVICE_KEY")
    if missing:
        raise ValueError(f"Missing required credentials: {', '.join(missing)}")

    return url, key


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        supabase_url, supabase_key = resolve_credentials(args)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    try:
        load_csv_to_supabase(
            csv_path=args.csv,
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        log.error("%s", exc)
        return 1

    if args.verify and not args.dry_run:
        try:
            from supabase import create_client
            client = create_client(supabase_url, supabase_key)
            verify_spot_check(client, DEFAULT_SPOT_TERMS)
        except Exception as exc:
            log.warning("Spot check failed: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
