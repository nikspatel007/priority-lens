#!/usr/bin/env python3
"""
One-shot data onboarding pipeline.

Processes a Gmail MBOX export through the complete ML pipeline:
1. Parse MBOX → JSONL
2. Import to PostgreSQL
3. Build thread relationships
4. Compute ML features (30 dimensions)
5. Generate embeddings (OpenAI)
6. Rule-based AI classification
7. LLM classification (optional)

Usage:
    python scripts/onboard_data.py /path/to/your.mbox
    python scripts/onboard_data.py /path/to/your.mbox --workers 10
    python scripts/onboard_data.py /path/to/your.mbox --skip-llm
    python scripts/onboard_data.py /path/to/your.mbox --start-from 5
    python scripts/onboard_data.py --status

Examples:
    # Full pipeline
    python scripts/onboard_data.py ~/Downloads/gmail.mbox

    # Skip expensive LLM step
    python scripts/onboard_data.py ~/Downloads/gmail.mbox --skip-llm

    # Resume from embeddings
    python scripts/onboard_data.py ~/Downloads/gmail.mbox --start-from 5
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Configuration
DB_URL = os.environ.get("DB_URL", "postgresql://postgres:postgres@localhost:5433/gmail_twoyrs")
SCRIPTS_DIR = Path(__file__).parent


def run_script(name: str, args: list = None, env: dict = None, description: str = None) -> bool:
    """Run a pipeline script."""
    script_path = SCRIPTS_DIR / name
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    script_env = os.environ.copy()
    if env:
        script_env.update(env)

    desc = description or name
    print(f"\n{'='*60}")
    print(f"Stage: {desc}")
    print(f"Script: {name}")
    print(f"{'='*60}")

    start = datetime.now()
    result = subprocess.run(cmd, env=script_env)
    elapsed = (datetime.now() - start).total_seconds()

    if result.returncode != 0:
        print(f"FAILED: {name} (exit code {result.returncode})")
        return False

    print(f"SUCCESS: {desc} ({elapsed:.1f}s)")
    return True


def check_postgres() -> bool:
    """Check PostgreSQL connection."""
    try:
        conn = psycopg2.connect(DB_URL)
        conn.close()
        return True
    except Exception as e:
        print(f"PostgreSQL error: {e}")
        return False


def check_api_keys() -> dict:
    """Check which API keys are available."""
    return {
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }


def get_status() -> dict:
    """Get current pipeline status."""
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()

        status = {}

        # Emails
        cur.execute("SELECT COUNT(*) FROM emails WHERE is_sent = FALSE")
        status["emails"] = cur.fetchone()[0]

        # Features
        cur.execute("SELECT COUNT(*) FROM email_features")
        status["features"] = cur.fetchone()[0]

        # Embeddings
        try:
            cur.execute("SELECT COUNT(*) FROM email_embeddings")
            status["embeddings"] = cur.fetchone()[0]
        except:
            status["embeddings"] = 0

        # AI Classification
        try:
            cur.execute("SELECT COUNT(*) FROM email_ai_classification")
            status["ai_classification"] = cur.fetchone()[0]
        except:
            status["ai_classification"] = 0

        # LLM Classification
        try:
            cur.execute("SELECT COUNT(*) FROM email_llm_classification")
            status["llm_classification"] = cur.fetchone()[0]
        except:
            status["llm_classification"] = 0

        # Needs LLM
        try:
            cur.execute("""
                SELECT COUNT(*) FROM email_ai_classification
                WHERE predicted_handleability = 'needs_llm'
            """)
            status["needs_llm"] = cur.fetchone()[0]
        except:
            status["needs_llm"] = 0

        conn.close()
        return status

    except Exception as e:
        return {"error": str(e)}


def print_status():
    """Print current pipeline status."""
    status = get_status()

    if "error" in status:
        print(f"Error getting status: {status['error']}")
        return

    print("\n" + "=" * 60)
    print("PIPELINE STATUS")
    print("=" * 60)
    print(f"Emails imported:        {status['emails']:,}")
    print(f"ML features computed:   {status['features']:,}")
    print(f"Embeddings generated:   {status['embeddings']:,}")
    print(f"Rule-based classified:  {status['ai_classification']:,}")
    print(f"LLM classified:         {status['llm_classification']:,}")
    print(f"Needs LLM (remaining):  {status['needs_llm'] - status['llm_classification']:,}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="One-shot data onboarding pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "mbox_path",
        type=Path,
        nargs="?",
        help="Path to Gmail MBOX file"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current pipeline status"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation"
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM classification (saves cost)"
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        choices=range(1, 8),
        help="Start from stage N (1-7)"
    )
    parser.add_argument(
        "--llm-model",
        choices=["gpt5", "haiku", "sonnet"],
        default="gpt5",
        help="LLM model for classification (default: gpt5)"
    )

    args = parser.parse_args()

    # Status mode
    if args.status:
        print_status()
        return

    # Require mbox path for processing
    if not args.mbox_path:
        parser.print_help()
        print("\nERROR: MBOX path required (or use --status)")
        sys.exit(1)

    if not args.mbox_path.exists():
        print(f"ERROR: MBOX file not found: {args.mbox_path}")
        sys.exit(1)

    # Check prerequisites
    print("\n=== Checking Prerequisites ===")

    if not check_postgres():
        print("\nERROR: PostgreSQL not accessible")
        print("Start with: docker compose up -d postgres")
        sys.exit(1)
    print("PostgreSQL: OK")

    keys = check_api_keys()
    if keys["openai"]:
        print("OpenAI API Key: OK")
    else:
        print("OpenAI API Key: Missing (embeddings will fail)")
        if not args.skip_embeddings:
            print("  Use --skip-embeddings to continue without embeddings")

    if keys["openai"] or keys["anthropic"]:
        print(f"LLM API Key: OK")
    else:
        print("LLM API Key: Missing (LLM classification will fail)")
        args.skip_llm = True

    # Setup environment
    data_dir = Path("./data/onboarding")
    data_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "MBOX_PATH": str(args.mbox_path.absolute()),
        "PARSED_JSONL": str((data_dir / "parsed_emails.jsonl").absolute()),
        "DB_URL": DB_URL,
    }

    # Print configuration
    print(f"\n=== Configuration ===")
    print(f"MBOX Path: {args.mbox_path}")
    print(f"Data Dir: {data_dir}")
    print(f"Workers: {args.workers}")
    print(f"Skip Embeddings: {args.skip_embeddings}")
    print(f"Skip LLM: {args.skip_llm}")
    print(f"Start From: Stage {args.start_from}")

    start_time = datetime.now()
    print(f"\n=== Starting Pipeline at {start_time.strftime('%Y-%m-%d %H:%M:%S')} ===")

    # Define stages
    stages = [
        (1, "parse_mbox.py", [], "Parse MBOX to JSONL"),
        (2, "import_to_postgres.py", [], "Import to PostgreSQL"),
        (3, "populate_threads.py", [], "Build thread relationships"),
        (4, "compute_basic_features.py", [], "Compute ML features (Phase 2)"),
        (5, "compute_embeddings.py", [f"--workers", str(args.workers)], "Generate embeddings (Phase 3)"),
        (6, "classify_ai_handleability.py", [], "Rule-based classification (Phase 0)"),
        (7, "run_llm_classification.py", ["--all", str(args.workers), args.llm_model], "LLM classification (Phase 4)"),
    ]

    failed = False
    for stage_num, script, script_args, description in stages:
        # Skip if before start point
        if stage_num < args.start_from:
            print(f"\nSkipping stage {stage_num}: {description}")
            continue

        # Skip embeddings if requested
        if args.skip_embeddings and script == "compute_embeddings.py":
            print(f"\nSkipping stage {stage_num}: {description} (--skip-embeddings)")
            continue

        # Skip LLM if requested
        if args.skip_llm and script == "run_llm_classification.py":
            print(f"\nSkipping stage {stage_num}: {description} (--skip-llm)")
            continue

        success = run_script(script, script_args, env, f"[{stage_num}/7] {description}")

        if not success:
            failed = True
            print(f"\nPipeline failed at stage {stage_num}")
            print(f"Resume with: python scripts/onboard_data.py {args.mbox_path} --start-from {stage_num}")
            break

    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'='*60}")
    if failed:
        print(f"PIPELINE FAILED after {duration}")
    else:
        print(f"PIPELINE COMPLETED in {duration}")
        print_status()
        print(f"\nNext steps:")
        print(f"  1. Explore data: uv run streamlit run apps/labeling_ui.py")
        print(f"  2. Create checkpoint: uv run python scripts/checkpoint.py create --name onboarding_complete")
    print(f"{'='*60}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
