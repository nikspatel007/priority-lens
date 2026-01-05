#!/usr/bin/env python3
"""Async parallel email feature extraction using Ollama.

Uses asyncio with 4 concurrent workers for parallel LLM requests.
Target: ~1s/email effective throughput.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import aiohttp
import psycopg2


# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "deepseek-r1:8b"

# PostgreSQL configuration
PG_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "rl_emails",
    "user": "postgres",
    "password": "postgres",
}


@dataclass
class ExtractedFeatures:
    """All features extracted from an email."""
    email_id: str
    is_service_email: bool
    service_type: Optional[str]
    tasks: list[dict]
    overall_urgency: float
    requires_response: bool
    topic_category: str
    summary: str
    extraction_time_ms: int
    parse_success: bool


# One-shot extraction prompt
SYSTEM_PROMPT = """You are an email analysis expert. Extract features from the email and respond with ONLY valid JSON.

Output this exact JSON structure (no markdown, no explanation):
{
  "is_service_email": true or false,
  "service_type": "newsletter" or "notification" or "billing" or "social" or "marketing" or "system" or "calendar" or null,
  "tasks": [{"description": "...", "deadline": "..." or null, "assignee": "user" or "other" or null, "task_type": "review" or "send" or "schedule" or "decision" or "create" or "followup", "urgency": 0.5}],
  "overall_urgency": 0.0 to 1.0,
  "requires_response": true or false,
  "topic_category": "project" or "meeting" or "request" or "fyi" or "social" or "admin",
  "summary": "one line summary"
}

Rules: is_service_email=true for automated emails. tasks=[] if no action items. Output ONLY JSON."""


def get_emails_from_db(limit: int = 20) -> list[dict]:
    """Query sample emails from PostgreSQL."""
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, message_id, subject, body_text, from_email, date_parsed
                FROM emails
                WHERE body_text IS NOT NULL AND body_text != ''
                  AND subject IS NOT NULL AND length(body_text) > 50
                ORDER BY date_parsed DESC
                LIMIT %s
            """, (limit,))

            rows = cur.fetchall()
            return [{
                "id": row[0],
                "message_id": row[1],
                "subject": row[2] or "",
                "body_text": row[3] or "",
                "from_email": row[4] or "",
                "date_parsed": str(row[5]) if row[5] else "",
            } for row in rows]
    finally:
        conn.close()


def parse_llm_response(text: str) -> Optional[dict]:
    """Parse LLM response to extract JSON."""
    # Clean up response
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in response
    # Look for { ... } pattern
    start = text.find('{')
    if start >= 0:
        # Find matching closing brace
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        break

    # Try removing markdown code blocks
    if "```" in text:
        lines = text.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```"):
                in_block = not in_block
                continue
            if in_block or (not in_block and line.strip().startswith("{")):
                json_lines.append(line)
        if json_lines:
            try:
                return json.loads("\n".join(json_lines))
            except json.JSONDecodeError:
                pass

    return None


async def extract_features_async(
    session: aiohttp.ClientSession,
    email: dict,
    model: str,
    semaphore: asyncio.Semaphore,
) -> ExtractedFeatures:
    """Extract features from a single email using Ollama."""

    email_id = email["message_id"]
    subject = email["subject"]
    body = email["body_text"][:2000]
    sender = email["from_email"]

    prompt = f"{SYSTEM_PROMPT}\n\nEmail:\nSubject: {subject}\nFrom: {sender}\n\n{body}"

    start_time = time.time()

    async with semaphore:
        try:
            async with session.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 512,
                    }
                },
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                elapsed_ms = int((time.time() - start_time) * 1000)

                if response.status != 200:
                    return ExtractedFeatures(
                        email_id=email_id,
                        is_service_email=False,
                        service_type=None,
                        tasks=[],
                        overall_urgency=0.0,
                        requires_response=False,
                        topic_category="fyi",
                        summary=f"HTTP {response.status}",
                        extraction_time_ms=elapsed_ms,
                        parse_success=False,
                    )

                result = await response.json()
                response_text = result.get("response", "")

                data = parse_llm_response(response_text)

                if data:
                    return ExtractedFeatures(
                        email_id=email_id,
                        is_service_email=data.get("is_service_email", False),
                        service_type=data.get("service_type"),
                        tasks=data.get("tasks", []),
                        overall_urgency=float(data.get("overall_urgency", 0.0)),
                        requires_response=data.get("requires_response", False),
                        topic_category=data.get("topic_category", "fyi"),
                        summary=str(data.get("summary", ""))[:100],
                        extraction_time_ms=elapsed_ms,
                        parse_success=True,
                    )
                else:
                    return ExtractedFeatures(
                        email_id=email_id,
                        is_service_email=False,
                        service_type=None,
                        tasks=[],
                        overall_urgency=0.0,
                        requires_response=False,
                        topic_category="fyi",
                        summary="JSON parse failed",
                        extraction_time_ms=elapsed_ms,
                        parse_success=False,
                    )

        except asyncio.TimeoutError:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return ExtractedFeatures(
                email_id=email_id,
                is_service_email=False,
                service_type=None,
                tasks=[],
                overall_urgency=0.0,
                requires_response=False,
                topic_category="fyi",
                summary="Timeout",
                extraction_time_ms=elapsed_ms,
                parse_success=False,
            )
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return ExtractedFeatures(
                email_id=email_id,
                is_service_email=False,
                service_type=None,
                tasks=[],
                overall_urgency=0.0,
                requires_response=False,
                topic_category="fyi",
                summary=f"Error: {str(e)[:50]}",
                extraction_time_ms=elapsed_ms,
                parse_success=False,
            )


async def process_batch(
    emails: list[dict],
    model: str,
    workers: int,
) -> tuple[list[ExtractedFeatures], float]:
    """Process all emails with parallel workers."""

    semaphore = asyncio.Semaphore(workers)

    async with aiohttp.ClientSession() as session:
        start_time = time.time()

        tasks = [
            extract_features_async(session, email, model, semaphore)
            for email in emails
        ]

        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

    return list(results), total_time


def save_to_database(features_list: list[ExtractedFeatures], model: str) -> int:
    """Save extracted features to database."""
    conn = psycopg2.connect(**PG_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS email_llm_features (
                    id SERIAL PRIMARY KEY,
                    email_id TEXT NOT NULL,
                    is_service_email BOOLEAN,
                    service_type TEXT,
                    tasks JSONB,
                    overall_urgency FLOAT,
                    requires_response BOOLEAN,
                    topic_category TEXT,
                    summary TEXT,
                    extraction_time_ms INTEGER,
                    parse_success BOOLEAN,
                    model TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(email_id, model)
                )
            """)

            inserted = 0
            for f in features_list:
                try:
                    cur.execute("""
                        INSERT INTO email_llm_features
                        (email_id, is_service_email, service_type, tasks,
                         overall_urgency, requires_response, topic_category,
                         summary, extraction_time_ms, parse_success, model)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (email_id, model) DO UPDATE SET
                            is_service_email = EXCLUDED.is_service_email,
                            service_type = EXCLUDED.service_type,
                            tasks = EXCLUDED.tasks,
                            overall_urgency = EXCLUDED.overall_urgency,
                            requires_response = EXCLUDED.requires_response,
                            topic_category = EXCLUDED.topic_category,
                            summary = EXCLUDED.summary,
                            extraction_time_ms = EXCLUDED.extraction_time_ms,
                            parse_success = EXCLUDED.parse_success,
                            created_at = NOW()
                    """, (
                        f.email_id, f.is_service_email, f.service_type,
                        json.dumps(f.tasks), f.overall_urgency, f.requires_response,
                        f.topic_category, f.summary, f.extraction_time_ms,
                        f.parse_success, model,
                    ))
                    inserted += 1
                except Exception as e:
                    print(f"  Warning: Insert failed: {e}", file=sys.stderr)

            conn.commit()
            return inserted
    finally:
        conn.close()


async def main_async(args):
    """Async main entry point."""

    model = args.model
    workers = args.workers
    count = args.count

    print("=" * 60)
    print("Ollama Parallel Email Feature Extraction")
    print("=" * 60)
    print(f"  Model: {model}")
    print(f"  Workers: {workers}")
    print(f"  Emails: {count}")
    print(f"  Target: ~1s/email effective throughput")
    print()

    # Check Ollama
    print("1. Checking Ollama...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434/api/tags") as resp:
                if resp.status != 200:
                    print(f"   ERROR: Ollama not responding (HTTP {resp.status})")
                    return 1
                data = await resp.json()
                models = [m["name"] for m in data.get("models", [])]
                print(f"   Available models: {models}")

                # Check if requested model is available
                model_base = model.split(":")[0]
                available = any(model_base in m for m in models)
                if not available:
                    print(f"   WARNING: {model} not found, will try anyway")
    except Exception as e:
        print(f"   ERROR: {e}")
        return 1
    print()

    # Query emails
    print("2. Querying emails...")
    try:
        emails = get_emails_from_db(limit=count)
        print(f"   Retrieved {len(emails)} emails")
    except Exception as e:
        print(f"   ERROR: {e}")
        return 1
    print()

    # Process in parallel
    print(f"3. Processing with {workers} parallel workers...")
    results, total_time = await process_batch(emails, model, workers)

    # Calculate metrics
    successes = sum(1 for r in results if r.parse_success)
    times_ms = [r.extraction_time_ms for r in results]
    avg_time_ms = sum(times_ms) / len(times_ms) if times_ms else 0
    throughput = len(emails) / total_time if total_time > 0 else 0
    effective_time_per_email = total_time / len(emails) if emails else 0

    print(f"   Completed in {total_time:.1f}s")
    print()

    # Save to database if requested
    if args.save_db:
        print("4. Saving to database...")
        inserted = save_to_database(results, model)
        print(f"   Saved {inserted} records")
        print()

    # Report
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    print("Timing:")
    print(f"  Total wall time:     {total_time:.1f}s")
    print(f"  Effective per email: {effective_time_per_email:.2f}s")
    print(f"  Throughput:          {throughput:.2f} emails/s")
    print(f"  Target:              1.0s/email")
    print(f"  Status:              {'PASS' if effective_time_per_email <= 1.0 else 'FAIL'}")
    print()
    print("Per-request timing:")
    print(f"  Avg LLM time:        {avg_time_ms:.0f}ms")
    print(f"  Min:                 {min(times_ms):.0f}ms")
    print(f"  Max:                 {max(times_ms):.0f}ms")
    print()
    print("Quality:")
    print(f"  Parse success:       {successes}/{len(results)} ({successes/len(results)*100:.0f}%)")
    print(f"  Tasks found:         {sum(len(r.tasks) for r in results)}")
    print(f"  Service emails:      {sum(1 for r in results if r.is_service_email)}/{len(results)}")
    print()

    # Comparison to baseline
    print("Comparison to phi-4 (sequential):")
    print(f"  phi-4:    5.38s/email")
    print(f"  Ollama:   {effective_time_per_email:.2f}s/email")
    print(f"  Speedup:  {5.38 / effective_time_per_email:.1f}x" if effective_time_per_email > 0 else "  Speedup:  N/A")
    print()

    # Save JSON output
    output_file = "ollama_features.json"
    output = {
        "generated_at": datetime.now().isoformat(),
        "model": model,
        "workers": workers,
        "email_count": len(emails),
        "timing": {
            "total_seconds": round(total_time, 2),
            "effective_per_email": round(effective_time_per_email, 3),
            "throughput": round(throughput, 3),
            "avg_llm_ms": round(avg_time_ms, 1),
            "target_met": effective_time_per_email <= 1.0,
        },
        "quality": {
            "parse_success_rate": round(successes / len(results) * 100, 1) if results else 0,
            "total_tasks": sum(len(r.tasks) for r in results),
        },
        "results": [asdict(r) for r in results],
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved to: {output_file}")
    print("=" * 60)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Parallel email feature extraction using Ollama"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Ollama model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=20,
        help="Number of emails to process (default: 20)"
    )
    parser.add_argument(
        "--save-db",
        action="store_true",
        help="Save results to database"
    )
    args = parser.parse_args()

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
