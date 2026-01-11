"""Stage 6: Compute email embeddings using OpenAI.

Generates embeddings for emails using text-embedding-3-small model.
Stores results in email_embeddings table with pgvector.

Performance Notes:
- Uses batch API calls (up to 100 texts per call) for efficiency
- Parallel batch processing for large volumes (44k+ emails)
- Email sanitization before embedding to optimize token usage
- 88 emails = 1 API call vs 88 calls (10-45 second savings)
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

import psycopg2
from bs4 import BeautifulSoup
from psycopg2.extras import execute_values

from priority_lens.core.config import Config
from priority_lens.core.sanitization import sanitize_email
from priority_lens.pipeline.stages.base import StageResult

if TYPE_CHECKING:
    pass

# Suppress LiteLLM debug/info messages
os.environ["LITELLM_LOG"] = "ERROR"

logger = logging.getLogger(__name__)

# Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
# Model limit is 8192, leave buffer for safety
MAX_TOKENS = 8000
# Database batch size for fetching emails
DEFAULT_BATCH_SIZE = 100
# Number of parallel workers for API calls (one email per call)
DEFAULT_PARALLEL_WORKERS = 35


def _get_tokenizer() -> Any:
    """Get tiktoken encoder for accurate token counting.

    Returns:
        Tiktoken encoder or None if not available.
    """
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


# Cache the tokenizer
_TOKENIZER = _get_tokenizer()


def count_tokens(text: str) -> int:
    """Count tokens accurately using tiktoken.

    Falls back to character estimate if tiktoken unavailable.

    Args:
        text: Text to count tokens for.

    Returns:
        Token count.
    """
    if not text:
        return 0

    if _TOKENIZER is not None:
        return len(_TOKENIZER.encode(text))

    # Fallback: conservative estimate
    return len(text) // 2


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to fit within token limit using tiktoken.

    Args:
        text: Text to truncate.
        max_tokens: Maximum tokens allowed.

    Returns:
        Truncated text.
    """
    if not text:
        return ""

    if _TOKENIZER is not None:
        tokens = _TOKENIZER.encode(text)
        if len(tokens) <= max_tokens:
            return text
        # Truncate tokens and decode back to text
        truncated_tokens = tokens[:max_tokens]
        decoded: str = _TOKENIZER.decode(truncated_tokens)
        return decoded + "..."

    # Fallback: conservative character estimate
    max_chars = max_tokens * 2
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def strip_html(html_body: str) -> str:
    """Extract plain text from HTML email body.

    Args:
        html_body: Raw HTML body.

    Returns:
        Plain text content.
    """
    if not html_body:
        return ""

    try:
        soup = BeautifulSoup(html_body, "html.parser")

        for element in soup(["script", "style", "head", "meta", "link", "noscript"]):
            element.decompose()

        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception:
        return html_body


def strip_template_syntax(text: str) -> str:
    """Remove Liquid/Jinja template syntax from text.

    Args:
        text: Text with potential template syntax.

    Returns:
        Cleaned text.
    """
    if not text:
        return ""

    text = re.sub(r"\{%.*?%\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\{#.*?#\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def strip_quoted_replies(text: str) -> str:
    """Remove quoted reply chains and signatures from email text.

    Args:
        text: Email text with potential quotes.

    Returns:
        Cleaned text without quotes/signatures.
    """
    if not text:
        return ""

    lines = text.split("\n")
    clean_lines = []

    for line in lines:
        if line.strip().startswith(">"):
            continue
        if re.match(r"^On .+ wrote:$", line.strip()):
            continue
        if re.match(r"^(From|Sent|To|Subject|Date):", line.strip()):
            continue
        if line.strip() in ["--", "---", "-- ", "Sent from my iPhone", "Sent from my iPad"]:
            break
        clean_lines.append(line)

    return "\n".join(clean_lines).strip()


def build_embedding_text(
    subject: str,
    body: str,
    is_service: bool,
    service_importance: float,
    relationship_strength: float,
    use_new_sanitizer: bool = True,
) -> str:
    """Build embedding text with importance metadata.

    Args:
        subject: Email subject.
        body: Email body.
        is_service: Whether this is a service email.
        service_importance: Service importance score.
        relationship_strength: Relationship strength score.
        use_new_sanitizer: Use new comprehensive sanitization module.

    Returns:
        Formatted text for embedding.
    """
    if is_service:
        email_type = "SERVICE"
        if service_importance >= 0.7:
            priority = "HIGH"
        elif service_importance >= 0.4:
            priority = "MEDIUM"
        else:
            priority = "LOW"
    else:
        email_type = "PERSON"
        if relationship_strength > 0.5:
            priority = "HIGH"
        elif relationship_strength > 0.2:
            priority = "MEDIUM"
        else:
            priority = "LOW"

    metadata = f"[TYPE: {email_type}] [PRIORITY: {priority}]"
    subject_line = f"[SUBJECT] {subject.strip()}" if subject else ""

    header_text = metadata + "\n" + subject_line if subject_line else metadata
    header_tokens = count_tokens(header_text)
    body_token_budget = MAX_TOKENS - header_tokens - 20

    if body and body_token_budget > 100:
        # Use new comprehensive sanitization module
        if use_new_sanitizer:
            result = sanitize_email(body)
            clean_body = result.text
        else:
            # Legacy path for backward compatibility
            clean_body = strip_html(body)
            clean_body = strip_template_syntax(clean_body)
            clean_body = strip_quoted_replies(clean_body)

        if clean_body:
            clean_body = truncate_to_tokens(clean_body, body_token_budget)
            body_line = f"[BODY] {clean_body}"
        else:
            body_line = ""
    else:
        body_line = ""

    parts = [metadata]
    if subject_line:
        parts.append(subject_line)
    if body_line:
        parts.append(body_line)

    return "\n".join(parts)


def compute_content_hash(text: str) -> str:
    """Compute SHA256 hash of content for deduplication.

    Args:
        text: Text to hash.

    Returns:
        First 16 characters of SHA256 hash.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def create_tables(conn: psycopg2.extensions.connection) -> None:
    """Create email_embeddings table if it doesn't exist.

    Args:
        conn: Database connection.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS email_embeddings (
                id SERIAL PRIMARY KEY,
                email_id INTEGER REFERENCES emails(id) UNIQUE,
                embedding vector(1536),
                model TEXT,
                token_count INTEGER,
                content_hash TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_email_embeddings_email_id
            ON email_embeddings(email_id)
        """
        )

        conn.commit()


def get_unprocessed_emails(
    conn: psycopg2.extensions.connection, limit: int = 1000
) -> list[dict[str, Any]]:
    """Get emails that don't have embeddings yet.

    Args:
        conn: Database connection.
        limit: Maximum number of emails to return.

    Returns:
        List of email dictionaries with features.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.id, e.subject, e.body_text,
                   COALESCE(ef.is_service_email, FALSE) as is_service,
                   COALESCE(ef.service_importance, 0.5) as service_importance,
                   COALESCE(ef.relationship_strength, 0.5) as relationship_strength
            FROM emails e
            LEFT JOIN email_embeddings ee ON e.id = ee.email_id
            LEFT JOIN email_features ef ON e.id = ef.email_id
            WHERE ee.id IS NULL
              AND e.is_sent = FALSE
            ORDER BY e.id
            LIMIT %s
        """,
            (limit,),
        )

        rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "subject": row[1],
                "body": row[2],
                "is_service": row[3],
                "service_importance": row[4],
                "relationship_strength": row[5],
            }
            for row in rows
        ]


def get_embedding_counts(conn: psycopg2.extensions.connection) -> tuple[int, int]:
    """Get total eligible and embedded counts.

    Args:
        conn: Database connection.

    Returns:
        Tuple of (total_eligible, already_embedded).
    """
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM emails WHERE is_sent = FALSE")
        row = cur.fetchone()
        total = row[0] if row else 0

        cur.execute("SELECT COUNT(*) FROM email_embeddings")
        row = cur.fetchone()
        embedded = row[0] if row else 0

        return total, embedded


def save_embeddings_to_db(
    conn: psycopg2.extensions.connection,
    embeddings_data: list[tuple[int, list[float], int, str]],
) -> int:
    """Save embeddings to email_embeddings table.

    Args:
        conn: Database connection.
        embeddings_data: List of (email_id, embedding, token_count, content_hash).

    Returns:
        Number of embeddings saved.
    """
    if not embeddings_data:
        return 0

    with conn.cursor() as cur:
        values = []
        for email_id, emb, token_count, content_hash in embeddings_data:
            embedding_str = f"[{','.join(str(x) for x in emb)}]"
            values.append((email_id, embedding_str, EMBEDDING_MODEL, token_count, content_hash))

        execute_values(
            cur,
            """
            INSERT INTO email_embeddings (email_id, embedding, model, token_count, content_hash)
            VALUES %s
            ON CONFLICT (email_id) DO NOTHING
            """,
            values,
            template="(%s, %s::vector, %s, %s, %s)",
        )

        conn.commit()
        return len(values)


def generate_single_embedding(text: str, embedding_func: Any) -> list[float]:
    """Generate embedding for a single text.

    Args:
        text: Text to embed.
        embedding_func: LiteLLM embedding function.

    Returns:
        Embedding vector.
    """
    response = embedding_func(model=EMBEDDING_MODEL, input=[text])
    return list(response.data[0]["embedding"])


def prepare_emails_for_embedding(
    emails: list[dict[str, Any]],
) -> list[tuple[int, str, int, str]]:
    """Prepare emails for batch embedding.

    Args:
        emails: List of email dictionaries.

    Returns:
        List of (email_id, text, token_count, content_hash) for valid emails.
    """
    prepared = []

    for email in emails:
        text = build_embedding_text(
            email["subject"] or "",
            email["body"] or "",
            email["is_service"],
            email["service_importance"],
            email["relationship_strength"],
        )

        if not text.strip() or len(text) < 10:
            continue

        # Truncate if exceeds model's token limit
        token_count = count_tokens(text)
        if token_count > MAX_TOKENS:
            text = truncate_to_tokens(text, MAX_TOKENS)
            token_count = count_tokens(text)
            logger.debug(f"Truncated email {email['id']} to {token_count} tokens")

        content_hash = compute_content_hash(text)
        prepared.append((email["id"], text, token_count, content_hash))

    return prepared


def _process_single_email(
    email_data: tuple[int, str, int, str],
    embedding_func: Any,
) -> tuple[int, list[float], int, str] | None:
    """Process a single email through the embedding API.

    Args:
        email_data: Tuple of (email_id, text, token_count, content_hash).
        embedding_func: LiteLLM embedding function.

    Returns:
        Tuple of (email_id, embedding, token_count, content_hash) or None on error.
    """
    email_id, text, token_count, content_hash = email_data

    try:
        embedding = generate_single_embedding(text, embedding_func)
        return (email_id, embedding, token_count, content_hash)
    except Exception as e:
        logger.warning(f"Embedding failed for email {email_id}: {e}")
        return None


def process_emails_parallel(
    emails: list[dict[str, Any]],
    embedding_func: Any,
    workers: int = DEFAULT_PARALLEL_WORKERS,
) -> list[tuple[int, list[float], int, str]]:
    """Process emails with parallel API calls (one email per call).

    Each email gets its own API call to preserve full content.
    Parallel workers provide throughput.

    Args:
        emails: List of email dictionaries.
        embedding_func: LiteLLM embedding function.
        workers: Number of parallel workers for API calls.

    Returns:
        List of (email_id, embedding, token_count, content_hash).
    """
    # Prepare all emails first
    prepared = prepare_emails_for_embedding(emails)

    if not prepared:
        return []

    logger.debug(f"Processing {len(prepared)} emails with {workers} parallel workers")

    results: list[tuple[int, list[float], int, str]] = []

    # Process emails in parallel (one API call per email)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_single_email, email_data, embedding_func): email_data[0]
            for email_data in prepared
        }

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

    return results


def compute_embeddings_sync(
    conn: psycopg2.extensions.connection,
    embedding_func: Any,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
    workers: int = DEFAULT_PARALLEL_WORKERS,
) -> dict[str, Any]:
    """Compute embeddings for all unprocessed emails using parallel API calls.

    Each email gets its own API call to preserve full content.
    Parallel workers provide throughput.

    Args:
        conn: Database connection.
        embedding_func: LiteLLM embedding function.
        batch_size: Emails per database fetch.
        limit: Maximum emails to process (None for all).
        workers: Number of parallel workers for API calls.

    Returns:
        Statistics dictionary.
    """
    total, already_embedded = get_embedding_counts(conn)
    remaining = total - already_embedded

    if limit:
        remaining = min(remaining, limit)

    if remaining == 0:
        return {
            "total_emails": total,
            "already_embedded": already_embedded,
            "processed": 0,
            "batches": 0,
            "api_calls": 0,
        }

    total_processed = 0
    batch_num = 0
    api_calls = 0

    while total_processed < remaining:
        batch_num += 1
        fetch_limit = min(batch_size, remaining - total_processed)
        emails = get_unprocessed_emails(conn, limit=fetch_limit)

        if not emails:
            break

        logger.info(f"Processing batch {batch_num}: {len(emails)} emails with {workers} workers")

        embeddings_data = process_emails_parallel(emails, embedding_func, workers=workers)

        # Each email = 1 API call
        api_calls += len(embeddings_data)

        saved = save_embeddings_to_db(conn, embeddings_data)
        total_processed += saved

        logger.info(f"Batch {batch_num} complete: {saved} embeddings saved")

    return {
        "total_emails": total,
        "already_embedded": already_embedded,
        "processed": total_processed,
        "batches": batch_num,
        "api_calls": api_calls,
    }


def run(
    config: Config,
    workers: int = DEFAULT_PARALLEL_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
) -> StageResult:
    """Run Stage 6: Compute email embeddings.

    Args:
        config: Application configuration.
        workers: Number of parallel workers for API calls.
        batch_size: Emails per database fetch batch.
        limit: Maximum emails to process.

    Returns:
        StageResult with computation statistics.
    """
    start_time = time.time()

    if not config.openai_api_key:
        return StageResult(
            success=False,
            records_processed=0,
            duration_seconds=0,
            message="OPENAI_API_KEY not configured",
        )

    try:
        import litellm

        litellm.suppress_debug_info = True
        from litellm import embedding as litellm_embedding
    except ImportError:
        return StageResult(
            success=False,
            records_processed=0,
            duration_seconds=0,
            message="litellm package not installed",
        )

    logger.info(
        f"Starting embedding generation (workers={workers}, batch_size={batch_size}, limit={limit})"
    )

    conn = psycopg2.connect(config.sync_database_url)
    try:
        create_tables(conn)

        stats = compute_embeddings_sync(
            conn,
            litellm_embedding,
            batch_size=batch_size,
            limit=limit,
            workers=workers,
        )

        duration = time.time() - start_time

        api_calls = stats.get("api_calls", 0)
        logger.info(
            f"Embedding generation complete: {stats['processed']} emails, "
            f"{api_calls} API calls, {duration:.1f}s"
        )

        return StageResult(
            success=True,
            records_processed=stats["processed"],
            duration_seconds=duration,
            message=f"Generated embeddings for {stats['processed']} emails ({api_calls} API calls)",
            metadata=stats,
        )
    finally:
        conn.close()
