"""Every setting the pipeline reads, in one module.

Both halves — ingestion and retrieval — must agree on these. If ingestion embeds
with one model and retrieval queries with another, the index returns well-formed
results with plausible scores that happen to be meaningless, and nothing errors.
The manifest in `index.py` is the guard against exactly that.
"""

import os
import re
from pathlib import Path

import tiktoken


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

# Google Gemini embedding model
EMBED_MODEL = "gemini-embedding-001"

# Google Gemini vision model
VISION_MODEL = "gemini-3.6-flash"

# Cross-encoder used to reorder candidates.
# Hosted by Pinecone, so no second provider account and no second API key.
RERANK_MODEL = os.getenv(
    "RERANK_MODEL",
    "bge-reranker-v2-m3"
)

# Google Gemini generation model
LLM_MODEL = "gemini-3.6-flash"


# ─────────────────────────────────────────────────────────────────────────────
# Index
# ─────────────────────────────────────────────────────────────────────────────

INDEX_NAME = "rag-docs"

NAMESPACE = ""

# Fixed at index creation and not changeable afterwards without rebuilding.
METRIC = "dotproduct"

# Stamped on every chunk and applied as a filter on every query.
ACCESS_GROUPS = ["public"]


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

# Retrieval-quality target.
# Smaller chunks match more precisely; larger ones carry more of the answer
# in one vector.
CHUNK_TOKEN_TARGET = int(
    os.getenv(
        "CHUNK_TOKEN_TARGET",
        "1024"
    )
)

# Minimum number of tokens a text entry should contain.
#
# _apply_floor() uses this value to merge small text entries with the
# following text entry.
#
# Set to 0 to completely disable minimum-chunk merging.
MIN_CHUNK_TOKENS = int(
    os.getenv(
        "MIN_CHUNK_TOKENS",
        "100"
    )
)

# Gemini embedding model sequence limit.
# Keep the chunk size safely below the provider limit.
SEQUENCE_LIMITS = {
    "gemini-embedding-001": 2048,
}


# ─────────────────────────────────────────────────────────────────────────────
# Documented API limits
# ─────────────────────────────────────────────────────────────────────────────

PINECONE_METADATA_BYTES = 40 * 1024

PINECONE_REQUEST_BYTES = 2 * 1024 * 1024

# Gemini embedding batch size.
GEMINI_EMBED_MAX_INPUTS = 100


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

CACHE_DIR = Path(".cache")

EMBED_CACHE = CACHE_DIR / "embeddings"

MANIFEST_PATH = CACHE_DIR / "manifest.json"

CLOUD, REGION = "aws", "us-east-1"

CACHE_DIR.mkdir(
    exist_ok=True
)

EMBED_CACHE.mkdir(
    exist_ok=True
)


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

# IMPORTANT:
# Gemini does not provide a Docling Groq-style tokenizer module.
# tiktoken is kept here for local chunk/token counting.

ENCODING = tiktoken.get_encoding(
    "cl100k_base"
)

CHUNK_TOKENS = min(
    CHUNK_TOKEN_TARGET,
    SEQUENCE_LIMITS.get(
        EMBED_MODEL,
        CHUNK_TOKEN_TARGET
    )
)


# ─────────────────────────────────────────────────────────────────────────────
# Figure prompt
# ─────────────────────────────────────────────────────────────────────────────

FIGURE_PROMPT = (
    "Describe this figure for a search index. State the chart type, what each axis "
    "measures and its units, the time period, the series or categories shown, and "
    "the main finding including the specific numbers and labels visible in the "
    "image. Report only what is shown; do not infer or interpret."
)

FIGURE_AREA_THRESHOLD = float(
    os.getenv(
        "FIGURE_AREA_THRESHOLD",
        "0.01"
    )
)

FIGURE_RENDER_SCALE = float(
    os.getenv(
        "FIGURE_RENDER_SCALE",
        "2.0"
    )
)


# ─────────────────────────────────────────────────────────────────────────────
# Processing options
# ─────────────────────────────────────────────────────────────────────────────

TABLE_MODE_ACCURATE = os.getenv(
    "TABLE_MODE_ACCURATE",
    "1"
) == "1"

DO_CHART_EXTRACTION = os.getenv(
    "DO_CHART_EXTRACTION",
    "0"
) == "1"

DO_CLASSIFICATION = os.getenv(
    "DO_CLASSIFICATION",
    "1"
) == "1"

DO_FORMULA = os.getenv(
    "DO_FORMULA",
    "1"
) == "1"

DO_CODE = os.getenv(
    "DO_CODE",
    "1"
) == "1"


# ─────────────────────────────────────────────────────────────────────────────
# Table summary
# ─────────────────────────────────────────────────────────────────────────────

TABLE_PROMPT = (
    "Summarise this table so it can be found by a natural-language search.\n\n"
    "Cover, in prose:\n"
    "1. What the table reports, and what one row represents.\n"
    "2. What each column measures, with its units.\n"
    "3. Patterns that hold across rows or columns but appear in no single cell: "
    "intervals and cadence stated in words such as 'every 4 weeks' or 'at each "
    "quarter end', totals, ranges, counts, and percentage or absolute change from "
    "first to last.\n"
    "4. Specific values worth naming: the largest and smallest, and any row or "
    "column that breaks the pattern the others follow.\n"
    "5. The words someone would type when looking for this table.\n\n"
    "Every statement must be verifiable by reading the cells. Computing an "
    "interval, a total, a range or a change is reading the table and is wanted. "
    "Claiming that a result is significant, expected, encouraging, or reflects some "
    "cause is not supported by the table; do not write it."
)

TABLE_MIN_CELLS = 4

# Google Gemini model used for table summaries
TABLE_MODEL = os.getenv(
    "TABLE_MODEL",
    "gemini-3.6-flash"
)


# ─────────────────────────────────────────────────────────────────────────────
# Reports
# ─────────────────────────────────────────────────────────────────────────────

REPORT_DIR = Path(
    os.getenv(
        "REPORT_DIR",
        "reports"
    )
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    """Filesystem- and identifier-safe form of arbitrary text.

    Used for document ids, cache filenames and index names, so it has to be stable:
    the same input must always produce the same slug, or cached parses stop being
    found and documents get re-ingested under a second id.
    """

    return re.sub(
        r"[^a-z0-9]+",
        "-",
        text.lower()
    ).strip("-")[:48]