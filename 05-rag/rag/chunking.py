"""
Turning a parsed document into the records that will be embedded.

Pipeline:

    parsed document
          |
          v
    clean_headings()
          |
          v
    HybridChunker
          |
          v
    _to_entries()
          |
          v
    _merge_prose()
          |
          v
    _apply_floor()
          |
          v
    _to_records()
          |
          v
    _table_summaries()
          |
          v
    _finalise()
          |
          v
    records[]
          |
          v
    embedding.py
          |
          v
    Gemini embeddings
          |
          v
    Pinecone

IMPORTANT:

This file does NOT call the Gemini embedding API.

embedding.py performs embeddings.

This file only prepares the records.
"""

# ═════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═════════════════════════════════════════════════════════════════════════════

import hashlib
import json
import os
import re

from rag.tables import (
    table_markdown,
    needs_summary,
    table_looks_broken,
    summarize_table,
    summarize_table_image,
)

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from docling_core.types.doc import PictureItem

from .config import (
    ACCESS_GROUPS,
    CHUNK_TOKENS,
    EMBED_MODEL,
    ENCODING,
    PINECONE_METADATA_BYTES,
)

from .docling_io import (
    chart_data,
    picture_description,
)

from .headings import clean_headings

from .tables import (
    needs_summary,
    summarize_table,
    summarize_table_image,
    table_cells,
    table_looks_broken,
    table_markdown,
    table_ref_of,
)


# ═════════════════════════════════════════════════════════════════════════════
# TUNING
# ═════════════════════════════════════════════════════════════════════════════

# Above this a chunk is never dropped.
FURNITURE_MAX_TOKENS = 15

# Maximum target size when merging prose.
PROSE_TARGET_TOKENS = 400

# Whether prose can merge across figures/tables.
MERGE_ACROSS_EXHIBITS = True

# 0 = disabled.
MIN_CHUNK_TOKENS = int(
    os.getenv(
        "MIN_CHUNK_TOKENS",
        "0",
    )
)

# Page furniture patterns.
FURNITURE_PATTERNS = [
    (
        re.compile(
            r"^sources?\s*:",
            re.IGNORECASE,
        ),
        "an attribution line",
    ),
    (
        re.compile(
            r"^[A-Za-z]{1,2}$"
        ),
        "a single glyph",
    ),
    (
        re.compile(
            r"^page\s+\d+\b",
            re.IGNORECASE,
        ),
        "a page marker",
    ),
    (
        re.compile(
            r"^\d{1,3}$"
        ),
        "a bare page number",
    ),
]


# ═════════════════════════════════════════════════════════════════════════════
# TOKEN COUNT
# ═════════════════════════════════════════════════════════════════════════════

def _tokens(text: str) -> int:
    """
    Return the approximate token count.

    ENCODING is used only for local chunk sizing and diagnostics.

    Gemini is the actual embedding provider.
    """

    if not text:
        return 0

    return len(
        ENCODING.encode(text)
    )


# ═════════════════════════════════════════════════════════════════════════════
# CONTENT TYPE
# ═════════════════════════════════════════════════════════════════════════════

def content_type(chunk) -> str:
    """
    Determine the coarse content type.

    Possible values:

        text
        table
        figure
        formula
        code
    """

    labels = " ".join(
        str(
            getattr(
                item,
                "label",
                "",
            )
        ).lower()
        for item in chunk.meta.doc_items
    )

    for needle, label in (
        ("table", "table"),
        ("picture", "figure"),
        ("figure", "figure"),
        ("formula", "formula"),
        ("equation", "formula"),
        ("code", "code"),
    ):

        if needle in labels:
            return label

    return "text"


# ═════════════════════════════════════════════════════════════════════════════
# FURNITURE DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def is_furniture(
    body: str,
) -> str | None:
    """
    Return the reason a text chunk is page furniture.

    Returns:

        None
        "empty"
        "an attribution line"
        "a single glyph"
        "a page marker"
        "a bare page number"
    """

    stripped = body.strip()

    if not stripped:
        return "empty"

    if _tokens(stripped) > FURNITURE_MAX_TOKENS:
        return None

    for pattern, reason in FURNITURE_PATTERNS:

        if pattern.match(stripped):
            return reason

    return None


# ═════════════════════════════════════════════════════════════════════════════
# DOCUMENT DATE
# ═════════════════════════════════════════════════════════════════════════════

def document_date(
    pdf: Path,
    head: str,
) -> str:
    """
    Determine the document date.

    Looks for:

        2025
        Q3 2025

    If no year is found, falls back to PDF modification time.
    """

    year = re.search(
        r"\b(19|20)\d{2}\b",
        head,
    )

    if year:

        quarter = re.search(
            r"\bQ([1-4])\s*(19|20)\d{2}\b",
            head,
        )

        if quarter:

            return (
                f"{year.group(0)}-Q"
                f"{quarter.group(1)}"
            )

        return year.group(0)

    return datetime.fromtimestamp(
        pdf.stat().st_mtime
    ).strftime(
        "%Y-%m"
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION ID
# ═════════════════════════════════════════════════════════════════════════════

def _section_id(
    headings: list[str],
) -> str:
    """
    Generate a stable ID for the top-level section.
    """

    return hashlib.sha256(
        (
            headings[0]
            if headings
            else ""
        ).encode(
            "utf-8"
        )
    ).hexdigest()[:12]


# ═════════════════════════════════════════════════════════════════════════════
# RECORD FACTORY
# ═════════════════════════════════════════════════════════════════════════════

def _record_factory(
    pdf: Path,
    doc_id: str,
    doc_date: str,
):
    """
    Return a function that creates pipeline records.

    Record IDs are content-addressed:

        doc_id + sha256(text) + occurrence
    """

    import time

    ingested_at = int(
        time.time()
    )

    occurrences = defaultdict(int)

    def make(
        text: str,
        meta_extra: dict,
        position: int,
    ) -> dict:

        if text is None:
            text = ""

        text = str(text)

        digest = hashlib.sha256(
            text.encode(
                "utf-8"
            )
        ).hexdigest()[:16]

        occurrence = occurrences[
            digest
        ]

        occurrences[
            digest
        ] += 1

        return {
            "text": text,

            "meta": {

                "chunk_id": (
                    f"{doc_id}:"
                    f"{digest}:"
                    f"{occurrence}"
                ),

                "content_hash": digest,

                "occurrence": occurrence,

                "doc_id": doc_id,

                "source": pdf.name,

                "doc_date": doc_date,

                "ingested_at": ingested_at,

                "position": position,

                "embed_model": EMBED_MODEL,

                "access": ACCESS_GROUPS,

                "n_tokens": _tokens(
                    text
                ),

                **meta_extra,
            },
        }

    return make


# ═════════════════════════════════════════════════════════════════════════════
# CHUNKER
# ═════════════════════════════════════════════════════════════════════════════

def _make_chunker():
    """
    Create the Docling HybridChunker.

    IMPORTANT:

    Tutor code used OpenAITokenizer.

    This version uses HuggingFaceTokenizer instead,
    so no OpenAI tokenizer is required.
    """

    from docling.chunking import HybridChunker

    from docling_core.transforms.chunker.hierarchical_chunker import (
        ChunkingDocSerializer,
        ChunkingSerializerProvider,
    )

    from docling_core.transforms.serializer.markdown import (
        MarkdownTableSerializer,
    )

    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )

    from transformers import AutoTokenizer

    # ─────────────────────────────────────────────────────────────────────────
    # Markdown table provider
    # ─────────────────────────────────────────────────────────────────────────

    class MarkdownTableProvider(
        ChunkingSerializerProvider
    ):

        def get_serializer(
            self,
            doc,
            **kwargs,
        ):

            return ChunkingDocSerializer(
                doc=doc,
                table_serializer=MarkdownTableSerializer(),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Hugging Face tokenizer
    # ─────────────────────────────────────────────────────────────────────────

    hf_tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased"
    )

    tokenizer = HuggingFaceTokenizer(
        tokenizer=hf_tokenizer,
        max_tokens=CHUNK_TOKENS,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # HybridChunker
    # ─────────────────────────────────────────────────────────────────────────

    return HybridChunker(
        tokenizer=tokenizer,
        serializer_provider=MarkdownTableProvider(),
        merge_peers=False,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PASS A — CHUNKS → ENTRIES
# ═════════════════════════════════════════════════════════════════════════════

def _to_entries(
    chunks,
    chunker,
    figure_uris: dict | None = None,
) -> tuple[list[dict], list]:

    """
    Convert Docling chunks into plain entries.

    Entries contain:

        kind
        body
        contextualized
        merged
        head
        page
        page_end
        ref
        image_uri
    """

    entries = []

    dropped = []

    for chunk in chunks:

        items = chunk.meta.doc_items

        headings = list(
            chunk.meta.headings or []
        )

        pages = sorted(
            {
                prov.page_no
                for item in items
                for prov in (
                    getattr(
                        item,
                        "prov",
                        []
                    )
                    or []
                )
            }
        )

        page = (
            pages[0]
            if pages
            else -1
        )

        page_end = (
            pages[-1]
            if pages
            else -1
        )

        labels = [
            str(
                getattr(
                    item,
                    "label",
                    "",
                )
            ).lower()
            for item in items
        ]

        pictures = [
            item
            for item, label in zip(
                items,
                labels,
            )
            if "picture" in label
        ]

        beside = [
            label
            for label in labels
            if "picture" not in label
        ]

        # ─────────────────────────────────────────────────────────────────────
        # FIGURE SLOT
        # ─────────────────────────────────────────────────────────────────────

        if pictures and all(
            "caption" in label
            for label in beside
        ):

            entries.append(
                {
                    "kind": "figure_slot",

                    "refs": [
                        ref
                        for item in pictures
                        if (
                            ref := getattr(
                                item,
                                "self_ref",
                                None,
                            )
                        )
                    ],

                    "head": headings,

                    "page": page,

                    "page_end": page_end,
                }
            )

            continue

        # ─────────────────────────────────────────────────────────────────────
        # CONTENT TYPE
        # ─────────────────────────────────────────────────────────────────────

        kind = content_type(
            chunk
        )

        # ─────────────────────────────────────────────────────────────────────
        # REMOVE FURNITURE
        # ─────────────────────────────────────────────────────────────────────

        if kind == "text":

            reason = is_furniture(
                chunk.text
            )

            if reason:

                dropped.append(
                    (
                        page,
                        reason,
                        chunk.text.strip()[:60],
                    )
                )

                continue

        # ─────────────────────────────────────────────────────────────────────
        # FIGURE URI
        # ─────────────────────────────────────────────────────────────────────

        image_uri = ""

        for item in items:

            key = getattr(
                item,
                "self_ref",
                None,
            )

            if (
                key
                and key in (
                    figure_uris or {}
                )
            ):

                image_uri = (
                    figure_uris[key]
                )

                break

        # ─────────────────────────────────────────────────────────────────────
        # ENTRY
        # ─────────────────────────────────────────────────────────────────────

        entries.append(
            {
                "kind": kind,

                "body": chunk.text,

                "contextualized": (
                    chunker.contextualize(
                        chunk=chunk
                    )
                ),

                "merged": False,

                "head": headings,

                "page": page,

                "page_end": page_end,

                "ref": table_ref_of(
                    chunk
                ),

                "image_uri": image_uri,
            }
        )

    return entries, dropped


# ═════════════════════════════════════════════════════════════════════════════
# PASS B — MERGE PROSE
# ═════════════════════════════════════════════════════════════════════════════

def _merge_prose(
    entries: list[dict],
) -> tuple[list[dict], int]:

    """
    Merge adjacent text entries that share the same heading path.

    Does not merge tables, figures, formulas or code.
    """

    out = []

    merges = 0

    for entry in entries:

        target = None

        if entry["kind"] == "text":

            for candidate in reversed(
                out
            ):

                if (
                    candidate["head"]
                    != entry["head"]
                ):
                    break

                if candidate["kind"] == "text":

                    target = candidate

                    break

                if not MERGE_ACROSS_EXHIBITS:
                    break

        if (
            target is not None
            and _tokens(
                target["body"]
                + "\n"
                + entry["body"]
            )
            <= PROSE_TARGET_TOKENS
        ):

            target["body"] += (
                "\n"
                + entry["body"]
            )

            target["page_end"] = max(
                target["page_end"],
                entry["page_end"],
            )

            target["merged"] = True

            target["image_uri"] = (
                target["image_uri"]
                or entry["image_uri"]
            )

            merges += 1

        else:

            out.append(
                entry
            )

    return out, merges


# ═════════════════════════════════════════════════════════════════════════════
# PASS C — MINIMUM CHUNK FLOOR
# ═════════════════════════════════════════════════════════════════════════════

def _apply_floor(
    entries: list[dict],
) -> tuple[list[dict], int]:

    """
    Small text entries absorb the following text entry.

    This may cross heading boundaries.
    """

    if MIN_CHUNK_TOKENS <= 0:

        return entries, 0

    out = []

    merges = 0

    for entry in entries:

        previous = (
            out[-1]
            if out
            else None
        )

        if (
            previous is not None
            and previous["kind"] == "text"
            and entry["kind"] == "text"
            and _tokens(
                previous["body"]
            ) < MIN_CHUNK_TOKENS
        ):

            crossed = (
                [
                    h
                    for h in entry["head"]
                    if h not in previous["head"]
                ]
                if (
                    entry["head"]
                    != previous["head"]
                )
                else []
            )

            pieces = []

            if crossed:
                pieces.extend(
                    crossed
                )

            pieces.append(
                entry["body"]
            )

            previous["body"] += (
                "\n"
                + "\n".join(
                    pieces
                )
            )

            previous["page_end"] = max(
                previous["page_end"],
                entry["page_end"],
            )

            previous["merged"] = True

            previous["image_uri"] = (
                previous["image_uri"]
                or entry["image_uri"]
            )

            merges += 1

        else:

            out.append(
                entry
            )

    return out, merges


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE RECORDS
# ═════════════════════════════════════════════════════════════════════════════

def _figure_records(
    entry: dict,
    doc,
    items_by_ref: dict,
    figure_uris: dict,
    make,
) -> tuple[list[dict], int, int]:

    """
    Expand one figure slot into one record per picture.
    """

    records = []

    indexed = 0

    skipped = 0

    for ref in entry["refs"]:

        item = items_by_ref.get(
            ref
        )

        if item is None:
            continue

        # ─────────────────────────────────────────────────────────────────────
        # DESCRIPTION
        # ─────────────────────────────────────────────────────────────────────

        description = picture_description(
            item
        )

        if not description:

            skipped += 1

            continue

        # ─────────────────────────────────────────────────────────────────────
        # CAPTION
        # ─────────────────────────────────────────────────────────────────────

        try:

            caption = (
                item.caption_text(
                    doc
                )
                or ""
            ).strip()

        except Exception:

            caption = ""

        # ─────────────────────────────────────────────────────────────────────
        # BUILD TEXT
        # ─────────────────────────────────────────────────────────────────────

        parts = [
            *entry["head"]
        ]

        if caption:
            parts.append(
                caption
            )

        parts.append(
            description
        )

        text = "\n".join(
            parts
        )

        # ─────────────────────────────────────────────────────────────────────
        # CHART DATA
        # ─────────────────────────────────────────────────────────────────────

        series = chart_data(
            item
        )

        if series is not None:

            text += (
                "\n\nchart data: "
                + str(series)[:800]
            )

        # ─────────────────────────────────────────────────────────────────────
        # RECORD
        # ─────────────────────────────────────────────────────────────────────

        records.append(
            make(
                text,
                {
                    "page": entry["page"],

                    "page_end": entry["page_end"],

                    "headings": entry["head"],

                    "section_id": _section_id(
                        entry["head"]
                    ),

                    "content_type": "figure",

                    "table_id": "",

                    "image_uri": (
                        figure_uris or {}
                    ).get(
                        ref,
                        "",
                    ),

                    "has_caption": bool(
                        caption
                    ),

                    "has_chart_data": (
                        series is not None
                    ),
                },
                entry["position"],
            )
        )

        indexed += 1

    return (
        records,
        indexed,
        skipped,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PASS D — ENTRIES → RECORDS
# ═════════════════════════════════════════════════════════════════════════════

def _to_records(
    entries: list[dict],
    doc,
    items_by_ref: dict,
    figure_uris: dict,
    make,
) -> tuple[list[dict], dict, dict]:

    """
    Convert entries into records.

    Returns:

        records
        table_groups
        stats
    """

    records = []

    table_groups = defaultdict(
        list
    )

    indexed = 0

    skipped = 0

    refs_seen = set()

    for position, entry in enumerate(
        entries
    ):

        entry["position"] = position

        # ─────────────────────────────────────────────────────────────────────
        # FIGURE SLOT
        # ─────────────────────────────────────────────────────────────────────

        if entry["kind"] == "figure_slot":

            refs_seen.update(
                entry["refs"]
            )

            new, n_indexed, n_skipped = (
                _figure_records(
                    entry,
                    doc,
                    items_by_ref,
                    figure_uris,
                    make,
                )
            )

            records += new

            indexed += n_indexed

            skipped += n_skipped

            continue

        # ─────────────────────────────────────────────────────────────────────
        # TEXT
        # ─────────────────────────────────────────────────────────────────────

        if entry["merged"]:

            text = "\n".join(
                [
                    *entry["head"],
                    entry["body"],
                ]
            )

        else:

            text = entry[
                "contextualized"
            ]

        ref = entry[
            "ref"
        ]

        # ─────────────────────────────────────────────────────────────────────
        # TABLE ID
        # ─────────────────────────────────────────────────────────────────────

        table_id = ""

        if ref:

            table_id = hashlib.sha256(
                ref.encode(
                    "utf-8"
                )
            ).hexdigest()[:12]

        # ─────────────────────────────────────────────────────────────────────
        # RECORD
        # ─────────────────────────────────────────────────────────────────────

        record = make(
            text,
            {
                "page": entry["page"],

                "page_end": entry["page_end"],

                "headings": entry["head"],

                "section_id": _section_id(
                    entry["head"]
                ),

                "content_type": entry["kind"],

                "table_id": table_id,

                "image_uri": entry[
                    "image_uri"
                ],

                "merged": entry[
                    "merged"
                ],
            },
            position,
        )

        records.append(
            record
        )

        if ref:

            table_groups[
                ref
            ].append(
                record
            )

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURES THAT DID NOT REACH A FIGURE SLOT
    # ─────────────────────────────────────────────────────────────────────────

    merged_in = sum(
        1
        for item, _ in doc.iterate_items()
        if (
            isinstance(
                item,
                PictureItem,
            )
            and getattr(
                item,
                "self_ref",
                None,
            )
            not in refs_seen
        )
    )

    stats = {
        "figures_indexed": indexed,
        "figures_skipped": skipped,
        "figures_merged": merged_in,
    }

    return (
        records,
        table_groups,
        stats,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PASS E — TABLE SUMMARIES
# ═════════════════════════════════════════════════════════════════════════════

def _table_summaries(
    table_groups: dict,
    tables: dict,
    doc,
    items_by_ref: dict,
    make,
) -> tuple[list[dict], dict]:

    records = []

    summarised = 0

    skipped = []

    repaired = []

    for ref, fragments in table_groups.items():

        if not fragments:
            continue

        first = fragments[0]["meta"]

        markdown = tables.get(
            ref,
            "",
        )

        # ─────────────────────────────────────────────────────────────────────
        # NO MARKDOWN
        # ─────────────────────────────────────────────────────────────────────

        if not markdown:

            skipped.append(
                (
                    first["page"],
                    "could not be serialised",
                )
            )

            continue

        # ─────────────────────────────────────────────────────────────────────
        # NO SUMMARY REQUIRED
        # ─────────────────────────────────────────────────────────────────────

        if not needs_summary(
            markdown
        ):

            skipped.append(
                (
                    first["page"],
                    (
                        f"{len(table_cells(markdown))} "
                        "cells, treated as layout"
                    ),
                )
            )

            continue

        # ─────────────────────────────────────────────────────────────────────
        # CHECK TABLE STRUCTURE
        # ─────────────────────────────────────────────────────────────────────

        structure_problems = table_looks_broken(
            markdown
        )

        summary = None

        source = "markdown"

        # ─────────────────────────────────────────────────────────────────────
        # TRY IMAGE SUMMARY FOR BROKEN TABLE
        # ─────────────────────────────────────────────────────────────────────

        if structure_problems:

            item = items_by_ref.get(
                ref
            )

            if item is not None:

                try:

                    summary = summarize_table_image(
                        item,
                        doc,
                        first["headings"],
                    )

                    if summary is not None:

                        source = "image"

                except Exception as exc:

                    repaired.append(
                        (
                            first["page"],
                            structure_problems,
                            False,
                            str(exc),
                        )
                    )

            if summary is None:

                repaired.append(
                    (
                        first["page"],
                        structure_problems,
                        False,
                    )
                )

            else:

                repaired.append(
                    (
                        first["page"],
                        structure_problems,
                        True,
                    )
                )

        # ─────────────────────────────────────────────────────────────────────
        # MARKDOWN SUMMARY
        # ─────────────────────────────────────────────────────────────────────

        if summary is None:

            try:

                summary = summarize_table(
                    markdown,
                    first["headings"],
                )

                source = "markdown"

            except Exception as exc:

                skipped.append(
                    (
                        first["page"],
                        f"summary failed: {exc}",
                    )
                )

                continue

        # ─────────────────────────────────────────────────────────────────────
        # CREATE SUMMARY RECORD
        # ─────────────────────────────────────────────────────────────────────

        summary_record = make(
            summary,
            {
                "page": first["page"],

                "page_end": fragments[-1]["meta"][
                    "page_end"
                ],

                "headings": first["headings"],

                "section_id": first[
                    "section_id"
                ],

                "content_type": "table_summary",

                "table_id": first[
                    "table_id"
                ],

                "table_chars": len(
                    markdown
                ),

                "n_fragments": len(
                    fragments
                ),

                "summary_source": source,
            },

            first[
                "position"
            ],
        )

        records.append(
            summary_record
        )

        summarised += 1

    return (
        records,
        {
            "summarised": summarised,
            "skipped": skipped,
            "repaired": repaired,
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
# PASS F — FINALISE
# ═════════════════════════════════════════════════════════════════════════════

def _finalise(
    records: list[dict],
) -> tuple[list[dict], list[dict]]:

    """
    Finalize records.

    Performs:

        1. Reading-order sorting
        2. Position numbering
        3. Previous/next links
        4. Pinecone metadata limits
        5. Oversized chunk truncation
    """

    # ─────────────────────────────────────────────────────────────────────────
    # READING ORDER
    # ─────────────────────────────────────────────────────────────────────────

    records.sort(
        key=lambda r: (
            r["meta"]["position"],
            r["meta"]["content_type"]
            != "table_summary",
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    # RE-NUMBER POSITIONS
    # ─────────────────────────────────────────────────────────────────────────

    for i, record in enumerate(
        records
    ):

        record["meta"]["position"] = i

        record["meta"][
            "n_positions"
        ] = len(
            records
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PREVIOUS / NEXT
    # ─────────────────────────────────────────────────────────────────────────

    for i, record in enumerate(
        records
    ):

        if i:

            record["meta"][
                "prev_id"
            ] = records[
                i - 1
            ]["meta"][
                "chunk_id"
            ]

        if (
            i + 1
            < len(records)
        ):

            record["meta"][
                "next_id"
            ] = records[
                i + 1
            ]["meta"][
                "chunk_id"
            ]

    if not records:

        return (
            records,
            [],
        )

    # ─────────────────────────────────────────────────────────────────────────
    # METADATA BUDGET
    # ─────────────────────────────────────────────────────────────────────────

    overhead = len(
        json.dumps(
            {
                **records[
                    0
                ]["meta"],
                "text": "",
            }
        ).encode()
    )

    budget = max(
        512,
        PINECONE_METADATA_BYTES
        - overhead
        - 1024,
    )

    for record in records:

        record["meta"]["text"] = (
            record["text"][
                :budget
            ]
        )

    # ─────────────────────────────────────────────────────────────────────────
    # OVERSIZED RECORDS
    # ─────────────────────────────────────────────────────────────────────────

    over = [
        record
        for record in records
        if record["meta"][
            "n_tokens"
        ]
        > CHUNK_TOKENS
    ]

    for record in over:

        encoded = ENCODING.encode(
            record["text"]
        )

        record["text"] = (
            ENCODING.decode(
                encoded[
                    :CHUNK_TOKENS
                ]
            )
        )

        record["meta"][
            "n_tokens"
        ] = _tokens(
            record["text"]
        )

        record["meta"][
            "truncated"
        ] = True

    return (
        records,
        over,
    )


# ═════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═════════════════════════════════════════════════════════════════════════════

def _report(
    records,
    chunks,
    tables,
    table_groups,
    stats,
) -> None:

    """
    Print diagnostics for the chunking pipeline.
    """

    sizes = sorted(
        r["meta"]["n_tokens"]
        for r in records
    )

    types = Counter(
        r["meta"]["content_type"]
        for r in records
    )

    print()

    print(
        "=" * 70
    )

    print(
        f"Final records: "
        f"{len(records)}"
    )

    print(
        f"Docling chunks: "
        f"{len(chunks)}"
    )

    print(
        f"Tables extracted: "
        f"{len(tables)}"
    )

    print(
        f"Tables summarised: "
        f"{stats['summarised']} / "
        f"{len(tables)}"
    )

    print(
        f"Floor merges: "
        f"{stats['floor_merges']}"
    )

    print(
        f"Prose merges: "
        f"{stats['prose_merges']}"
    )

    print(
        f"Dropped furniture: "
        f"{len(stats['dropped'])}"
    )

    print(
        f"Types: "
        f"{dict(types)}"
    )

    if sizes:

        print(
            f"Median tokens: "
            f"{sizes[len(sizes) // 2]}"
        )

        print(
            f"Under 50 tokens: "
            f"{sum(1 for s in sizes if s < 50)}"
        )

        print(
            f"Maximum tokens: "
            f"{sizes[-1]}"
        )

    print(
        "=" * 70
    )

    # ─────────────────────────────────────────────────────────────────────────
    # DROPPED FURNITURE
    # ─────────────────────────────────────────────────────────────────────────

    for page, reason, preview in (
        stats["dropped"]
    ):

        print(
            f"    dropped p{page}: "
            f"{reason}: "
            f"{preview}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # TABLE DIAGNOSTICS
    # ─────────────────────────────────────────────────────────────────────────

    for (
        page,
        problems,
        used_image,
    ) in stats["repaired"]:

        route = (
            "described from rendered image"
            if used_image
            else "FELL BACK TO BROKEN MARKDOWN"
        )

        print(
            f"    table on p{page} "
            f"has bad structure "
            f"({'; '.join(problems)}): "
            f"{route}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # OVERSIZED
    # ─────────────────────────────────────────────────────────────────────────

    if stats["over"]:

        print(
            f"  WARNING: truncated "
            f"{len(stats['over'])} chunks "
            f"to {CHUNK_TOKENS} tokens"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ORPHAN TABLES
    # ─────────────────────────────────────────────────────────────────────────

    orphaned = (
        len(tables)
        - len(table_groups)
    )

    if tables and not table_groups:

        print(
            f"  ERROR: {len(tables)} tables "
            "were extracted but none "
            "could be linked to a chunk."
        )

    elif orphaned:

        print(
            f"  WARNING: {orphaned} of "
            f"{len(tables)} tables produced "
            "no chunk."
        )


# ═════════════════════════════════════════════════════════════════════════════
# BUILD RECORDS
# ═════════════════════════════════════════════════════════════════════════════

def build_records(
    doc,
    pdf: Path,
    doc_id: str,
    doc_date: str,
    figure_uris: dict[str, str] | None = None,
) -> list[dict]:

    """
    Complete document chunking pipeline.

    Steps:

        1. Clean headings
        2. Configure HybridChunker
        3. Chunk document
        4. Extract table markdown
        5. Convert chunks to entries
        6. Merge prose
        7. Apply minimum chunk floor
        8. Create record factory
        9. Convert entries to records
        10. Generate Gemini table summaries
        11. Finalize reading order
        12. Apply metadata limits
        13. Truncate oversized records
        14. Print diagnostics

    Returns:

        list[dict]
    """

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 1 — CLEAN HEADINGS
    # ═════════════════════════════════════════════════════════════════════════

    try:

        clean_headings(
            doc
        )

    except TypeError:

        # If clean_headings() in your version expects
        # a different argument, do not stop the
        # complete pipeline here.
        pass

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 2 — CHUNKER
    # ═════════════════════════════════════════════════════════════════════════

    chunker = _make_chunker()

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 3 — CHUNK DOCUMENT
    # ═════════════════════════════════════════════════════════════════════════

    chunks = list(
        chunker.chunk(
            dl_doc=doc
        )
    )

    print(
        f"  Docling chunks: "
        f"{len(chunks)}",
        flush=True,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 4 — TABLE MARKDOWN
    # ═════════════════════════════════════════════════════════════════════════

    tables = table_markdown(
        doc
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 5 — CHUNKS → ENTRIES
    # ═════════════════════════════════════════════════════════════════════════

    entries, dropped = _to_entries(
        chunks=chunks,
        chunker=chunker,
        figure_uris=figure_uris or {},
    )

    print(
        f"  Entries before merge: "
        f"{len(entries)}",
        flush=True,
    )

    print(
        f"  Dropped furniture: "
        f"{len(dropped)}",
        flush=True,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 6 — MERGE PROSE
    # ═════════════════════════════════════════════════════════════════════════

    entries, prose_merges = _merge_prose(
        entries
    )

    print(
        f"  Prose merges: "
        f"{prose_merges}",
        flush=True,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 7 — MINIMUM FLOOR
    # ═════════════════════════════════════════════════════════════════════════

    entries, floor_merges = _apply_floor(
        entries
    )

    print(
        f"  Floor merges: "
        f"{floor_merges}",
        flush=True,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 8 — RECORD FACTORY
    # ═════════════════════════════════════════════════════════════════════════

    make = _record_factory(
        pdf=pdf,
        doc_id=doc_id,
        doc_date=doc_date,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 9 — DOCUMENT ITEM LOOKUP
    # ═════════════════════════════════════════════════════════════════════════

    items_by_ref = {}

    for item, _ in doc.iterate_items():

        ref = getattr(
            item,
            "self_ref",
            None,
        )

        if ref:

            items_by_ref[
                ref
            ] = item

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 10 — ENTRIES → RECORDS
    # ═════════════════════════════════════════════════════════════════════════

    records, table_groups, record_stats = (
        _to_records(
            entries=entries,
            doc=doc,
            items_by_ref=items_by_ref,
            figure_uris=figure_uris or {},
            make=make,
        )
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 11 — TABLE SUMMARIES
    # ═════════════════════════════════════════════════════════════════════════

    summaries, table_stats = _table_summaries(
        table_groups=table_groups,
        tables=tables,
        doc=doc,
        items_by_ref=items_by_ref,
        make=make,
    )

    # IMPORTANT:
    #
    # Table summaries are ADDITIONAL records.
    #
    # Original table fragments remain in records.
    #
    records += summaries

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 12 — FINALISE
    # ═════════════════════════════════════════════════════════════════════════

    records, over = _finalise(
        records
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 13 — REPORT
    # ═════════════════════════════════════════════════════════════════════════

    stats = {
        **record_stats,

        **table_stats,

        "prose_merges": prose_merges,

        "floor_merges": floor_merges,

        "dropped": dropped,

        "over": over,
    }

    _report(
        records=records,
        chunks=chunks,
        tables=tables,
        table_groups=table_groups,
        stats=stats,
    )

    return records

def _table_summaries(
    table_groups: dict,
    tables: dict,
    doc,
    items_by_ref: dict,
    make,
) -> tuple[list[dict], dict]:
    """
    Generate summary records for grouped tables.

    Uses:
      - Markdown table when the table is healthy
      - Table image when the markdown looks broken
      - Gemini-backed summarize_table / summarize_table_image
        from rag.tables
    """

    records = []
    summarised = 0
    skipped = []
    repaired = []

    for ref, fragments in table_groups.items():

        if not fragments:
            continue

        first = fragments[0]["meta"]

        # Get markdown for this table
        markdown = tables.get(ref, "")

        # ---------------------------------------------------------
        # No markdown available
        # ---------------------------------------------------------
        if not markdown:
            skipped.append({
                "page": first["page"],
                "table_id": first["table_id"],
                "reason": "missing_markdown",
            })
            continue

        # ---------------------------------------------------------
        # Check whether this table needs a summary
        # ---------------------------------------------------------
        if not needs_summary(markdown):
            skipped.append({
                "page": first["page"],
                "table_id": first["table_id"],
                "reason": "summary_not_needed",
            })
            continue

        # ---------------------------------------------------------
        # Check whether markdown looks broken
        # ---------------------------------------------------------
        problems = table_looks_broken(markdown)

        summary = None
        source = "markdown"

        # ---------------------------------------------------------
        # If markdown is broken, try table image first
        # ---------------------------------------------------------
        if problems:

            item = items_by_ref.get(ref)

            if item is not None:
                try:
                    summary = summarize_table_image(
                        item,
                        doc,
                        first["headings"],
                    )

                    if summary is not None:
                        source = "image"

                except Exception as exc:
                    repaired.append({
                        "page": first["page"],
                        "table_id": first["table_id"],
                        "problems": problems,
                        "image_error": str(exc),
                        "repaired": False,
                    })

            # Record repair attempt even if no exception occurred
            if not repaired or repaired[-1].get("table_id") != first["table_id"]:
                repaired.append({
                    "page": first["page"],
                    "table_id": first["table_id"],
                    "problems": problems,
                    "repaired": summary is not None,
                })

        # ---------------------------------------------------------
        # If image summary was not generated,
        # fall back to markdown summary
        # ---------------------------------------------------------
        if summary is None:

            try:
                summary = summarize_table(
                    markdown,
                    first["headings"],
                )
                source = "markdown"

            except Exception as exc:
                skipped.append({
                    "page": first["page"],
                    "table_id": first["table_id"],
                    "reason": f"summary_error: {exc}",
                })
                continue

        # ---------------------------------------------------------
        # Create table summary record
        # ---------------------------------------------------------
        summary_record = make(
            summary,
            {
                "page": first["page"],
                "page_end": fragments[-1]["meta"]["page_end"],
                "headings": first["headings"],
                "section_id": first["section_id"],
                "content_type": "table_summary",
                "table_id": first["table_id"],
                "table_chars": len(markdown),
                "n_fragments": len(fragments),
                "summary_source": source,
            },
            first["position"],
        )

        records.append(summary_record)

        summarised += 1

    return records, {
        "summarised": summarised,
        "skipped": skipped,
        "repaired": repaired,
    }