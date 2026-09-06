"""
Reading tables, judging whether the parse worked, and describing them.

    TableItem
        |
        v
    export_to_markdown()
        |
        v
    table_looks_broken()
        |
        +-- no --> summarise the text
        |
        +-- yes --> render the table as an image
                         |
                         v
                    Gemini Vision
                         |
                         v
                       summary

WHY A TABLE NEEDS A SUMMARY AT ALL

A grid of numbers shares almost no vocabulary with a question like
"how did revenue grow" — those words may appear nowhere in the cells.

The summary supplies the missing vocabulary. The raw rows remain indexed
alongside it and carry the exact values.

Gemini is used for:
    - Table text summaries
    - Table image summaries

The Gemini API key is read from GEMINI_API_KEY.
"""

import base64
import hashlib
import io
import os
import re

from google import genai

from .config import (
    CACHE_DIR,
    TABLE_MODEL,
    TABLE_PROMPT,
    TABLE_MIN_CELLS,
    VISION_MODEL,
)


# ═════════════════════════════════════════════════════════════════════
# GOOGLE GEMINI CLIENT
# ═════════════════════════════════════════════════════════════════════

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. "
        "Add GEMINI_API_KEY to your .env file."
    )

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ═════════════════════════════════════════════════════════════════════
# TABLE CELLS
# ═════════════════════════════════════════════════════════════════════

def table_cells(markdown: str) -> list[str]:
    """Cell contents of a markdown table.

    The count decides whether something is a real table, so it has to be honest.

    Two things would inflate it:
        - separator rows
        - empty strings created by splitting pipes
    """

    return [
        cell.strip()
        for line in markdown.splitlines()
        for cell in line.split("|")
        if cell.strip()
        and not set(cell.strip()) <= set("-: ")
    ]


# ═════════════════════════════════════════════════════════════════════
# DOES THE TABLE NEED A SUMMARY?
# ═════════════════════════════════════════════════════════════════════

def needs_summary(markdown: str) -> bool:
    """Whether this is a real table rather than a layout artefact."""

    return len(
        table_cells(markdown)
    ) >= TABLE_MIN_CELLS


# ═════════════════════════════════════════════════════════════════════
# CHECK WHETHER TABLE IS BROKEN
# ═════════════════════════════════════════════════════════════════════

def table_looks_broken(markdown: str) -> list[str]:
    """Signals that TableFormer failed to recover the grid.

    Two common failures:

        duplicate adjacent header cells

        label and value in one cell
    """

    rows = [
        line
        for line in markdown.splitlines()
        if "|"
        in line
        and not set(line.strip()) <= set("|-: ")
    ]

    if not rows:
        return []

    issues = []

    header = [
        cell.strip()
        for cell in rows[0].split("|")
        if cell.strip()
    ]

    if any(
        a == b and a
        for a, b in zip(
            header,
            header[1:],
        )
    ):
        issues.append(
            "duplicate adjacent header cells"
        )

    cells = table_cells(
        markdown
    )

    crammed = sum(
        1
        for cell in cells
        if re.search(
            r"[A-Za-z]{3,}\s+[\d,]{3,}",
            cell,
        )
    )

    if (
        cells
        and crammed / len(cells) > 0.10
    ):
        issues.append(
            f"{crammed}/{len(cells)} cells hold "
            f"a label and a number"
        )

    return issues


# ═════════════════════════════════════════════════════════════════════
# EXTRACT ALL TABLES
# ═════════════════════════════════════════════════════════════════════

def table_markdown(doc) -> dict[str, str]:
    """Full markdown for every table in the document.

    Keyed by the table's element reference.
    """

    from docling_core.types.doc import TableItem

    tables = {}

    for item, _ in doc.iterate_items():

        if not isinstance(
            item,
            TableItem,
        ):
            continue

        ref = (
            getattr(
                item,
                "self_ref",
                None,
            )
            or str(id(item))
        )

        try:

            tables[ref] = (
                item.export_to_markdown(
                    doc
                )
            )

        except Exception:

            tables[ref] = ""

    return tables


# ═════════════════════════════════════════════════════════════════════
# GET TABLE REFERENCE
# ═════════════════════════════════════════════════════════════════════

def table_ref_of(chunk) -> str | None:
    """The element ref of the table this chunk came from."""

    for item in chunk.meta.doc_items:

        label = str(
            getattr(
                item,
                "label",
                "",
            )
        ).lower()

        if "table" in label:

            ref = getattr(
                item,
                "self_ref",
                None,
            )

            if ref:
                return ref

    return None


# ═════════════════════════════════════════════════════════════════════
# SUMMARIZE TABLE USING GEMINI
# ═════════════════════════════════════════════════════════════════════

def summarize_table(
    markdown: str,
    headings: list[str],
) -> str:
    """Describe a table using Google Gemini.

    The result is cached by table content.
    """

    digest = hashlib.sha256(
        markdown.encode()
    ).hexdigest()[:20]

    cached = (
        CACHE_DIR
        / "tables"
        / f"{digest}.txt"
    )

    cached.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if cached.exists():
        return cached.read_text()

    context = (
        " > ".join(headings)
        if headings
        else ""
    )

    prompt = (
        f"{TABLE_PROMPT}\n\n"
        f"Section: {context}\n\n"
        f"{markdown[:12000]}"
    )

    # ═════════════════════════════════════════════════════════════════
    # GEMINI GENERATION
    # ═════════════════════════════════════════════════════════════════

    response = gemini_client.models.generate_content(
        model=TABLE_MODEL,
        contents=prompt,
    )

    summary = (
        response.text
        or ""
    ).strip()

    cached.write_text(
        summary
    )

    return summary


# ═════════════════════════════════════════════════════════════════════
# SUMMARIZE TABLE IMAGE USING GEMINI VISION
# ═════════════════════════════════════════════════════════════════════

def summarize_table_image(
    item,
    doc,
    headings: list[str],
) -> str | None:
    """Describe a table by looking at its rendered image.

    This is used when the parsed table structure is known to be broken.
    """

    image = item.get_image(
        doc
    )

    if image is None:
        return None

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    raw = buffer.getvalue()

    digest = hashlib.sha256(
        raw
    ).hexdigest()[:20]

    cached = (
        CACHE_DIR
        / "tables"
        / f"{digest}.img.txt"
    )

    cached.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if cached.exists():
        return cached.read_text()

    context = (
        " > ".join(headings)
        if headings
        else ""
    )

    prompt = (
        f"Section: {context}\n\n"
        f"{TABLE_PROMPT}"
    )

    # ═════════════════════════════════════════════════════════════════
    # GEMINI VISION
    # ═════════════════════════════════════════════════════════════════

    response = gemini_client.models.generate_content(
        model=VISION_MODEL,
        contents=[
            prompt,
            image,
        ],
    )

    summary = (
        response.text
        or ""
    ).strip()

    cached.write_text(
        summary
    )

    return summary