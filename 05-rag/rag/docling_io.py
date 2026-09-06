"""Parsing a PDF, and reading the objects Docling returns.

    PDF
     |
     v
    six models, in order
     |
     +--  layout          finds the boxes, labels them, sets reading order
     +--  TableFormer     rows and columns inside a table box
     +--  OCR             text on scanned pages with no text layer
     +--  classifier      chart / photo / logo / diagram
     +--  CodeFormula     equations and code
     +--  vision model    writes each chart's description
     |
     v
    DoclingDocument       a graph of typed objects, not a string

WHAT THIS FILE DOES NOT DO

    It does NOT chunk.
    It does NOT embed.
    It does NOT decide what is worth keeping.

It returns the parsed document. Everything after that is inspect.py,
tables.py and chunking.py.

THE TRAP: almost every enrichment is OFF by default. A default
PdfPipelineOptions() gives you text and little else — an equation becomes the
placeholder `formula-not-decoded` and its content is gone, with no error.

"Parsing" is several models in sequence — layout analysis, table structure, OCR,
figure classification, formula reading, chart extraction, and a remote vision model
for figure descriptions. `build_pipeline_options` documents which is which and what
each one's failure looks like.

The accessor functions exist because Docling's object model moves between releases,
and a naive read of a renamed attribute returns an empty list rather than raising —
so a capability silently stops working with no error anywhere.
"""

import io
import os
import time
import warnings
from pathlib import Path

from .config import (
    DO_CHART_EXTRACTION,
    DO_CLASSIFICATION,
    DO_CODE,
    DO_FORMULA,
    FIGURE_AREA_THRESHOLD,
    FIGURE_PROMPT,
    FIGURE_RENDER_SCALE,
    TABLE_MODE_ACCURATE,
    VISION_MODEL,
)


def picture_annotations(item) -> list:
    """Annotations attached to a picture, across Docling versions."""

    meta = getattr(item, "meta", None)

    if meta is not None:

        # Current layout: annotations hang off meta.
        found = getattr(
            meta,
            "annotations",
            None,
        )

        if found:
            return list(found)

        # Some builds make meta itself the sequence.
        if isinstance(
            meta,
            (list, tuple),
        ) and meta:

            return list(meta)

    # Deprecated location.
    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore",
            DeprecationWarning,
        )

        return list(
            getattr(
                item,
                "annotations",
                None,
            )
            or []
        )


def picture_description(item) -> str | None:
    """The natural-language description of a picture, if one was produced."""

    annotations = picture_annotations(item)

    # A picture can carry several annotations — a classification,
    # a description, possibly extracted chart data.
    for annotation in annotations:

        if (
            getattr(
                annotation,
                "kind",
                "",
            )
            == "description"
            and getattr(
                annotation,
                "text",
                None,
            )
        ):

            return annotation.text

    # Older builds tag it differently.
    for annotation in annotations:

        text = getattr(
            annotation,
            "text",
            None,
        )

        if text:
            return text

    return None


def chart_data(item):
    """Structured series extracted from a chart."""

    for annotation in picture_annotations(item):

        for field in (
            "chart_data",
            "data",
            "series",
            "table",
        ):

            value = getattr(
                annotation,
                field,
                None,
            )

            if value:
                return value

    return None


def check_model_access() -> None:
    """Fail early and legibly if Docling's model downloads will be rejected."""

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )
    )

    if not token:
        return

    try:

        from huggingface_hub import HfApi

        HfApi().model_info(
            "docling-project/DocumentFigureClassifier-v2.5"
        )

    except Exception as exc:

        if (
            "401" in str(exc)
            or "expired" in str(exc).lower()
        ):

            os.environ.pop(
                "HF_TOKEN",
                None,
            )

            os.environ.pop(
                "HUGGING_FACE_HUB_TOKEN",
                None,
            )

            print(
                "  NOTE: the HuggingFace token in this environment is rejected; "
                "continuing anonymously (Docling's models are public)",
                flush=True,
            )

        else:
            raise


def build_pipeline_options():
    """Assemble the Docling pipeline configuration."""

    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        PictureDescriptionApiOptions,
        TableFormerMode,
    )

    # Enrichment flag names have moved between releases.
    flags = {
        name: field.default
        for name, field in PdfPipelineOptions.model_fields.items()
        if name.startswith(
            (
                "do_",
                "generate_",
                "enable_",
            )
        )
    }

    opts = PdfPipelineOptions()

    # TableFormer.
    opts.do_table_structure = True

    opts.table_structure_options.mode = (
        TableFormerMode.ACCURATE
        if TABLE_MODE_ACCURATE
        else TableFormerMode.FAST
    )
    opts.do_ocr = False

    # Requested enrichment models.
    requested = {

        "do_formula_enrichment": DO_FORMULA,

        "do_code_enrichment": DO_CODE,

        "do_picture_classification": DO_CLASSIFICATION,

        # Gemini vision model.
        "do_picture_description": True,

        # Required to create figure images.
        "generate_picture_images": True,

        # Required for table image rendering.
        "generate_table_images": True,

        # Required for remote Gemini API.
        "enable_remote_services": True,
    }

    # Chart extraction flag varies between Docling releases.
    for alias in (
        "do_chart_extraction",
        "do_chart_data_extraction",
        "do_chart_understanding",
        "do_picture_data",
    ):

        if alias in flags:

            requested[alias] = DO_CHART_EXTRACTION

            break

    else:

        print(
            "  NOTE: this docling build exposes no chart-extraction flag; "
            "charts will be described but their numeric series not read",
            flush=True,
        )

    applied = []
    unavailable = []

    for flag, value in requested.items():

        if flag in flags:

            setattr(
                opts,
                flag,
                value,
            )

            applied.append(flag)

        else:

            unavailable.append(flag)

    opts.images_scale = FIGURE_RENDER_SCALE

    # ═════════════════════════════════════════════════════════════════════
    # GOOGLE GEMINI
    # ═════════════════════════════════════════════════════════════════════
    #
    # Docling's PictureDescriptionApiOptions expects an OpenAI-compatible
    # HTTP endpoint.
    #
    # Google Gemini provides an OpenAI-compatible endpoint, so Docling can
    # send its picture-description request to Gemini without using
    # OPENAI_API_KEY.
    #
    # Google endpoint:
    #
    # https://generativelanguage.googleapis.com/v1beta/openai/
    #
    # The API key comes from:
    #
    # GEMINI_API_KEY
    #
    # ═════════════════════════════════════════════════════════════════════

    gemini_api_key = os.environ.get(
        "GEMINI_API_KEY"
    )

    if not gemini_api_key:

        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Add GEMINI_API_KEY to your .env file."
        )

    opts.picture_description_options = (
        PictureDescriptionApiOptions(
            url=(
                "https://generativelanguage.googleapis.com/"
                "v1beta/openai/chat/completions"
            ),

            headers={
                "Authorization": (
                    f"Bearer {gemini_api_key}"
                ),
                "Content-Type": "application/json",
            },

            params={
                "model": VISION_MODEL,
                "max_tokens": 400,
                "temperature": 0,
            },

            prompt=FIGURE_PROMPT,

            picture_area_threshold=(
                FIGURE_AREA_THRESHOLD
            ),

            timeout=120,
        )
    )

    print(
        f"  enrichments: {', '.join(applied)}",
        flush=True,
    )

    if unavailable:

        print(
            "  NOT AVAILABLE in this docling build: "
            + ", ".join(unavailable),
            flush=True,
        )

    print(
        f"  vision model: {VISION_MODEL}",
        flush=True,
    )

    print(
        "  vision provider: Google Gemini",
        flush=True,
    )

    return opts


def parse_pdf(pdf: Path):
    """Parse a PDF into a DoclingDocument."""

    from docling.datamodel.base_models import InputFormat

    from docling.document_converter import (
        DocumentConverter,
        PdfFormatOption,
    )

    check_model_access()

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=build_pipeline_options()
            )
        }
    )

    started = time.time()

    doc = converter.convert(
        str(pdf)
    ).document

    elapsed = time.time() - started

    pages = len(doc.pages)

    print(
        f"  parsed in {elapsed:.0f}s  "
        f"({elapsed / max(pages, 1):.0f}s per page)",
        flush=True,
    )

    if (
        elapsed / max(pages, 1)
        > 30
    ):

        print(
            "  SLOW. Run `python profile_parse.py <pdf>` "
            "to see which enrichment is costing this — "
            "it times each one separately.",
            flush=True,
        )

    return doc


def save_figures(
    doc,
    doc_id: str,
    bucket: str | None,
) -> dict[str, str]:
    """Write each figure PNG to S3 and return a map of element ref -> URI.

    The pixels have already been rendered so the Gemini vision model could read them.
    """

    if not bucket:
        return {}

    import boto3

    from docling_core.types.doc import PictureItem

    s3 = boto3.client(
        "s3"
    )

    uris = {}

    for n, (item, _) in enumerate(
        doc.iterate_items()
    ):

        if not isinstance(
            item,
            PictureItem,
        ):
            continue

        image = item.get_image(
            doc
        )

        if image is None:
            continue

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="PNG",
        )

        key = (
            f"figures/"
            f"{doc_id}/"
            f"fig_{n:04d}.png"
        )

        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=buffer.getvalue(),
            ContentType="image/png",
        )

        uris[
            getattr(
                item,
                "self_ref",
                None,
            )
            or str(id(item))
        ] = (
            f"s3://{bucket}/{key}"
        )

    return uris