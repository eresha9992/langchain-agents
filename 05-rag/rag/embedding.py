"""
Embedding, with a cache keyed on content.

    texts[]
       |
       v
    hash each one            sha256(model + text)
       |
       v
    cached?  --yes-->  read from disk
       |
      no
       |
       v
    batch by token count and by count
       |
       v
    Google Gemini embeddings API
       |
       v
    write to cache  ->  ndarray

THE CACHE KEY IS (model, text), AND THAT MATTERS

The key fully determines the result, so this cache cannot go stale. Contrast a
cache keyed on a filename, where changing a setting returns work made under the
old one and the change appears to have done nothing.

Returns a numpy array, NOT lists. Seven times less memory at corpus scale, and
`sync.py` calls `.tolist()` at the point of upsert.

The cache is what makes experimentation free. Re-running after a chunking or
retrieval change re-embeds nothing, and boilerplate shared across documents is
embedded once for the whole corpus.

Its key is (model, text), which fully determines the result — so unlike a cache
keyed on a filename, this one cannot return work made under different settings.
"""

import hashlib
import time
from pathlib import Path

import numpy as np

from .clients import EMBED_DIMS, client
from .config import (
    EMBED_CACHE,
    EMBED_MODEL,
    ENCODING,
    GEMINI_EMBED_MAX_INPUTS,
    slugify,
)


def _cache_path(digest: str) -> Path:
    """Where a single embedding is cached on disk.

    Sharded by the first two characters of the digest. A corpus of a few thousand
    documents produces hundreds of thousands of these files, and most filesystems
    degrade badly when that many land in one directory — directory lookups go
    linear, and `ls` becomes unusable for debugging.

    The model name is part of the path, not just the digest, so switching embedding
    models cannot silently return vectors from the previous one.
    """

    shard = (
        EMBED_CACHE
        / slugify(EMBED_MODEL)
        / digest[:2]
    )

    shard.mkdir(
        parents=True,
        exist_ok=True,
    )

    return shard / f"{digest}.npy"


def _batches(texts: list[str]) -> list[list[int]]:
    """Group text indices into Gemini API requests.

    Gemini embedding requests are grouped according to the configured maximum
    number of inputs.

    Returns indices rather than the texts themselves so the caller can write
    results back into the right positions.
    """

    groups = []
    current = []

    for i, text in enumerate(texts):

        if (
            current
            and len(current) >= GEMINI_EMBED_MAX_INPUTS
        ):
            groups.append(current)
            current = []

        current.append(i)

    if current:
        groups.append(current)

    return groups


def embed(
    texts: list[str],
    use_cache: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """Embed texts using Google Gemini, caching each one by (model, text).

    Returns a float32 array rather than nested lists.

    The cache is what makes experimentation free. Re-running after a chunking or
    retrieval change re-embeds nothing, and boilerplate shared across documents is
    embedded once for the whole corpus.
    """

    digests = [
        hashlib.sha256(
            (
                EMBED_MODEL
                + "\x00"
                + t
            ).encode()
        ).hexdigest()[:24]
        for t in texts
    ]

    out = np.empty(
        (
            len(texts),
            EMBED_DIMS,
        ),
        dtype=np.float32,
    )

    missing = list(
        range(
            len(texts)
        )
    )

    # ═════════════════════════════════════════════════════════════════════
    # LOAD CACHED EMBEDDINGS
    # ═════════════════════════════════════════════════════════════════════

    if use_cache:

        missing = []

        for i, digest in enumerate(digests):

            path = _cache_path(
                digest
            )

            if path.exists():

                vector = np.load(
                    path
                )

                if vector.shape != (
                    EMBED_DIMS,
                ):
                    missing.append(i)
                    continue

                out[i] = vector

            else:

                missing.append(i)

    # ═════════════════════════════════════════════════════════════════════
    # CREATE NEW GEMINI EMBEDDINGS
    # ═════════════════════════════════════════════════════════════════════

    missing_texts = [
        texts[i]
        for i in missing
    ]

    for group in _batches(
        missing_texts
    ):

        # `_batches()` indexes into missing_texts, so map back to
        # the original text positions.
        indices = [
            missing[j]
            for j in group
        ]

        for attempt in range(4):

            try:

                response = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=[
                        texts[i]
                        for i in indices
                    ],
                )

                break

            except Exception:

                # Rate limits and transient network errors both land here.
                # Four attempts with exponential backoff covers temporary
                # failures without hanging forever.
                if attempt == 3:
                    raise

                time.sleep(
                    2 ** attempt
                )

        # Gemini returns one embedding for each input.
        for i, embedding in zip(
            indices,
            response.embeddings,
        ):

            vector = np.asarray(
                embedding.values,
                dtype=np.float32,
            )

            if vector.shape != (
                EMBED_DIMS,
            ):

                raise ValueError(
                    f"Gemini returned embedding dimension "
                    f"{vector.shape[0]}, expected {EMBED_DIMS}"
                )

            out[i] = vector

            if use_cache:

                np.save(
                    _cache_path(
                        digests[i]
                    ),
                    vector,
                )

    if verbose:

        print(
            f"{len(texts)} texts | "
            f"{len(texts) - len(missing)} cached | "
            f"{len(missing)} embedded"
        )

    return out


def embed_stream(
    texts: list[str],
    batch: int = 512,
    use_cache: bool = True,
):
    """Yield (offset, vectors) so the caller can upsert as it goes.

    Ingesting a large corpus should never hold every vector in memory at once.
    A 250-page document produces thousands of chunks, and materialising all of
    them before the first upsert makes peak memory a function of document size.

    Windowing keeps that flat regardless of how long the document is.
    """

    for start in range(
        0,
        len(texts),
        batch,
    ):

        yield (
            start,
            embed(
                texts[
                    start:start + batch
                ],
                use_cache=use_cache,
            ),
        )