"""Shared API clients and the probed embedding dimension.

Separate from `config` so that importing configuration does not require API keys.
Anything that needs to talk to a provider imports from here; anything that only
needs a setting does not.
"""

import os

from google import genai
from pinecone import Pinecone

from .config import EMBED_MODEL

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Probed rather than assumed. Dimension varies with the embedding model,
# so we determine it by creating a test embedding.
EMBED_DIMS = len(
    client.models.embed_content(
        model=EMBED_MODEL,
        contents="probe"
    ).embeddings[0].values
)