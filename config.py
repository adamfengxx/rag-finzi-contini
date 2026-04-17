"""
Central configuration for the Il Giardino dei Finzi-Contini RAG project.
All paths and hyperparameters live here — edit once, applies everywhere.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent
DATA_DIR   = ROOT_DIR / "data"
RAW_DIR    = ROOT_DIR / "raw_data"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", ROOT_DIR / "chroma_db"))

EPUB_FILE  = RAW_DIR / "Il giardino dei Finzi-Contini (Giorgio Bassani) (z-library.sk, 1lib.sk, z-lib.sk).epub"
MD_FILE    = PROCESSED_DIR / "il_giardino_dei_finzi_contini.md"

# ── Book metadata ────────────────────────────────────────────────────────────
BOOK_TITLE  = "Il giardino dei Finzi-Contini"
BOOK_AUTHOR = "Giorgio Bassani"
BOOK_LANG   = "it"

# ── Chunking ─────────────────────────────────────────────────────────────────
# 700 chars ~ 200 tokens for Italian prose; generous overlap preserves context
CHUNK_SIZE    = 700   # characters (RecursiveCharacterTextSplitter unit)
CHUNK_OVERLAP = 120   # characters (~17%)

# ── Embedding model ──────────────────────────────────────────────────────────
# multilingual-e5-large: top-tier Italian support, free, runs locally
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
# Instruction prefix required by E5 models for query vs. passage
E5_QUERY_PREFIX   = "query: "
E5_PASSAGE_PREFIX = "passage: "

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "finzi_contini"

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K_RETRIEVE = 10   # initial dense retrieval candidates
TOP_K_RERANK   = 4    # after reranker, pass this many to LLM

# ── Reranker ─────────────────────────────────────────────────────────────────
# bge-reranker-v2-m3: multilingual cross-encoder, strong Italian support
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# ── Generation model ─────────────────────────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
# Override via OPENAI_CHAT_MODEL env var if needed
CHAT_MODEL       = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
CHAT_TEMPERATURE = 0.1   # low temperature for factual RAG answers
MAX_TOKENS       = 1024
