"""Configuration: loads environment variables and defines constants."""
import os
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# --- API keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

# --- Paths ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DOCS_DIR = os.path.join(PROJECT_ROOT, "data", "docs")
INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "index.npz")
CHUNKS_PATH = os.path.join(PROJECT_ROOT, "data", "chunks.json")
EVAL_DIR = os.path.join(PROJECT_ROOT, "eval")

# --- Retrieval settings ---
CHUNK_SIZE_WORDS = 130        # ~130 words per chunk
CHUNK_OVERLAP_WORDS = 20      # 20 word overlap
TOP_K = 4                     # number of chunks to retrieve

# --- Evaluator thresholds ---
UPPER_THRESHOLD = 0.7         # score >= this -> CORRECT
LOWER_THRESHOLD = 0.4         # score < this -> INCORRECT; between -> AMBIGUOUS

# --- Embedding model ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
