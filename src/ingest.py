"""Ingest documents: load, chunk, embed, and cache."""
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import (DOCS_DIR, INDEX_PATH, CHUNKS_PATH,
                         CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS, EMBEDDING_MODEL)

def load_documents(docs_dir: str = DOCS_DIR) -> list[dict]:
    """Load all .txt and .pdf files from the docs directory."""
    documents = []
    for filename in sorted(os.listdir(docs_dir)):
        filepath = os.path.join(docs_dir, filename)
        if filename.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            documents.append({"filename": filename, "text": text})
        elif filename.endswith('.pdf'):
            try:
                from pypdf import PdfReader
                reader = PdfReader(filepath)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                documents.append({"filename": filename, "text": text})
            except Exception as e:
                print(f"[Ingest] Warning: could not read {filename}: {e}")
    return documents

def chunk_text(text: str, source: str,
               chunk_size: int = CHUNK_SIZE_WORDS,
               overlap: int = CHUNK_OVERLAP_WORDS) -> list[dict]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append({"text": chunk_text, "source": source, "start_word": start})
        start += chunk_size - overlap
    return chunks

def build_index(docs_dir: str = DOCS_DIR) -> int:
    """Load docs, chunk, embed, and save index. Returns number of chunks."""
    # Load documents
    documents = load_documents(docs_dir)
    if not documents:
        raise ValueError(f"No documents found in {docs_dir}")

    # Chunk all documents
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["text"], doc["filename"])
        all_chunks.extend(chunks)

    # Embed chunks
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    # Save index and chunks
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    np.savez(INDEX_PATH, embeddings=embeddings)
    with open(CHUNKS_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"[Ingest] Indexed {len(all_chunks)} chunks from {len(documents)} documents.")
    return len(all_chunks)

if __name__ == "__main__":
    build_index()
