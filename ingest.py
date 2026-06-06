from pathlib import Path
import re

import chromadb
from sentence_transformers import SentenceTransformer


DOCUMENTS_DIR = Path("documents")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "international_student_survival"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 75


def clean_text(text: str) -> str:
    """Clean copied Reddit/page text before chunking."""
    noise_patterns = [
        r"Skip to main content",
        r"Open menu",
        r"Log In",
        r"Expand user menu",
        r"Create Post",
        r"Join",
        r"Search in",
        r"Reddit Rules",
        r"Privacy Policy",
        r"User Agreement",
        r"Promoted",
        r"Advertisement",
        r"Sort by:",
        r"Best",
        r"Top",
        r"New",
    ]

    for pattern in noise_patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - chunk_overlap

    return chunks


def load_and_chunk_documents() -> list[dict]:
    """Load every .txt file, clean it, and split it into chunks."""
    all_chunks = []

    text_files = sorted(DOCUMENTS_DIR.glob("*.txt"))

    if not text_files:
        raise FileNotFoundError("No .txt files found inside the documents folder.")

    for file_path in text_files:
        raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
        cleaned_text = clean_text(raw_text)
        chunks = chunk_text(cleaned_text)

        for index, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "id": f"{file_path.stem}_chunk_{index}",
                    "text": chunk,
                    "metadata": {
                        "source": file_path.name,
                        "chunk_index": index,
                    },
                }
            )

    return all_chunks


def get_existing_collection_names(client) -> list[str]:
    """Handle different ChromaDB versions safely."""
    collections = client.list_collections()
    names = []

    for collection in collections:
        if hasattr(collection, "name"):
            names.append(collection.name)
        else:
            names.append(str(collection))

    return names


def main() -> None:
    print("Loading and chunking documents...")
    chunks = load_and_chunk_documents()
    print(f"Created {len(chunks)} chunks.")

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [chunk["text"] for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    print("Creating embeddings...")
    embeddings = model.encode(texts).tolist()

    print("Saving chunks to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    existing_collection_names = get_existing_collection_names(client)

    if COLLECTION_NAME in existing_collection_names:
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.create_collection(name=COLLECTION_NAME)

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print("Done. Your document database is ready.")


if __name__ == "__main__":
    main()
