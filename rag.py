import os

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "international_student_survival"
TOP_K = 4
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"


load_dotenv()


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")
    return Groq(api_key=api_key)


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION_NAME)


def retrieve_context(question: str, top_k: int = TOP_K) -> list[dict]:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    collection = get_collection()

    question_embedding = embedding_model.encode([question]).tolist()[0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    context_items = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        context_items.append({
            "text": text,
            "source": metadata.get("source", "unknown"),
            "chunk_index": metadata.get("chunk_index", "unknown"),
            "distance": distance,
        })
    return context_items


def build_prompt(question: str, context_items: list[dict]) -> str:
    context_text = ""
    for index, item in enumerate(context_items, start=1):
        context_text += (
            f"\n[Source {index}] {item['source']} | chunk {item['chunk_index']}\n"
            f"{item['text']}\n"
        )

    return f"""You are a helpful assistant for international students in the United States.

Answer the user's question using ONLY the context below.
If the context does not contain enough information, say:
"The documents do not provide enough information to answer this fully."

Be clear, practical, and easy to understand.
At the end, include a short "Sources used" section with the document names.

Context:
{context_text}

Question:
{question}

Answer:"""


def answer_question(question: str) -> dict:
    if not question.strip():
        return {"answer": "Please ask a question.", "sources": []}

    context_items = retrieve_context(question)

    if not context_items:
        return {"answer": "I could not find relevant information.", "sources": []}

    prompt = build_prompt(question, context_items)
    client = get_groq_client()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=700,
    )

    answer = response.choices[0].message.content
    sources = [
        f"{item['source']} (chunk {item['chunk_index']}, distance: {item['distance']:.3f})"
        for item in context_items
    ]

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    test_question = "What are the biggest mistakes international students make in their first semester?"
    print(f"QUESTION: {test_question}\n")
    print("="*70)
    result = answer_question(test_question)
    print(result["answer"])
    print("\n" + "="*70)
    print("RETRIEVED CHUNKS:")
    for src in result["sources"]:
        print(f"  • {src}")
