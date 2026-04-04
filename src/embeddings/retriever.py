from functools import lru_cache

from sentence_transformers import SentenceTransformer
import chromadb


@lru_cache(maxsize=4)
def _get_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@lru_cache(maxsize=8)
def _get_collection(chroma_dir: str, collection_name: str):
    client = chromadb.PersistentClient(path=chroma_dir)
    return client.get_collection(collection_name)


def clear_collection_cache() -> None:
    _get_collection.cache_clear()


def encode_texts_normalized(model_name: str, texts: list[str]) -> list[list[float]]:
    if not texts:
        raise ValueError("texts cannot be empty")
    model = _get_embedder(model_name)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def retrieve_top_k_resumes(
    chroma_dir: str,
    collection_name: str,
    embedding_model_name: str,
    job_text: str,
    top_k: int,
) -> dict:
    if not job_text or not isinstance(job_text, str):
        raise ValueError("job_text must be a non-empty string")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    model = _get_embedder(embedding_model_name)
    query_embedding = model.encode([job_text], normalize_embeddings=True)[0].tolist()

    collection = _get_collection(chroma_dir, collection_name)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results
