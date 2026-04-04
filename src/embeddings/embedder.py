from sentence_transformers import SentenceTransformer


def encode_texts(model_name: str, texts: list[str]) -> list[list[float]]:
    if not texts:
        raise ValueError("texts cannot be empty")
    model = SentenceTransformer(model_name)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return vectors.tolist()
