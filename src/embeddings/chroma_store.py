from pathlib import Path
import chromadb


def build_chroma_resume_index(
    chroma_dir: str,
    collection_name: str,
    resume_ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    Path(chroma_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(name=collection_name)
    collection.add(ids=resume_ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def upsert_resume_documents(
    chroma_dir: str,
    collection_name: str,
    resume_ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    Path(chroma_dir).mkdir(parents=True, exist_ok=True)
    if not (
        len(resume_ids) == len(embeddings) == len(documents) == len(metadatas)
    ):
        raise ValueError("resume_ids, embeddings, documents, and metadatas must have the same length")
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(name=collection_name)
    collection.upsert(
        ids=resume_ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
