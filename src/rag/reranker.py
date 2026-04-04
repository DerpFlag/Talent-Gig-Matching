from typing import Any
from functools import lru_cache

import torch
from transformers import AutoTokenizer

from src.models.siamese_model import SiameseMatcher


class PairReranker:
    def __init__(self, encoder_name: str, model_path: str, max_length: int) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be > 0")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.model = SiameseMatcher(encoder_name=encoder_name)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length

    @torch.no_grad()
    def score(self, job_text: str, resume_texts: list[str]) -> list[float]:
        if not job_text:
            raise ValueError("job_text cannot be empty")
        if not resume_texts:
            raise ValueError("resume_texts cannot be empty")

        job_batch = self.tokenizer(
            [job_text] * len(resume_texts),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        resume_batch = self.tokenizer(
            resume_texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        logits = self.model(
            job_input_ids=job_batch["input_ids"].to(self.device),
            job_attention_mask=job_batch["attention_mask"].to(self.device),
            resume_input_ids=resume_batch["input_ids"].to(self.device),
            resume_attention_mask=resume_batch["attention_mask"].to(self.device),
        )
        return torch.sigmoid(logits).detach().cpu().tolist()


@lru_cache(maxsize=4)
def get_cached_reranker(encoder_name: str, model_path: str, max_length: int) -> PairReranker:
    return PairReranker(encoder_name=encoder_name, model_path=model_path, max_length=max_length)


def rerank_candidates(
    reranker: PairReranker,
    job_text: str,
    candidate_ids: list[str],
    candidate_texts: list[str],
    candidate_distances: list[float],
) -> list[dict[str, Any]]:
    if not (len(candidate_ids) == len(candidate_texts) == len(candidate_distances)):
        raise ValueError("Candidate arrays must have identical length.")
    scores = reranker.score(job_text=job_text, resume_texts=candidate_texts)
    rows = []
    for rid, text, distance, score in zip(candidate_ids, candidate_texts, candidate_distances, scores):
        rows.append(
            {
                "resume_id": str(rid),
                "retrieval_distance": float(distance),
                "rerank_score": float(score),
                "resume_text": text,
            }
        )
    rows.sort(key=lambda x: x["rerank_score"], reverse=True)
    return rows
