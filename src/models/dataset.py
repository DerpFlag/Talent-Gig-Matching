from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


@dataclass
class PairBatch:
    job_input_ids: torch.Tensor
    job_attention_mask: torch.Tensor
    resume_input_ids: torch.Tensor
    resume_attention_mask: torch.Tensor
    labels: torch.Tensor


class JobResumePairDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int,
    ) -> None:
        required = {"job_text", "resume_text", "label"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns in training dataframe: {missing}")
        if data.empty:
            raise ValueError("Training dataframe is empty")
        if max_length <= 0:
            raise ValueError("max_length must be > 0")
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        job = self.tokenizer(
            str(row["job_text"]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        resume = self.tokenizer(
            str(row["resume_text"]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "job_input_ids": job["input_ids"].squeeze(0),
            "job_attention_mask": job["attention_mask"].squeeze(0),
            "resume_input_ids": resume["input_ids"].squeeze(0),
            "resume_attention_mask": resume["attention_mask"].squeeze(0),
            "labels": torch.tensor(float(row["label"]), dtype=torch.float32),
        }


def build_tokenizer(model_name: str) -> AutoTokenizer:
    if not model_name:
        raise ValueError("model_name cannot be empty")
    return AutoTokenizer.from_pretrained(model_name)
