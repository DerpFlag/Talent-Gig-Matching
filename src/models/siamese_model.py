import torch
import torch.nn as nn
from transformers import AutoModel


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class SiameseMatcher(nn.Module):
    def __init__(self, encoder_name: str, dropout: float = 0.1) -> None:
        super().__init__()
        if not encoder_name:
            raise ValueError("encoder_name cannot be empty")
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return mean_pool(outputs.last_hidden_state, attention_mask)

    def forward(
        self,
        job_input_ids: torch.Tensor,
        job_attention_mask: torch.Tensor,
        resume_input_ids: torch.Tensor,
        resume_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        job_vec = self.encode(job_input_ids, job_attention_mask)
        resume_vec = self.encode(resume_input_ids, resume_attention_mask)
        pair_vec = torch.cat([torch.abs(job_vec - resume_vec), job_vec * resume_vec], dim=1)
        logits = self.classifier(pair_vec).squeeze(-1)
        return logits
