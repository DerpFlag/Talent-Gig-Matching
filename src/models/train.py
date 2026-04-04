from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.models.dataset import JobResumePairDataset, build_tokenizer
from src.models.losses import build_bce_loss
from src.models.siamese_model import SiameseMatcher
from src.utils.io import ensure_parent, read_csv_strict


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_training_frame(labels_path: str, jobs_path: str, resumes_path: str) -> pd.DataFrame:
    labels = read_csv_strict(labels_path)
    jobs = read_csv_strict(jobs_path)
    resumes = read_csv_strict(resumes_path)

    required_labels = {"job_id", "resume_id", "label"}
    required_jobs = {"job_id", "job_text"}
    required_resumes = {"resume_id", "resume_text"}
    if required_labels - set(labels.columns):
        raise ValueError(f"Missing label columns: {required_labels - set(labels.columns)}")
    if required_jobs - set(jobs.columns):
        raise ValueError(f"Missing job columns: {required_jobs - set(jobs.columns)}")
    if required_resumes - set(resumes.columns):
        raise ValueError(f"Missing resume columns: {required_resumes - set(resumes.columns)}")

    frame = labels.merge(jobs[["job_id", "job_text"]], on="job_id", how="inner")
    frame = frame.merge(resumes[["resume_id", "resume_text"]], on="resume_id", how="inner")
    if frame.empty:
        raise ValueError("Merged training dataframe is empty.")
    return frame


def _build_loaders(
    frame: pd.DataFrame,
    model_name: str,
    max_length: int,
    batch_size: int,
    validation_size: float,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    unique_jobs = frame["job_id"].unique()
    if len(unique_jobs) < 2:
        raise ValueError("Need at least 2 unique jobs for train/validation split.")

    train_jobs, val_jobs = train_test_split(
        unique_jobs,
        test_size=validation_size,
        random_state=seed,
        shuffle=True,
    )
    train_df = frame[frame["job_id"].isin(train_jobs)].copy()
    val_df = frame[frame["job_id"].isin(val_jobs)].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Train/validation split resulted in empty split.")

    tokenizer = build_tokenizer(model_name)
    train_ds = JobResumePairDataset(train_df, tokenizer=tokenizer, max_length=max_length)
    val_ds = JobResumePairDataset(val_df, tokenizer=tokenizer, max_length=max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _run_epoch(
    model: SiameseMatcher,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
) -> float:
    training = optimizer is not None
    model.train() if training else model.eval()
    losses = []
    for batch in loader:
        for key in batch:
            batch[key] = batch[key].to(device)
        logits = model(
            job_input_ids=batch["job_input_ids"],
            job_attention_mask=batch["job_attention_mask"],
            resume_input_ids=batch["resume_input_ids"],
            resume_attention_mask=batch["resume_attention_mask"],
        )
        loss = criterion(logits, batch["labels"])
        losses.append(float(loss.detach().cpu().item()))

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
    if not losses:
        raise ValueError("No batches were produced by loader.")
    return float(np.mean(losses))


def train_matcher(
    labels_path: str,
    jobs_path: str,
    resumes_path: str,
    model_name: str,
    max_length: int,
    batch_size: int,
    epochs: int,
    lr_encoder: float,
    lr_head: float,
    weight_decay: float,
    validation_size: float,
    random_seed: int,
    positive_class_weight: float,
    model_output_path: str,
) -> dict:
    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    if not (0 < validation_size < 1):
        raise ValueError("validation_size must be in (0,1)")

    set_seed(random_seed)
    frame = _load_training_frame(labels_path, jobs_path, resumes_path)
    train_loader, val_loader = _build_loaders(
        frame=frame,
        model_name=model_name,
        max_length=max_length,
        batch_size=batch_size,
        validation_size=validation_size,
        seed=random_seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseMatcher(encoder_name=model_name).to(device)
    criterion = build_bce_loss(positive_class_weight=positive_class_weight, device=device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": lr_encoder},
            {"params": model.classifier.parameters(), "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        val_loss = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ensure_parent(model_output_path)
            torch.save(model.state_dict(), model_output_path)

    metrics_path = str(Path(model_output_path).with_suffix(".metrics.json"))
    Path(metrics_path).write_text(json.dumps({"best_val_loss": best_val_loss, "history": history}, indent=2), encoding="utf-8")
    return {"best_val_loss": best_val_loss, "history": history, "model_output_path": model_output_path}
