# Model Card: Talent-Gig Matcher

> **Authoritative technical walkthrough** (architecture, training data, hyperparameters): [`COMPLETE_PROJECT_GUIDE.md`](COMPLETE_PROJECT_GUIDE.md).  
> Keep this file as a **published-facing** summary for stakeholders.

## Model
- Type: Siamese-style transformer ranking model
- Framework: PyTorch
- Base encoder: `sentence-transformers/all-MiniLM-L6-v2`

## Intended Use
- Rank candidate resumes for a given gig/job description.
- Intended for recruiter-assistive shortlisting, not autonomous hiring decisions.

## Training Data
- Weakly supervised resume-job pairs built from:
  - embedding similarity
  - skill overlap
  - top-k positive + random negative sampling per job

## Evaluation
- Primary metrics: Precision@k, Recall@k, MRR
- Inference benchmark: see `data/artifacts/reports/benchmark.json`

## Limitations
- Weak labels introduce noise and can encode dataset bias.
- Resume quality and domain coverage affect recommendation quality.
- Should be used with human-in-the-loop review.

## Responsible AI Notes
- Do not use protected attributes in ranking logic.
- Monitor disparate impact across candidate groups.
- Keep an audit trail for model versions and metric drift.
