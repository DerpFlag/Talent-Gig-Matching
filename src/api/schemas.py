from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    job_description: str = Field(..., min_length=10)
    top_k: int | None = Field(default=None, gt=0)


class CandidateExplanation(BaseModel):
    matched_skills: list[str]
    missing_job_skills: list[str]
    matched_skill_count: int


class CandidateResult(BaseModel):
    resume_id: str
    retrieval_distance: float
    rerank_score: float
    resume_text: str
    explanation: CandidateExplanation


class RecommendResponse(BaseModel):
    job_text: str
    top_k: int
    candidates: list[CandidateResult]


class IngestedResume(BaseModel):
    resume_id: str
    char_count: int
    filename: str | None = None


class IngestPdfBatchResponse(BaseModel):
    ingested: list[IngestedResume]


class IngestTextItem(BaseModel):
    resume_id: str = Field(..., min_length=1, max_length=256)
    text: str = Field(..., min_length=20)


class IngestTextBatchRequest(BaseModel):
    items: list[IngestTextItem] = Field(..., min_length=1, max_length=200)
