import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.api.schemas import (
    IngestPdfBatchResponse,
    IngestedResume,
    IngestTextBatchRequest,
    RecommendRequest,
    RecommendResponse,
)
from src.api.service import run_recommendation
from src.data.pdf_extract import extract_text_from_pdf
from src.embeddings.ingest import ingest_resume_entries

app = FastAPI(title="Talent-Gig Matching API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest) -> RecommendResponse:
    try:
        data = run_recommendation(
            job_description=payload.job_description,
            top_k=payload.top_k,
        )
        return RecommendResponse(**data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ingest/resume-pdfs", response_model=IngestPdfBatchResponse)
async def ingest_resume_pdfs(files: list[UploadFile] = File(...)) -> IngestPdfBatchResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file is required")
    pairs: list[tuple[str, str]] = []
    extras: list[dict[str, str]] = []
    out: list[IngestedResume] = []
    for upload in files:
        name = upload.filename or "unnamed.pdf"
        if not name.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF uploads are supported, got {name!r}",
            )
        data = await upload.read()
        try:
            text = extract_text_from_pdf(data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"{name}: {exc}") from exc
        rid = f"upload_{uuid.uuid4().hex}"
        pairs.append((rid, text))
        extras.append({"filename": name[:220], "source": "pdf_api"})
        out.append(IngestedResume(resume_id=rid, char_count=len(text), filename=name))
    try:
        ingest_resume_entries(pairs, extra_metadatas=extras)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return IngestPdfBatchResponse(ingested=out)


@app.post("/ingest/resume-texts", response_model=IngestPdfBatchResponse)
def ingest_resume_texts(payload: IngestTextBatchRequest) -> IngestPdfBatchResponse:
    pairs = [(item.resume_id, item.text) for item in payload.items]
    extras = [{"source": "text_api"} for _ in pairs]
    try:
        ingest_resume_entries(pairs, extra_metadatas=extras)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    out = [IngestedResume(resume_id=item.resume_id, char_count=len(item.text)) for item in payload.items]
    return IngestPdfBatchResponse(ingested=out)
