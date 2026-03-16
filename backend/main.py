"""
LitReview AI — FastAPI Backend

IMPORTANT: load_dotenv MUST be called before any other local imports.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from routers import review_router
from utils.state_manager import state_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_pk = os.environ.get("GROQ_API_KEY", "")
_wk = os.environ.get("GROQ_API_KEY_WRITER", "")
logger.info(f"GROQ_API_KEY        : {'SET (' + _pk[:8] + '...)' if _pk else 'NOT SET'}")
logger.info(f"GROQ_API_KEY_WRITER : {'SET (' + _wk[:8] + '...)' if _wk else 'NOT SET'}")

app = FastAPI(
    title="LitReview AI",
    description="Autonomous multi-agent literature review generator with WebSocket streaming",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR   = BASE_DIR / "frontend" / "static"
TEMPLATE_DIR = BASE_DIR / "frontend" / "templates"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(review_router.router, prefix="/api", tags=["review"])


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open(TEMPLATE_DIR / "index.html", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health_check():
    return {
        "status":            "healthy",
        "version":           "2.0.0",
        "groq_primary_key":  bool(_pk),
        "groq_writer_key":   bool(_wk),
        "features":          ["websocket_streaming", "stage_checkpoints", "apa_references"],
    }


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    status = state_manager.get_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
