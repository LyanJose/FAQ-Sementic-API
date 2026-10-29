
from contextlib import asynccontextmanager
from pathlib import Path
import os
import sqlite3

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("FAQ_DB_PATH", str(BASE_DIR / "faq.db")))
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

MATCH_THRESHOLD = 0.55
REVIEW_THRESHOLD = 0.40

model = None
faq_cache = []
faq_vectors = None


def connect_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_faqs():
    with connect_db() as conn:
        rows = conn.execute(
            "SELECT id, question, answer FROM faqs "
            "WHERE enabled = 1 ORDER BY id"
        ).fetchall()
    return [dict(row) for row in rows]


def encode(texts):
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def refresh_cache():
    global faq_cache, faq_vectors
    faq_cache = get_faqs()
    faq_vectors = (
        encode([faq["question"] for faq in faq_cache])
        if faq_cache else None
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    with connect_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS faqs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1
            )
        """)
        conn.commit()

    model = SentenceTransformer(MODEL_NAME)
    refresh_cache()
    yield
    model = None


app = FastAPI(
    title="FAQ Semantic Matching API",
    version="1.0.0",
    lifespan=lifespan,
)


class FAQInput(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    answer: str = Field(min_length=1, max_length=10000)
    enabled: bool = True


class MatchInput(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=20)


@app.get("/")
def home():
    return {"name": "FAQ Semantic API", "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/faqs")
def list_faqs():
    return {"items": get_faqs()}


@app.post("/faqs", status_code=201)
def add_faq(data: FAQInput):
    with connect_db() as conn:
        cursor = conn.execute(
            "INSERT INTO faqs (question, answer, enabled) VALUES (?, ?, ?)",
            (data.question.strip(), data.answer.strip(), int(data.enabled)),
        )
        faq_id = cursor.lastrowid
        conn.commit()

    refresh_cache()
    return {"id": faq_id, **data.model_dump()}


@app.post("/match")
def match_question(data: MatchInput):
    if not faq_cache or faq_vectors is None:
        raise HTTPException(
            status_code=409,
            detail="Aucune FAQ active. Ajoutez d'abord des FAQ via POST /faqs.",
        )

    query_vector = encode([data.question])[0]
    scores = faq_vectors @ query_vector
    indices = np.argsort(scores)[::-1][:data.top_k]

    results = []
    for index in indices:
        faq = faq_cache[int(index)]
        score = float(scores[index])
        results.append({
            "faq_id": faq["id"],
            "question": faq["question"],
            "answer": faq["answer"],
            "similarity": round(score, 4),
            "similarity_percent": round(
                max(0.0, min(1.0, score)) * 100, 2
            ),
        })

    best_score = float(scores[indices[0]])

    if best_score >= MATCH_THRESHOLD:
        decision = "matched"
        best_match = results[0]
    elif best_score >= REVIEW_THRESHOLD:
        decision = "needs_review"
        best_match = None
    else:
        decision = "no_match"
        best_match = None

    return {
        "input_question": data.question,
        "decision": decision,
        "best_match": best_match,
        "results": results,
        "note": (
            "Le pourcentage représente une échelle de similarité, "
            "pas une probabilité de réponse correcte."
        ),
    }
