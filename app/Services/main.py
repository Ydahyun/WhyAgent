# app/Services/main.py
import os
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from ml.predict_latest import predict_next_pct
from .mlflow_loader import predict_and_top_features
from .explain import (
    fetch_recent_news_bullets,
    build_explain_prompt,
    llm_explain_with_openai,
)
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from .chat import ChatIn, handle_chat

# --- env ---
load_dotenv()
MODEL_URI = os.getenv("MODEL_URI", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

app = FastAPI(title="WhyAgent API")

# --- Schemas ---
class PredictIn(BaseModel):
    ticker: str

class ExplainIn(BaseModel):
    ticker: str
    pred_pct: Optional[float] = None      # 제공되면 그 값으로 설명만 생성
    top_k: int = 5
    news_query: Optional[str] = None      # 예: "Apple earnings 2025"


# --- Routes ---
@app.get("/")
def root():
    # 필요하면 아래 한 줄로 문서로 리다이렉트도 가능: return RedirectResponse("/docs")
    return {"service": "WhyAgent API", "endpoints": ["/health", "/predict", "/explain", "/docs"]}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(body: PredictIn):
    if not MODEL_URI:
        raise HTTPException(status_code=500, detail="MODEL_URI is not set")
    yhat = predict_next_pct(body.ticker, MODEL_URI)
    return {"ticker": body.ticker, "pred_pct": yhat}

@app.post("/explain")
def explain(body: ExplainIn):
    # 사전 체크
    if not MODEL_URI:
        raise HTTPException(status_code=500, detail="MODEL_URI is not set")
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")
    # 1) 예측/중요도
    if body.pred_pct is None:
        pred_pct, top_feats = predict_and_top_features(body.ticker, MODEL_URI)
    else:
        pred_pct = body.pred_pct
        _, top_feats = predict_and_top_features(body.ticker, MODEL_URI)

    # 2) 뉴스 수집 (Serper 키 없으면 빈 리스트 허용)
    query = (body.news_query or f"{body.ticker} stock latest news").strip()
    news_bullets = fetch_recent_news_bullets(query, max_items=5)

    # 3) LLM 설명
    prompt = build_explain_prompt(
        ticker=body.ticker,
        pred_pct=pred_pct,
        top_features=top_feats,
        news_bullets=news_bullets,
    )
    explanation = llm_explain_with_openai(prompt)

    return {
        "ticker": body.ticker,
        "pred_pct": pred_pct,
        "pred_pct_percent": round(pred_pct * 100, 2),
        "top_features": top_feats,
        "news_count": len(news_bullets),
        "explanation": explanation,
    }
# 정적 파일(간단 프런트엔드)
app.mount("/web", StaticFiles(directory="app/web", html=True), name="web")
@app.post("/api/chat")
def chat_api(body: ChatIn):
    if not MODEL_URI:
        raise HTTPException(status_code=500, detail="MODEL_URI is not set")
    return handle_chat(body, MODEL_URI)
