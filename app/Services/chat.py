import os, re
from typing import Optional, Dict, Any
from pydantic import BaseModel

from .mlflow_loader import predict_and_top_features
from .explain import fetch_recent_news_bullets, build_explain_prompt, llm_explain_with_openai

TICKER_RE = re.compile(r"\b([A-Z]{1,5}(?:\.[A-Z]{1,2})?)\b")

class ChatIn(BaseModel):
    message: str
    ticker: Optional[str] = None  # 사용자가 직접 지정하면 우선

def _guess_ticker(text: str) -> Optional[str]:
    # 간단 추출: 대문자 1~5자리(예: AAPL, MSFT, GOOG, BRK.B)
    m = TICKER_RE.search(text.upper())
    if not m: 
        return None
    return m.group(1)

def handle_chat(body: ChatIn, model_uri: str) -> Dict[str, Any]:
    # 1) 티커 결정
    ticker = body.ticker or _guess_ticker(body.message)
    if not ticker:
        return {"answer": "질문에 종목 티커가 없습니다. 예: 'AAPL이 왜 이렇게 예측돼?'", "ok": False}

    # 2) 예측/중요도
    pred_pct, top_feats = predict_and_top_features(ticker, model_uri)

    # 3) 뉴스 수집
    news_bullets = fetch_recent_news_bullets(f"{ticker} stock latest news", max_items=5)

    # 4) LLM 설명
    prompt = build_explain_prompt(
        ticker=ticker,
        pred_pct=pred_pct,
        top_features=top_feats,
        news_bullets=news_bullets
    )
    explanation = llm_explain_with_openai(prompt)

    return {
        "ok": True,
        "ticker": ticker,
        "pred_pct": pred_pct,
        "pred_pct_percent": round(pred_pct*100, 2),
        "top_features": top_feats,
        "news_count": len(news_bullets),
        "explanation": explanation
    }
