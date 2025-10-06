import os
import datetime
import json
from typing import List, Dict
import requests

from langchain_openai import ChatOpenAI

# -------- 뉴스 수집 (Serper) --------
def fetch_recent_news_bullets(query: str, max_items: int = 5) -> List[str]:
    """
    Serper(구글 검색 대행)로 최근 관련 뉴스 타이틀/링크 몇 개 가져오기.
    실패 시 빈 리스트 반환.
    """
    api_key = os.getenv("SERPER_API_KEY", "")
    if not api_key:
        return []
    try:
        resp = requests.post(
            "https://google.serper.dev/news",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": max_items, "gl": "us"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("news", [])[:max_items]
        bullets = []
        for it in items:
            title = it.get("title", "")
            source = it.get("source", "")
            link = it.get("link", "")
            date = it.get("date", "")
            bullets.append(f"- {title} ({source}, {date})\n  {link}")
        return bullets
    except Exception:
        return []

# -------- 프롬프트 작성 --------
def build_explain_prompt(
    ticker: str,
    pred_pct: float,
    top_features: List[Dict[str, float]],
    news_bullets: List[str],
) -> str:
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    tf = "\n".join([f"- {f['feature']}: {round(float(f['importance']), 4)}" for f in top_features]) if top_features else "- (중요도 없음)"
    nb = "\n".join(news_bullets) if news_bullets else "- (관련 기사 부족)"

    return f"""
[예측 요약]
- 종목: {ticker}
- 대상일: {tomorrow}
- 예측 변동률(%): {round(pred_pct * 100, 2)}

[모델 피처 중요도 Top-K]
{tf}

[관련 기사 목록(최근)]
{nb}

[지시]
1) 위 정보를 근거로 예측(상승/하락)의 타당성을 3~5개 포인트로 설명하라.
2) "정합성"을 높음/보통/낮음 중 하나로 평가하라.
3) 마지막에 위험요인(리스크) 1~2개를 제시하라.
한국어로 간결하게.
"""

# -------- LLM 호출(설명 생성) --------
def llm_explain_with_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    llm = ChatOpenAI(model=model, temperature=0)
    out = llm.invoke(prompt)
    return out.content.strip()

# -------- XGB 중요도 로딩 (모델에서 바로 뽑기) --------
def get_feature_importance_from_model(model) -> List[Dict[str, float]]:
    """
    XGBRegressor(Sklearn) 로드 후 feature_importances_에서 상위 K개 반환.
    """
    import numpy as np
    import pandas as pd

    names = ["ret_1","ret_3","ret_5","vol_chg_3","dow"]
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return []
        s = (
            pd.Series(importances, index=names)
            .sort_values(ascending=False)
            .head(5)
        )
        return [{"feature": k, "importance": float(v)} for k, v in s.to_dict().items()]
    except Exception:
        return []
