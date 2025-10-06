import re
import pandas as pd

def _flatten_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) 멀티인덱스 → 1레벨로 평탄화
    2) 문자열로 굳은 멀티인덱스(예: "('close','aapl')")도 파싱해서 첫 토큰만 추출
    결과: open/high/low/close/adj close/volume/ticker/date 같은 표준 키로 맞춤
    """
    # 1) MultiIndex면 첫 레벨만 사용
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(c[0]).lower() for c in df.columns]

    # 2) 문자열로 굳은 "(...,...)" 형태 파싱
    new_cols = []
    for c in df.columns:
        s = str(c).strip().lower()
        if s.startswith("(") and s.endswith(")") and "," in s:
            # "('close', 'aapl')" -> 'close'
            inner = s[1:-1]
            first = inner.split(",")[0]
            first = first.replace("'", "").replace('"', "").strip()
            new_cols.append(first)
        else:
            new_cols.append(s)
    df = df.copy()
    df.columns = new_cols
    return df

def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.lower() for c in df.columns})
    cols = set(df.columns)

    if "date" in cols:
        df["date"] = pd.to_datetime(df["date"])
        return df
    if "datetime" in cols:
        df = df.rename(columns={"datetime": "date"})
        df["date"] = pd.to_datetime(df["date"])
        return df
    if "index" in cols:
        df = df.rename(columns={"index": "date"})
        df["date"] = pd.to_datetime(df["date"])
        return df
    if df.index.name is not None or not isinstance(df.index, pd.RangeIndex):
        df = df.copy()
        df["date"] = pd.to_datetime(df.index)
        df = df.reset_index(drop=True)
        return df
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df

def _standardize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    가격 컬럼 표준화:
    - close가 없고 adj close만 있으면 close=adj close 생성
    - open/high/low 비어 있으면 close로 채워 최소 실행 보장
    - volume 없으면 0으로 채움
    """
    df = df.rename(columns={c: c.lower() for c in df.columns})
    cols = set(df.columns)

    # close 확보
    if "close" not in cols:
        if "adj close" in cols:
            df["close"] = df["adj close"]
        else:
            raise KeyError(f"'close' 컬럼을 찾을 수 없습니다. 현재 컬럼: {list(df.columns)}")

    for c in ["open", "high", "low"]:
        if c not in df.columns:
            df[c] = df["close"]

    if "volume" not in df.columns:
        df["volume"] = 0

    return df

def make_features(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    df: columns [date|datetime|index, open, high, low, close|adj close, volume, ticker]
    horizon: t+horizon 예측
    """
    # 0) 컬럼 평탄화 (멀티인덱스/튜플문자열 처리)
    df = _flatten_price_columns(df)

    # 1) 날짜 정규화
    df = _ensure_date_column(df)

    # 2) 가격 컬럼 표준화
    df = _standardize_price_columns(df)

    # 3) 정렬
    df = df.sort_values("date")

    # 4) 기본 피처
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_5"] = df["close"].pct_change(5)
    df["vol_chg_3"] = df["volume"].pct_change(3)

    # 5) 요일 더미
    df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek

    # 6) 라벨: 다음 종가 수익률
    df["y"] = df["close"].shift(-horizon) / df["close"] - 1.0

    out = df.dropna().reset_index(drop=True)
    if len(out) < 30:
        raise ValueError(f"학습 데이터가 너무 적습니다. 행 수={len(out)}")
    return out
