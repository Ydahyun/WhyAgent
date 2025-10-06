import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
import yfinance as yf
import yaml, pandas as pd
from pathlib import Path
from utils.io_utils import price_path

# Stooq fallback
try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None

CFG = Path(__file__).resolve().parents[1] / "configs" / "tickers.yml"

def try_yf(ticker, period, interval):
    """야후 1차 시도 (세션 없이)"""
    return yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

def try_yf_daily(ticker):
    """intraday 실패 시 일봉 3개월 재시도"""
    return yf.download(
        tickers=ticker,
        period="3mo",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

def try_stooq(ticker):
    if pdr is None:
        return pd.DataFrame()
    # Stooq는 미국 종목을 종종 'META.US' 형태로 요구
    candidates = [ticker]
    if not ticker.endswith(".US"):
        candidates.append(f"{ticker}.US")
    for sym in candidates:
        try:
            df = pdr.DataReader(sym, data_source="stooq")  # 일봉
            if not df.empty:
                df = df.sort_index()
                print(f"[stooq] hit: {sym}")
                return df
        except Exception as e:
            print(f"[stooq] miss: {sym} -> {e}")
    return pd.DataFrame()


def run():
    cfg = yaml.safe_load(open(CFG, "r", encoding="utf-8"))
    tickers = cfg["tickers"]
    period = cfg.get("price_period", "3mo")     # 일단 보수적으로
    interval = cfg.get("price_interval", "1d")  # 일봉 먼저 수집

    for t in tickers:
        saved = False
        # 1) 야후 1차
        try:
            df = try_yf(t, period, interval)
            if not df.empty:
                df = df.reset_index().rename(columns=str.lower)
                df["ticker"] = t
                df.to_parquet(price_path(t), index=False)
                print(f"[saved:yf] {t}: rows={len(df)}")
                saved = True
            else:
                print(f"[warn] yf empty: {t} (period={period}, interval={interval})")
        except Exception as e:
            print(f"[warn] yf error: {t} -> {e}")

        if saved:
            continue

        # 2) 야후 일봉 재시도
        try:
            df2 = try_yf_daily(t)
            if not df2.empty:
                df2 = df2.reset_index().rename(columns=str.lower)
                df2["ticker"] = t
                df2.to_parquet(price_path(t), index=False)
                print(f"[saved:yf-daily] {t}: rows={len(df2)}")
                saved = True
            else:
                print(f"[warn] yf-daily empty: {t}")
        except Exception as e:
            print(f"[warn] yf-daily error: {t} -> {e}")

        if saved:
            continue

        # 3) Stooq 폴백
        df3 = try_stooq(t)
        if not df3.empty:
            # ✅ 정규화(인덱스→컬럼, 소문자, ticker 추가)
            df3 = df3.reset_index().rename(columns=str.lower)
            df3["ticker"] = t
            df3.to_parquet(price_path(t), index=False)
            print(f"[saved:stooq] {t}: rows={len(df3)}")
            saved = True
        else:
            print(f"[skip] {t}: all sources empty")


if __name__ == "__main__":
    run()
