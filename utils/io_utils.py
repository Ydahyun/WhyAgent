from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
(DATA / "prices").mkdir(parents=True, exist_ok=True)
def price_path(ticker: str) -> str:
    return str(DATA / "prices" / f"{ticker}.parquet")
