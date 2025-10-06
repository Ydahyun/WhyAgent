import os, pandas as pd, mlflow, mlflow.sklearn
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ml.features import make_features
from utils.io_utils import price_path

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","sqlite:///mlflow.db"))

def train_one(ticker: str):
    df = pd.read_parquet(price_path(ticker))
    # 열이 어떻게 오든 make_features에서 정규화 처리
    feats = make_features(df, 1)

    X = feats[["ret_1","ret_3","ret_5","vol_chg_3","dow"]]
    y = feats["y"]

    split = int(len(X) * 0.8)
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]

    params = dict(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    with mlflow.start_run(run_name=f"xgb_{ticker}"):
        model = XGBRegressor(**params).fit(Xtr, ytr)
        pred = model.predict(Xte)
        mlflow.log_params(params)
        mlflow.log_metric("mae", float(mean_absolute_error(yte, pred)))
        mlflow.log_metric("rmse", float(mean_squared_error(yte, pred, squared=False)))
        mlflow.sklearn.log_model(model, "model")

    print(f"[trained] {ticker}")

# ml/train.py (파일 하단)
import yaml
from pathlib import Path

if __name__ == "__main__":
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "tickers.yml"
    tickers = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))["tickers"]
    for t in tickers:
        train_one(t)
