import os, pandas as pd, mlflow.sklearn
from ml.features import make_features
from utils.io_utils import price_path

def predict_next_pct(ticker: str, model_uri: str) -> float:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","sqlite:///mlflow.db"))
    df = pd.read_parquet(price_path(ticker))
    if "datetime" in df.columns: df = df.rename(columns={"datetime":"date"})
    feats = make_features(df, 1)
    X = feats[["ret_1","ret_3","ret_5","vol_chg_3","dow"]]
    model = mlflow.sklearn.load_model(model_uri)
    return float(model.predict(X.iloc[[-1]])[0])
