import os
import mlflow.sklearn
import pandas as pd
from utils.io_utils import price_path
from ml.features import make_features
from .explain import get_feature_importance_from_model

def load_model(model_uri: str):
    mlflow = __import__("mlflow")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    return mlflow.sklearn.load_model(model_uri)

def predict_and_top_features(ticker: str, model_uri: str):
    model = load_model(model_uri)
    # 최신 피처 한 행 구성
    df = pd.read_parquet(price_path(ticker))
    feats = make_features(df, horizon=1)
    X = feats[["ret_1","ret_3","ret_5","vol_chg_3","dow"]]
    yhat = float(model.predict(X.iloc[[-1]])[0])
    top_feats = get_feature_importance_from_model(model)
    return yhat, top_feats
