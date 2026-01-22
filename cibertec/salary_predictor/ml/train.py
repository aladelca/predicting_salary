from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

from salary_predictor.ml.preprocess import (
    build_total_text,
    ensure_nltk_data,
    preprocess_series,
)

VECTORIZER_FILENAME = "vectorizer.joblib"
MODEL_FILENAME = "catboost_model.cbm"


def filter_salary_rows(df):
    return df[df["normalized_salary"].isna() == False].copy()


def train_model(
    data_path,
    artifacts_dir,
    *,
    test_size=0.2,
    random_state=42,
    model_params=None,
):
    data_path = Path(data_path)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df = filter_salary_rows(df)
    df = build_total_text(df)

    ensure_nltk_data()

    X = df["total_text"]
    y = df["normalized_salary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train_final = preprocess_series(X_train)
    X_test_final = preprocess_series(X_test)

    vectorizer = TfidfVectorizer(max_features=500)
    X_train_vectorized = vectorizer.fit_transform(X_train_final)
    X_test_vectorized = vectorizer.transform(X_test_final)

    params = {
        "iterations": 3000,
        "learning_rate": 0.01,
        "depth": 6,
        "verbose": 100,
    }
    if model_params:
        params.update(model_params)

    model = CatBoostRegressor(**params)
    model.fit(X_train_vectorized, y_train)

    preds = model.predict(X_test_vectorized)
    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "mape": mean_absolute_percentage_error(y_test, preds),
    }

    vectorizer_path = artifacts_dir / VECTORIZER_FILENAME
    model_path = artifacts_dir / MODEL_FILENAME
    joblib.dump(vectorizer, vectorizer_path)
    model.save_model(model_path)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "metrics": metrics,
        "artifacts": {
            "vectorizer": vectorizer_path,
            "model": model_path,
        },
    }
