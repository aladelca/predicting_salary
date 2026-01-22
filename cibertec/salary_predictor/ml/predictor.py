from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostRegressor
from django.conf import settings

from salary_predictor.ml.preprocess import (
    build_total_text,
    ensure_nltk_data,
    preprocess_series,
)
from salary_predictor.ml.train import MODEL_FILENAME, VECTORIZER_FILENAME


def _build_single_text(description, skills_desc, title):
    df = pd.DataFrame(
        {
            "description": [description],
            "skills_desc": [skills_desc],
            "title": [title],
        }
    )
    df = build_total_text(df)
    return df.loc[0, "total_text"]


@lru_cache(maxsize=1)
def get_artifacts():
    artifacts_dir = Path(settings.SALARY_MODEL_DIR)
    vectorizer_path = artifacts_dir / VECTORIZER_FILENAME
    model_path = artifacts_dir / MODEL_FILENAME

    if not vectorizer_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run `python manage.py train_salary_model`."
        )

    vectorizer = joblib.load(vectorizer_path)
    model = CatBoostRegressor()
    model.load_model(model_path)

    return vectorizer, model


def predict_salary(description, skills_desc, title):
    ensure_nltk_data()

    total_text = _build_single_text(description, skills_desc, title)
    series = pd.Series([total_text])
    processed = preprocess_series(series)

    vectorizer, model = get_artifacts()
    vectorized = vectorizer.transform(processed)
    prediction = model.predict(vectorized)

    return float(prediction[0])
