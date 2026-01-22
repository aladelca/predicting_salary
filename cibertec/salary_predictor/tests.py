from django.conf import settings
from django.test import SimpleTestCase

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import nltk

from django.test import SimpleTestCase, TestCase, override_settings

from salary_predictor.ml import preprocess
from salary_predictor.ml import train as train_module
from salary_predictor.ml import predictor as predictor_module
from salary_predictor.models import Prediction


class SettingsConfigurationTests(SimpleTestCase):
    def test_salary_paths_configured(self):
        self.assertTrue(hasattr(settings, "SALARY_DATA_PATH"))
        self.assertTrue(hasattr(settings, "SALARY_MODEL_DIR"))


class PreprocessPipelineTests(SimpleTestCase):
    def test_build_total_text_handles_nulls_and_concat(self):
        df = pd.DataFrame(
            {
                "description": [np.nan, "Job desc"],
                "skills_desc": ["Skills", np.nan],
                "title": [np.nan, "Engineer"],
            }
        )
        result = preprocess.build_total_text(df)
        self.assertEqual(
            result.loc[0, "total_text"],
            "no description Skills no title",
        )
        self.assertEqual(
            result.loc[1, "total_text"],
            "Job desc no skills desc Engineer",
        )

    def test_clean_text_matches_notebook_regex(self):
        cleaned = preprocess.clean_text("Hello, WORLD!! 123  \n")
        self.assertEqual(cleaned, "hello world 123")

    def test_full_token_pipeline(self):
        required = [
            "tokenizers/punkt",
            "corpora/stopwords",
            "corpora/wordnet",
        ]
        for resource in required:
            try:
                nltk.data.find(resource)
            except LookupError:
                self.skipTest(f"Missing NLTK resource: {resource}")

        text = preprocess.clean_text("Cats and DOGS.")
        tokens = preprocess.tokenize(text)
        no_stop = preprocess.remove_stopwords(tokens)
        lemmatized = preprocess.lemmatize_tokens(no_stop)
        final_text = preprocess.tokens_to_text(lemmatized)
        self.assertEqual(final_text, "cat dog")


class TrainingPipelineTests(SimpleTestCase):
    def test_filter_salary_rows_drops_nulls(self):
        df = pd.DataFrame(
            {
                "description": ["a", "b"],
                "skills_desc": ["c", "d"],
                "title": ["e", "f"],
                "normalized_salary": [1000, np.nan],
            }
        )
        filtered = train_module.filter_salary_rows(df)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["normalized_salary"], 1000)

    def test_train_model_saves_artifacts(self):
        try:
            import catboost  # noqa: F401
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("Missing ML dependencies for training")

        required = [
            "tokenizers/punkt",
            "corpora/stopwords",
            "corpora/wordnet",
        ]
        for resource in required:
            try:
                nltk.data.find(resource)
            except LookupError:
                self.skipTest(f"Missing NLTK resource: {resource}")

        df = pd.DataFrame(
            {
                "description": ["Data analyst", "Engineer", "Manager"],
                "skills_desc": ["SQL", "Python", "Leadership"],
                "title": ["Analyst", "Software Engineer", "Manager"],
                "normalized_salary": [3000, 4000, 5000],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.csv"
            df.to_csv(data_path, index=False)

            result = train_module.train_model(
                data_path,
                tmpdir,
                model_params={"iterations": 5, "learning_rate": 0.1, "depth": 2},
            )

            self.assertTrue((Path(tmpdir) / "vectorizer.joblib").exists())
            self.assertTrue((Path(tmpdir) / "catboost_model.cbm").exists())
            self.assertEqual(result["vectorizer"].max_features, 500)


class PredictorInferenceTests(SimpleTestCase):
    def test_predict_salary_uses_cached_artifacts(self):
        try:
            import catboost  # noqa: F401
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("Missing ML dependencies for inference")

        required = [
            "tokenizers/punkt",
            "corpora/stopwords",
            "corpora/wordnet",
        ]
        for resource in required:
            try:
                nltk.data.find(resource)
            except LookupError:
                self.skipTest(f"Missing NLTK resource: {resource}")

        df = pd.DataFrame(
            {
                "description": ["Data analyst", "Engineer", "Manager"],
                "skills_desc": ["SQL", "Python", "Leadership"],
                "title": ["Analyst", "Software Engineer", "Manager"],
                "normalized_salary": [3000, 4000, 5000],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.csv"
            df.to_csv(data_path, index=False)
            train_module.train_model(
                data_path,
                tmpdir,
                model_params={"iterations": 5, "learning_rate": 0.1, "depth": 2},
            )

            with override_settings(SALARY_MODEL_DIR=tmpdir):
                predictor_module.get_artifacts.cache_clear()
                vec1, model1 = predictor_module.get_artifacts()
                vec2, model2 = predictor_module.get_artifacts()

                self.assertIs(vec1, vec2)
                self.assertIs(model1, model2)

                prediction = predictor_module.predict_salary(
                    "Data analyst", "SQL", "Analyst"
                )
                self.assertIsInstance(prediction, float)


class PredictViewTests(TestCase):
    def test_get_renders_form(self):
        response = self.client.get("/predict/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Predict Salary")

    def test_post_returns_prediction(self):
        with patch("salary_predictor.views.predict_salary", return_value=123.45):
            response = self.client.post(
                "/predict/",
                data={
                    "description": "Data analyst",
                    "skills_desc": "SQL",
                    "title": "Analyst",
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertIsNotNone(response.context.get("prediction"))
            self.assertEqual(Prediction.objects.count(), 1)
            prediction = Prediction.objects.first()
            self.assertEqual(prediction.predicted_salary, 123.45)


class HomeViewTests(TestCase):
    def test_home_lists_predictions(self):
        Prediction.objects.create(
            description="Desc one",
            skills_desc="Skills one",
            title="Title one",
            predicted_salary=1000.0,
        )
        Prediction.objects.create(
            description="Desc two",
            skills_desc="Skills two",
            title="Title two",
            predicted_salary=2000.0,
        )
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Previous Predictions")
        self.assertContains(response, "Title one")
        self.assertContains(response, "Title two")
