# Job Posting Salary Prediction

Django app to predict salary from job description, skills description, and title. The model pipeline mirrors `notebooks/machine_learning_model.ipynb` (cleaning, tokenization, stopwords, lemmatization, TF-IDF, CatBoost). Predictions are persisted in SQLite and displayed on the home page.

## Features
- Home page with prediction history stored in the database
- Prediction form at `/predict/`
- Training command that produces model artifacts (vectorizer + model)
- Minimal black/white/gray UI

## Project Structure
- `cibertec/`: Django project
  - `salary_predictor/`: app with ML pipeline, views, templates
- `data/postings.csv`: training dataset (normalized_salary is target)
- `notebooks/machine_learning_model.ipynb`: reference notebook

## Setup
From the repo root:

```bash
cd cibertec
python -m venv ../.venv
source ../.venv/bin/activate
pip install -r requirements.txt
```

## Migrations
```bash
cd cibertec
python manage.py migrate
```

## Train the model
This writes artifacts to `cibertec/salary_predictor/artifacts/` by default.

```bash
cd cibertec
python manage.py train_salary_model
```

If you see a message about missing NLTK resources, install them once:

```bash
python -m nltk.downloader stopwords wordnet punkt
```

## Run the app
```bash
cd cibertec
python manage.py runserver
```

Open:
- `http://127.0.0.1:8000/` (home with prediction history)
- `http://127.0.0.1:8000/predict/` (prediction form)

## Notes
- Settings live in `cibertec/cibertec/settings.py`:
  - `SALARY_DATA_PATH` defaults to `data/postings.csv`
  - `SALARY_MODEL_DIR` defaults to `cibertec/salary_predictor/artifacts`
- If the app shows "Model artifacts not found", run `python manage.py train_salary_model`.

## Tests
```bash
cd cibertec
python manage.py test salary_predictor
```
