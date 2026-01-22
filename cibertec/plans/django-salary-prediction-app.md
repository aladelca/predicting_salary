# Salary Prediction Django App Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use core-executing-plans to implement this plan task-by-task.

**Goal:** Build a Django app that predicts salary from job description, skills description, and title using the same data transformations and model flow from `notebooks/machine_learning_model.ipynb`.

**Architecture:** A small ML module handles preprocessing, vectorization, and model training using the notebook steps (clean → tokenize → stopwords → lemmatize → TF‑IDF). A management command trains and saves the vectorizer/model artifacts. A Django view + form loads artifacts at runtime to serve predictions via a simple HTML form (optionally JSON). Artifacts and data paths are configured in settings for repeatable training and inference.

**Reasoning:** Reusing the exact preprocessing flow avoids training/serving skew. Separating training (command) and inference (predictor module + view) keeps runtime fast and makes artifacts reproducible. Settings-based paths keep the data file and model artifacts discoverable without hardcoding.

**Tech Stack:** Django 5.x, pandas, numpy, nltk, scikit-learn (TF‑IDF), catboost, joblib.

---

### Task 1: Project configuration and dependencies

**Files:**
- Create: `requirements.txt`
- Modify: `cibertec/settings.py`
- Modify: `salary_predictor/apps.py`

**Step 1: Write the failing test**
- Add a Django settings test that expects `SALARY_DATA_PATH` and `SALARY_MODEL_DIR` to be configured.

**Step 2: Run test to verify it fails**
- `python manage.py test salary_predictor`

**Step 3: Write minimal implementation**
- Add dependencies to `requirements.txt` (Django, pandas, numpy, nltk, scikit-learn, catboost, joblib).
- Add `salary_predictor` to `INSTALLED_APPS`.
- Add `SALARY_DATA_PATH` (default `BASE_DIR.parent / 'data' / 'postings.csv'`).
- Add `SALARY_MODEL_DIR` (default `BASE_DIR / 'salary_predictor' / 'artifacts'`).

**Step 4: Run test to verify it passes**
- `python manage.py test salary_predictor`

**Step 5: Commit (only after approval)**

---

### Task 2: Implement preprocessing pipeline (notebook parity)

**Files:**
- Create: `salary_predictor/ml/preprocess.py`
- Modify: `salary_predictor/tests.py`

**Step 1: Write the failing test**
- Add tests that assert the preprocessing steps match the notebook:
  - Null handling for `description`, `skills_desc`, `title`.
  - `total_text` concatenation in the same order.
  - `clean_text` lowercases and removes non‑alphanumeric chars (regex `[^a-z0-9\s]`).
  - Tokenization → stopword removal → lemmatization → join tokens.

**Step 2: Run test to verify it fails**
- `python manage.py test salary_predictor`

**Step 3: Write minimal implementation**
- Implement functions mirroring the notebook exactly:
  - `clean_text(text)`
  - `tokenize(text)` (nltk `word_tokenize`)
  - `remove_stopwords(tokens)` (english stopwords)
  - `lemmatize(tokens)` (WordNetLemmatizer)
  - `tokens_to_text(tokens)`
  - `build_total_text(df)` for concatenation with null fill values: `"no description"`, `"no skills desc"`, `"no title"`.

**Step 4: Run test to verify it passes**
- `python manage.py test salary_predictor`

**Step 5: Commit (only after approval)**

---

### Task 3: Training pipeline and artifact persistence

**Files:**
- Create: `salary_predictor/ml/train.py`
- Create: `salary_predictor/management/commands/train_salary_model.py`
- Modify: `salary_predictor/tests.py`

**Step 1: Write the failing test**
- Add tests for training pipeline that:
  - Filters out rows with null `normalized_salary`.
  - Applies preprocessing to build `total_text`.
  - Produces a fitted `TfidfVectorizer(max_features=500)`.
  - Trains a `CatBoostRegressor(iterations=3000, learning_rate=0.01, depth=6)` (use a small override in tests).
  - Saves `vectorizer.joblib` and `catboost_model.cbm` to `SALARY_MODEL_DIR`.

**Step 2: Run test to verify it fails**
- `python manage.py test salary_predictor`

**Step 3: Write minimal implementation**
- `train.py` should expose `train_model(data_path, artifacts_dir, *, test_size=0.2, random_state=42)`.
- Mirror notebook steps: clean → tokenize → stopwords → lemmatize → join → TF‑IDF.
- Persist artifacts with deterministic names.
- Add the management command to call `train_model` and print metrics (MAE, MAPE).

**Step 4: Run test to verify it passes**
- `python manage.py test salary_predictor`

**Step 5: Commit (only after approval)**

---

### Task 4: Predictor module for runtime inference

**Files:**
- Create: `salary_predictor/ml/predictor.py`
- Modify: `salary_predictor/tests.py`

**Step 1: Write the failing test**
- Add tests that:
  - Load artifacts once (cache) from `SALARY_MODEL_DIR`.
  - Accept three text inputs and return a float prediction.
  - Use the same preprocessing + vectorizer used in training.

**Step 2: Run test to verify it fails**
- `python manage.py test salary_predictor`

**Step 3: Write minimal implementation**
- Implement `predict_salary(description, skills_desc, title)`:
  - Build `total_text` with null handling.
  - Run preprocessing and vectorization.
  - Run `CatBoostRegressor.predict` and return scalar.

**Step 4: Run test to verify it passes**
- `python manage.py test salary_predictor`

**Step 5: Commit (only after approval)**

---

### Task 5: Django form, view, and routing

**Files:**
- Create: `salary_predictor/forms.py`
- Modify: `salary_predictor/views.py`
- Create: `salary_predictor/urls.py`
- Modify: `cibertec/urls.py`
- Create: `salary_predictor/templates/salary_predictor/predict.html`
- Modify: `salary_predictor/tests.py`

**Step 1: Write the failing test**
- Add view tests that:
  - GET renders the prediction form.
  - POST with valid inputs returns a prediction in context.

**Step 2: Run test to verify it fails**
- `python manage.py test salary_predictor`

**Step 3: Write minimal implementation**
- Build a Django `Form` with three text fields.
- Create a view that handles GET/POST and calls `predict_salary`.
- Add URLs for `/predict/` and include app routes in project `urls.py`.
- Create a simple template with the form and a prediction result section.

**Step 4: Run test to verify it passes**
- `python manage.py test salary_predictor`

**Step 5: Commit (only after approval)**

---

### Task 6: Developer docs and runbook

**Files:**
- Create: `README.md`

**Step 1: Write the failing test**
- Add a documentation test placeholder (or a checklist in README) for manual verification.

**Step 2: Run test to verify it fails**
- `python manage.py test salary_predictor`

**Step 3: Write minimal implementation**
- Document:
  - How to install dependencies.
  - How to train: `python manage.py train_salary_model`.
  - Where artifacts are saved.
  - How to run server and use `/predict/`.

**Step 4: Run test to verify it passes**
- `python manage.py test salary_predictor`

**Step 5: Commit (only after approval)**

---

## Verification

- Unit tests: `python manage.py test salary_predictor`
- Manual check:
  - Train artifacts: `python manage.py train_salary_model` (expects `vectorizer.joblib` and `catboost_model.cbm` in `salary_predictor/artifacts/`).
  - Run server: `python manage.py runserver` and verify `/predict/` returns a numeric prediction.

## Notes on Notebook Parity (must match exactly)

- Filter: keep only rows where `normalized_salary` is not null.
- Null fill defaults:
  - `description` → `"no description"`
  - `skills_desc` → `"no skills desc"`
  - `title` → `"no title"`
- Concatenate: `description + " " + skills_desc + " " + title` into `total_text`.
- Cleaning: lowercase and `re.sub(r'[^a-z0-9\s]', '', text)` then trim spaces.
- Tokenization: `nltk.word_tokenize`.
- Stopwords: NLTK English stopwords.
- Lemmatization: `WordNetLemmatizer`.
- Vectorizer: `TfidfVectorizer(max_features=500)`.
- Model: `CatBoostRegressor(iterations=3000, learning_rate=0.01, depth=6)`.
