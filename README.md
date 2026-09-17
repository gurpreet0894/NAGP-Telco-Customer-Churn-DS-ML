# Telco Customer Churn — Prediction Project

End-to-end machine learning solution to predict customer churn for a telecom company.

WorkFlow:
business problem → data preparation → EDA →
feature engineering → Decision Tree model → evaluation → interpretation → saved model →
REST API.

## Assumptions

- **Environment:** Python 3.12+ is required (see Setup) — the pinned dependency versions
  in `requirements.txt` do not install/run correctly on Python 3.10/3.11.
- **Reproducibility:** `random_state=42` is fixed everywhere (train/test split, model
  training, cross-validation), as explicitly required by the assignment.
- **Train/test split:** 70:30, stratified on `Churn`, per the assignment's explicit
  instruction — not tuned or chosen empirically.
- **Deliverable model:** the assignment requires a Decision Tree Classifier as the final
  model. Random Forest and Logistic Regression are included only as bonus context/
  comparison and were never candidates for the "final model" slot.
- **Data quality:** the 11 blank `TotalCharges` values are assumed to be brand-new
  customers who haven't been billed yet (all have `tenure == 0`), so they're filled with
  `0` rather than statistically imputed.
- **`customerID` is assumed non-predictive** (a unique identifier) and is dropped
  entirely rather than encoded.
- **Categorical variables are assumed nominal** (no inherent order — e.g.
  `PaymentMethod`, `InternetService`), so one-hot encoding is used throughout rather than
  ordinal/label encoding.
- **No feature scaling or outlier removal** — assumed unnecessary because Decision Trees
  split on value thresholds/order, not distance or magnitude (see
  `Analysis_Documentation.docx` for the full reasoning).
- **Business framing:** the model's output is assumed to feed a *human-reviewed*
  retention action (a call or offer), not an automated decision — this assumption is
  what justifies prioritizing recall over precision (see notebook Section 5).
- **API scope:** `POST /predict` is assumed to serve **one customer per request** (not
  batch), matching the assignment's example request/response format.
- **Unseen categories:** a new customer sent to the API with a category value not seen
  during training (e.g. a future new `PaymentMethod`) is assumed safe to encode as
  all-zeros (`handle_unknown='ignore'`) rather than rejected outright.

## Project structure

```
DS-Assignment/
├── data/
│   ├── Data Science Assignment.pdf            # assignment brief
│   ├── TelcoCustomerChurn.csv                  # dataset
│   └── TelcoCustomerChurn - Data Dictionary.csv
├── notebook/
│   └── churn_analysis.ipynb                    # full analysis & modelling notebook
├── model/
│   └── churn_model.pkl                         # saved final pipeline (created by the notebook)
├── preprocessing.py                            # shared cleaning/feature-engineering/pipeline code
├── app.py                                      # Flask REST API (POST /predict) + basic web UI (GET /)
├── templates/
│   └── index.html                              # basic HTML/JS form UI for /predict
├── requirements.txt
├── sample_request.json
└── README.md
```

`preprocessing.py` is imported by **both** the notebook and `app.py`, so the exact same
cleaning, feature engineering, and encoding logic that the model was trained with is
applied to any new customer sent to the API — no separate, easy-to-forget preprocessing
step.

## Setup

1. **Python 3.12 or newer is required** — the pinned package versions in
   `requirements.txt` do not install/run correctly on Python 3.10 or older. Check your
   version first:

   ```bash
   python --version
   ```

   Then create and activate a virtual environment using that Python 3.12+ install:

   ```bash
   python -m venv venv
   # Windows (Command Prompt)
   venv\Scripts\activate
   # Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   # Windows (Git Bash)
   source venv/Scripts/activate
   # macOS/Linux
   source venv/bin/activate
   ```

   Confirm activation worked before continuing — `where python` (Windows) or
   `which python` (macOS/Linux/Git Bash) should point **inside the `venv` folder**, not a
   system Python install. If your prompt doesn't show a `(venv)` prefix, activation
   didn't take effect.

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the notebook

```bash
jupyter notebook notebook/churn_analysis.ipynb
```

Run all cells top to bottom. The last section of the notebook saves the final trained
pipeline to `model/churn_model.pkl` — **this file must exist before starting the API.**

## Running the API

```bash
python app.py
```

The API starts on `http://127.0.0.1:5000`.

### Web UI

Open `http://127.0.0.1:5000/` in a browser for a basic HTML form — fill in a customer's
attributes and click **Predict churn** to see the prediction and probability rendered on
the page. It's a thin client: the form's JavaScript just calls `POST /predict` (below)
under the hood, so it always reflects the exact same model and validation as the API.

### Endpoints

- `GET /` — basic web UI (HTML form) for trying out predictions in a browser.
- `GET /health` — simple liveness check, returns `{"status": "ok"}`.
- `POST /predict` — accepts a single customer's raw attributes as JSON and returns the
  churn prediction and probability.

### Sample request

Request body (also in `sample_request.json`):

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Mailed check",
  "MonthlyCharges": 53.85,
  "TotalCharges": 108.15
}
```

Example call:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

Sample response:

```json
{
  "prediction": "Yes",
  "churn_probability": 0.6061
}
```

### Error handling

- Missing fields → `400` with an `error` message listing which fields are missing.
- Wrong types (e.g. non-numeric `tenure`) → `400` with a descriptive `error` message.
- Malformed/non-JSON body → `400` with an `error` message.
- Unknown routes / wrong HTTP method → `404` / `405` with an `error` message.

## Model summary

- **Algorithm:** Decision Tree Classifier (`scikit-learn`), selected after comparing a
  shallow baseline tree, a deeper class-weighted tree, and a `GridSearchCV`-tuned tree
  (plus Random Forest / Logistic Regression shown for context only).
- **Final configuration:** `max_depth=6`, `min_samples_leaf=25`, `criterion='gini'`,
  `class_weight='balanced'` — the `GridSearchCV`-tuned tree, chosen because it matches or
  slightly beats the manually-configured alternative on every metric while being simpler
  (shallower), and because it prioritizes **recall** (catching churners), which matters
  more than precision for a proactive retention use case (see the notebook's Section 5
  for the full reasoning).
- **Test-set performance:** Accuracy 0.73, Precision 0.50, Recall 0.79, F1 0.61,
  ROC-AUC 0.83 (see the notebook for the full breakdown and confusion matrix).
- **Feature engineering:** `num_addon_services`, `avg_monthly_charge`, and `tenure_group`
  — see notebook Section 3 for details and rationale.

## MLOps practices already applied

This project is a single-model, not a production deployment, so it
doesn't include heavier MLOps tooling (experiment tracking, CI/CD, containerization,
automated tests, drift monitoring). It does, however, already apply several core MLOps
practices at a scope that fits this project:

- **Reproducibility:** `random_state=42` is fixed across the train/test split, model
  training, and cross-validation, so re-running the notebook always reproduces the same
  model and metrics.
- **Environment pinning:** `requirements.txt` pins exact package versions (`==`), not
  ranges, so the environment the model was trained/tested in can be reproduced exactly
  rather than drifting as libraries update.
- **Train/serve consistency (no training-serving skew):** the cleaning, feature
  engineering, and encoding logic lives once, in `preprocessing.py`, imported by both the
  notebook and `app.py`, wrapped inside a single `sklearn.Pipeline` fit only on the
  training split. New data sent to the API is guaranteed to be transformed identically to
  how the training data was — no separate, easy-to-drift preprocessing copy.
- **Model packaging as one artifact:** `model/churn_model.pkl` saves the *entire* fitted
  pipeline (feature engineering + encoder + classifier) as a single object, not just the
  bare classifier, so the API has nothing extra to reimplement or keep in sync.
- **Artifact validation before trusting it:** the notebook reloads the saved `.pkl` from
  disk and checks its predictions match the in-memory model before considering it usable
  — catching serialization issues before they'd surface in the API.
- **Model serving via a REST API:** `POST /predict` decouples *using* the model from the
  Python/notebook environment it was trained in — any client (the web UI, `curl`, another
  service) can get predictions without needing scikit-learn installed.
- **Documented, auditable model selection:** multiple Decision Tree configurations were
  trained, compared on identical metrics, and the final choice justified in writing
  (notebook Section 4 and `Analysis_Documentation.docx`), rather than picking a model without
  a recorded rationale.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'flask'` (or pandas/sklearn/etc.)** — this means
  the `python` command in your current terminal is not the virtual environment's Python.
  Dependencies were installed *only* inside `venv`, not system-wide. Fix: activate the
  venv in **this terminal** (see Setup step 1) and re-run. To confirm which Python is
  actually being used, run `where python` (Windows) / `which python` (macOS/Linux/Git
  Bash) — the path shown must be inside the project's `venv` folder. If you don't want to
  rely on activation at all, you can always call the venv's Python directly instead:
  `venv\Scripts\python app.py` (Windows) or `venv/bin/python app.py` (macOS/Linux).
- **`FileNotFoundError` / model not found when starting the API** — `model/churn_model.pkl`
  doesn't exist yet. Run the notebook fully (top to bottom) first — its last section
  creates that file.
- **`pip install -r requirements.txt` fails, or install succeeds but imports break** —
  your virtual environment was likely created with Python 3.10 or older. Run
  `python --version` (with the venv activated) to confirm; if it's below 3.12, delete the
  `venv` folder, install Python 3.12+, and recreate the venv with that version (see Setup
  step 1).

## Notes

- Random seed `42` is used throughout for reproducibility (train/test split, model
  training).
- The dataset's known data-quality quirk — 11 customers with blank `TotalCharges` (all
  brand-new, `tenure == 0`) — is handled inside `preprocessing.ChurnFeatureEngineer`, so
  it's fixed consistently for training, testing, and any new customer sent to the API.
