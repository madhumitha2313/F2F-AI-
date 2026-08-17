# F2F AI — Failure Intelligence Engine

MLOps project (AD4V71) — predicts whether a proposed Buchwald–Hartwig
cross-coupling reaction is likely to fail (yield < 5%) before it is run,
using Logistic Regression, Decision Tree, Random Forest, and XGBoost,
tracked end-to-end with DVC + MLflow and served via a Flask API.

Madhumitha N (24AD0162) · Manga Haripriya (24AD0164) — AI & Data Science,
Chennai Institute of Technology

## Project structure

```
f2f-ai/
├── data/
│   ├── raw/                  # place Dreher_and_Doyle_input_data.xlsx here
│   └── processed/            # written by src/preprocess.py
├── src/
│   ├── config.py             # shared paths, schema, hyperparameters
│   ├── preprocess.py         # clean data, build failure label, encode features
│   ├── train.py              # train + evaluate 4 models, log to MLflow
│   └── predict.py            # single-reaction inference (CLI + shared logic)
├── models/                   # written by train.py: best_model.pkl, scaler.pkl, ...
├── api/
│   ├── app.py                # Flask REST API + login-gated web UI
│   ├── templates/            # login.html, index.html (chemical-reaction themed UI)
│   ├── static/                # css/, js/ for the web UI
│   ├── requirements.txt
│   └── Dockerfile
├── monitoring/
│   ├── logger.py             # SQLite logging of every prediction request
│   └── drift_detector.py     # chi-square drift check vs. training distribution
├── dvc.yaml                  # DVC pipeline: preprocess -> train
├── params.yaml                # DVC-tracked hyperparameters
└── requirements.txt           # top-level dev environment
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place your dataset at `data/raw/Dreher_and_Doyle_input_data.xlsx`.

## 2. Run the pipeline

Either run the two stages directly:

```bash
python src/preprocess.py
python src/train.py
```

Or, if you have DVC set up (`dvc init` once, then `dvc remote add ...`):

```bash
dvc repro
```

This writes `models/best_model.pkl`, `models/label_encoders.pkl`,
`models/scaler.pkl`, `models/model_config.json`, and a comparison table +
plots for all four models.

## 3. Inspect experiments in MLflow

```bash
mlflow ui
```

Open http://localhost:5000 (or the port MLflow prints) to see every run's
parameters, metrics, and logged model artifact side by side.

## 4. Serve predictions

Locally:

```bash
python api/app.py
```

Then open **http://localhost:5000** for the chemical-reaction-themed web UI
(landing page, pipeline overview, model comparison, and a live "Try It"
form). The UI is login-gated — sign in at `/login` with the demo credentials
below, or override them with environment variables before running:

```bash
export F2F_DEMO_USER=chemist       # default: chemist
export F2F_DEMO_PASS=f2fai2026     # default: f2fai2026
export SECRET_KEY=change-me        # Flask session signing key
```

The JSON API is unchanged and still callable directly (the browser session
cookie authenticates `/predict`, `/stats`, and `/recent-predictions`; only
`/health` stays public for container/monitoring checks):

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -b cookies.txt \
  -d '{"ligand": "L1", "additive": "A3", "base": "B2", "aryl_halide": "AH5"}'
```

Or with Docker, from the project root (so the build context includes `src/` and `models/`):

```bash
docker build -f api/Dockerfile -t f2f-ai-api .
docker run -p 5000:5000 -v $(pwd)/monitoring:/app/monitoring f2f-ai-api
```

## 5. Monitoring and drift

Every `/predict` call is logged to `monitoring/predictions.db`.
`GET /stats` and `GET /recent-predictions` read from that log.

Run the drift check on a schedule (cron, GitHub Action, or a scheduled
Colab cell):

```bash
python monitoring/drift_detector.py
```

This compares the categories seen in live traffic against the training
distribution per feature (chi-square test) and writes
`monitoring/drift_report.json`.

## 6. Retraining

Once drift is flagged or enough new labeled reactions are collected:
append them to `data/raw/`, rerun `dvc repro` (or `preprocess.py` +
`train.py` directly), and compare the new `models/model_config.json`
metrics against the currently deployed model before promoting it.

## Notes

- The scaler is only actually used by Logistic Regression; tree models
  ignore it, but it's kept as a shared artifact so `predict.py` doesn't
  need per-model branching logic beyond the `use_scaled_input` flag in
  `model_config.json`.
- The best model is selected by **recall on the failure class**, not
  accuracy — see the accompanying paper's Evaluation Metrics section for
  why a missed failure is costlier than a false alarm for this system.
