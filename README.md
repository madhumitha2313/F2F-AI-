
## Project Structure

```
nullspace-ai/
├── data/
│   ├── raw/                  # place Dreher_and_Doyle_input_data.xlsx here
│   └── processed/            # written by src/preprocess.py
├── src/
│   ├── config.py             # shared paths, schema, hyperparameters
│   ├── preprocess.py         # clean data, build failure label, encode features
│   ├── train.py              # train + evaluate 4 models, log to MLflow
│   ├── predict.py            # single-reaction inference (CLI + shared logic)
│   ├── load_check.py / write_load_check.py
│   └── write_training_distribution.py
├── models/                   # written by train.py: best_model.pkl, scaler.pkl, ...
├── api/
│   ├── app.py                # Flask REST API + dashboard
│   ├── requirements.txt      # serving environment (Python 3.12+)
│   └── Dockerfile
├── monitoring/
│   ├── logger.py             # SQLite logging of every prediction request
│   └── drift_detector.py     # chi-square drift check vs. training distribution
├── colab/                    # F2F_AI_model_training_colab.py — notebook-style training
├── dvc.yaml                  # DVC pipeline: preprocess -> train
├── params.yaml                # DVC-tracked hyperparameters
└── requirements.txt           # top-level dev/training environment
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Serving the API?** Install from `api/requirements.txt` instead (Python
> 3.12+). The top-level `requirements.txt` pins an older `xgboost` (2.1.0)
> that loads the shipped `models/best_model.pkl` without error but reads it
> wrongly, silently shifting every failure probability upward. The API
> refuses to start in such an environment — see `src/load_check.py`.

Place your dataset at `data/raw/Dreher_and_Doyle_input_data.xlsx`.

## 2. Run the Pipeline

Either run the two stages directly:

```bash
python src/preprocess.py
python src/train.py
```

Or, with DVC set up (`dvc init` once, then `dvc remote add ...`):

```bash
dvc repro
```

This writes `models/best_model.pkl`, `models/label_encoders.pkl`,
`models/scaler.pkl`, `models/model_config.json`, and a comparison table
(`models/model_comparison.csv`) + plots for all four models.

## 3. Inspect Experiments in MLflow

```bash
mlflow ui
```

Open http://localhost:5000 (or the port MLflow prints) to see every run's
parameters, metrics, and logged model artifact side by side.

## 4. Serve Predictions

Locally:

```bash
python api/app.py
```

**Sign-in:** the dashboard is open by default — any non-empty username and
password gets in, and the login page says so. To require real credentials,
set `F2F_DEV_BYPASS_AUTH=0` together with `F2F_USERNAME`, `F2F_PASSWORD` and
`SECRET_KEY`. The JSON endpoints below are public either way.

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Landing page |
| `/login`, `/signup`, `/logout` | GET/POST | Auth pages (share one animated component) |
| `/dashboard` | GET | Main interactive dashboard |
| `/predict` | POST | Single-reaction failure prediction |
| `/explain` | POST | Per-input contribution + "try this instead" suggestions |
| `/predict/batch` | POST | Batch scoring (CSV or pasted rows, ≤200 rows / 100 KB) |
| `/stats` | GET | Model comparison + input-importance stats |
| `/recent-predictions` | GET | Recent prediction log |
| `/activity` | GET | Predictions per hour (5 days) / per day (14 days) |
| `/coverage` | GET | How often each known value has been queried |
| `/health` | GET | Liveness + model load-check status |

The dashboard is a grid of widgets you can rearrange (drag the handle, or
use each widget's arrow buttons — order is remembered per browser). Its
**"Why this?"** and **"Try this instead"** widgets call `/explain`, which
returns exact per-input contributions from the XGBoost model and one-input
swaps that lower the failure probability — describing what the model
learned, not a chemical mechanism.

Choosing a value draws its skeletal structure under the dropdown, and
**"The reaction"** widget shows your picks as a scheme (aryl halide +
4-methylaniline) above an animated Buchwald–Hartwig catalytic cycle — both
illustrations, not predictions; the model itself sees category IDs, not
mechanism. Structures are drawn client-side by SmilesDrawer 2.1.7 (MIT),
vendored in `api/static/vendor/` — nothing is loaded from a CDN.

Four more widgets read the prediction log and refresh after each
prediction: **Activity**, **Trend**, **Input coverage**, and **Batch**. All
dates/hours are UTC. Batch runs are not written to the prediction log, so
they don't skew the live stats.

Test the API directly:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"ligand": "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P([C@@]3(C[C@@H]4C5)C[C@H](C4)C[C@H]5C3)[C@]6(C7)C[C@@H](C[C@@H]7C8)C[C@@H]8C6)C(OC)=CC=C2OC",
       "additive": "CC1=CC(C)=NO1",
       "base": "CN1CCCN2C1=NCCC2",
       "aryl_halide": "ClC1=CC=C(OC)C=C1"}'
```

The API expects full SMILES strings, exactly as the dashboard's dropdowns
show them. Short display labels like `L1` or `AH5` are login-page names
only — sending them returns predictions flagged "not seen during training."

Or with Docker, from the project root (so the build context includes `src/`
and `models/`):

```bash
docker build -f api/Dockerfile -t nullspace-ai-api .
docker run -p 5000:5000 -v $(pwd)/monitoring:/app/monitoring nullspace-ai-api
```

## 5. Monitoring and Drift

Every `/predict` call is logged to `monitoring/predictions.db`. `/stats`,
`/recent-predictions`, `/activity` and `/coverage` all read from that log.

`/coverage` compares live usage against the training mix in
`models/training_distribution.json`. The training grid covers every value
about equally, so an "uneven" verdict describes usage patterns, not model
staleness — the real drift check is `monitoring/drift_detector.py`. The
verdict is a chi-square test per input (four tests sharing one 5%
false-alarm budget), withheld until every input has 30 logged asks with
known values.

Regenerate after changing processed data or option lists:

```bash
python src/write_training_distribution.py
```

Run drift checks on a schedule (cron / GitHub Action / scheduled Colab
cell):

```bash
python monitoring/drift_detector.py
```

This writes `monitoring/drift_report.json`.

## 6. Retraining

Once drift is flagged or enough new labeled reactions are collected: append
them to `data/raw/`, rerun `dvc repro` (or `preprocess.py` + `train.py`
directly), and compare the new `models/model_config.json` metrics against
the currently deployed model before promoting it.

After every retrain, regenerate the load-time check in the verified serving
environment (`api/requirements.txt`):

```bash
python src/write_load_check.py
```

Until you do, `/health` reports `"load_check": "stale"` and the API logs a
warning that the new model's predictions are unverified.

## Notes

- The scaler is only actually used by Logistic Regression; tree models
  ignore it, but it's kept as a shared artifact so `predict.py` doesn't
  need per-model branching beyond the `use_scaled_input` flag in
  `model_config.json`.
- The best model is selected by **recall on the failure class**, not
  accuracy — see the accompanying report's Results and Discussion chapter
  for why a missed failure is costlier than a false alarm here.
- `models/load_check.json` and `models/training_distribution.json` are
  environment-verified snapshots, not raw training artifacts — regenerate
  them (steps above) after any retrain or environment change.

## Tech Stack

Python · pandas/NumPy/scikit-learn · XGBoost · Flask · SQLite · MLflow ·
DVC · Docker · SmilesDrawer (vendored, MIT)
