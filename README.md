<div align="center">
  <img src="assets/hero-banner.svg" alt="Automated ML Drift &amp; Risk Monitor" width="100%" />
  <br/><br/>
  <p>
    <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/Flask%20%2B%20Gunicorn-Diagnostics%20API-000000?style=flat-square&amp;logo=flask" alt="Flask" />
    <img src="https://img.shields.io/badge/Control%20Loop-Automated%20Drift%20Retraining-0D9488?style=flat-square" alt="MLOps" />
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-475569?style=flat-square" alt="License" /></a>
  </p>
</div>

---

![Closed-Loop ML Drift Detection & Automated Retraining Blueprint](docs/images/architecture_blueprint.png)

Closed-loop MLOps dynamic risk assessment and concept-drift remediation system in Python and Flask. Continuously monitors `sourcedata/` via a 10-minute Cron loop (`fullprocess.py`), performs deduplicated multi-source ingestion, evaluates **raw-comparison F1 model drift** against the deployed champion (`production_deployment/latestscore.txt`), conditionally triggers automated retraining + redeployment, regenerates confusion matrices, and captures live diagnostic telemetry across 4 REST endpoints.

![Closed-Loop Drift & Diagnostics Telemetry](images/drift_control_loop_telemetry.png)

---

## 1. Closed-Loop Control Architecture

```mermaid
flowchart TD
    A["Cron Scheduler (Every 10m)<br/>fullprocess.py"] --> B{"Compare sourcedata/ CSVs vs.<br/>production_deployment/ingestedfiles.txt"}
    B -- "No Unseen Batches" --> Z["Terminate Cleanly (Zero Compute Spend)"]
    B -- "New Batches Detected" --> C["ingestion.py<br/>Concatenate + Drop Duplicates + Write Manifest"]
    C --> D["Raw-Comparison Drift Test<br/>Score Deployed Champion on New Ingested Data"]
    D --> E{"new_f1 < deployed_f1 (0.5714)?"}
    E -- "False (No Concept Drift)" --> Z
    E -- "True (Model Drift Confirmed)" --> F["training.py -> scoring.py -> deployment.py<br/>Promote Fresh Artifacts to production_deployment/"]
    F --> G["reporting.py (confusionmatrix2.png)<br/>+ apicalls.py -> Flask API (apireturns2.txt)"]
```

---

## 2. Practice Baseline vs. Post-Drift Production Confusion Matrices

| Initial Practice Baseline (`confusionmatrix.png`) | Retrained Production Model (`confusionmatrix2.png`) |
| :---: | :---: |
| ![Practice Confusion Matrix](confusionmatrix.png) | ![Production Confusion Matrix](confusionmatrix2.png) |

---

## 3. Verified Flask API Telemetry ([`apireturns2.txt`](apireturns2.txt))

Following drift-triggered retraining and redeployment, `apicalls.py` queries all 4 live Flask endpoints (`POST /prediction`, `GET /scoring`, `GET /summarystats`, `GET /diagnostics`):

```json
=== Model Predictions ===
Status: 200
Response:
[
  0,
  1,
  1,
  1,
  1
]


=== Model Score (F1) ===
Status: 200
Response:
0.5714285714285714


=== Summary Statistics ===
Status: 200
Response:
[
  5625.923076923077,
  425.0,
  19444.774621317832,
  763.5384615384615,
  97.5,
  2016.6093569309696,
  457.46153846153845,
  99.0,
  800.604832899189,
  0.5,
  0.5,
  0.5099019513592785
]


=== Diagnostics ===
Status: 200
Response:
{
  "dependencies": [
    [
      "click",
      "8.3.1",
      "8.5.0"
    ],
    [
      "cycler",
      "0.12.1",
      "0.12.1"
    ],
    [
      "Flask",
      "3.1.3",
      "3.1.3"
    ],
    [
      "gunicorn",
      "23.0.0",
      "26.2.0"
    ],
    ...
  ],
  "execution_time": [0.82, 3.22],
  "missing_data": [0.0, 0.0, 0.0, 0.0, 0.0]
}
```

---

## 4. Module Reference & Quick Start

| Module | Responsibility |
| :--- | :--- |
| `ingestion.py` | Scans configured input folder, merges CSVs, removes duplicate rows, and writes `ingesteddata/finaldata.csv` + `ingestedfiles.txt`. |
| `training.py` | Fits regularized `LogisticRegression` on `lastmonth_activity`, `lastyear_activity`, and `number_of_employees` → `trainedmodel.pkl`. |
| `scoring.py` | Computes held-out test F1 score and persists `latestscore.txt`. |
| `deployment.py` | Promotes `trainedmodel.pkl`, `latestscore.txt`, and `ingestedfiles.txt` atomically into `production_deployment/`. |
| `diagnostics.py` | Benchmarks ingestion/training execution latency, column-wise NA rates, summary statistics, and `pip list --outdated` dependency drift. |
| `app.py` / `wsgi.py` | Flask + Gunicorn REST microservice serving `/prediction`, `/scoring`, `/summarystats`, and `/diagnostics`. |
| `fullprocess.py` | Autonomous end-to-end orchestrator invoked every 10 minutes via [`cronjob.txt`](cronjob.txt). |

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start Flask diagnostics & inference server
gunicorn --bind 0.0.0.0:8000 wsgi:app

# Trigger closed-loop drift check & conditional retraining
python fullprocess.py
```

## License

MIT License — see [LICENSE](LICENSE).
