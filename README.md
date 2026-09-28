# ChurnDeploy — ML Deployment with AWS CI/CD

ChurnDeploy predicts telecom customer churn from the IBM Telco Customer Churn dataset. The portfolio focus is the production path around a deliberately straightforward classifier: modular training, a Flask application, automated tests, containerization, and continuous integration.

## Problem

Estimate whether a customer is likely to churn (`Churn = 1`) or stay (`Churn = 0`). The model is a baseline, not a claim of optimized business performance. `customerID` is excluded from model inputs. Empty `TotalCharges` values are coerced to missing and imputed during preprocessing.

## ML Pipeline

```text
CSV data
→ schema validation and stratified split
→ fold-local preprocessing
→ classifier comparison by cross-validated ROC-AUC
→ held-out model evaluation
→ model.pkl and preprocessor.pkl
```

The candidates are LogisticRegression, RandomForestClassifier, and XGBClassifier. The selected model is evaluated on the held-out test split with ROC-AUC, F1, precision, recall, and accuracy. Preprocessing is fitted inside each cross-validation fold and then fitted once on the full training split for the selected model.

## Application

Flask serves a customer form at `/`, accepts form or JSON input at `POST /predict`, and exposes `GET /health` for liveness and container smoke tests. Predictions include a class, estimated churn probability, and a demo risk band:

- `LOW`: probability below 0.30
- `MEDIUM`: 0.30 to below 0.60
- `HIGH`: 0.60 or above

These bands are product/demo thresholds, not statistically optimized or validated business decision thresholds.

## Current Architecture

```text
Developer
	↓
GitHub
	↓
GitHub Actions
	↓
CI: Ruff + pytest + Docker health smoke test
	↓
Docker container
	↓
Flask + ML model
```

CI is implemented in `.github/workflows/ci.yml`. Pull requests and pushes to `main` use Python 3.12, run lint and tests, download and validate the Telco CSV through the `TELCO_DATA_URL` repository variable, train the model artifacts, verify both artifacts, build Docker, and check container health. The Dropbox location is only a CI data-provisioning mechanism; its URL is not stored in this repository. AWS deployment is not implemented.

## Technology Stack

Current: Python, Pandas, Scikit-learn, XGBoost, Flask, Gunicorn, Pytest, Ruff, Docker, and GitHub Actions CI. Application logs are written to stdout/stderr in a format suitable for later collection by CloudWatch.

## Planned Security Model

The planned AWS phase will use GitHub OIDC for short-lived AWS sessions and will not store permanent AWS access keys in GitHub. Its IAM trust policy should be restricted to this repository and the `main` ref, with least-privilege access to ECR and the target EC2 instance. No AWS credentials or deployment workflow are present in the repository.

## Local Setup

The raw dataset is not committed to this repository. Local development expects it at `notebook/data/WA_Fn-UseC_-Telco-Customer-Churn.csv`; that CSV path is ignored by Git. The dataset is described on the [Kaggle Telco Customer Churn page](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), which attributes it to IBM sample data and displays “Data files © Original Authors” rather than an explicit standard redistribution license. IBM also hosts a related [Telco churn code-pattern repository](https://github.com/IBM/telco-customer-churn-on-icp4d) under Apache-2.0; that code-pattern license does not unambiguously establish terms for every dataset file.

For GitHub Actions, configure the repository variable `TELCO_DATA_URL` with the public Dropbox shared link. CI downloads it to the same expected path and validates the file, row count, and required columns before training. The Dropbox location is only a CI data-provisioning mechanism and its URL is not hardcoded in this repository. Obtain/use a dataset copy only where its source terms permit your intended use; do not commit the raw CSV until redistribution terms have been verified.

```powershell
python -m pip install -e ".[test]"
python -m src.pipeline.train_pipeline
python app.py
```

The app listens on port 5000 by default; set `PORT` to override it. Training generates `artifacts/model.pkl` and `artifacts/preprocessor.pkl`; both are ignored by Git and regenerated in CI before Docker builds. The fitted artifacts are included in the Docker build context; the raw dataset, notebooks, tests, caches, and logs are excluded from the image.

### Docker

Train the model first, then build and run:

```powershell
docker build -t churndeploy:local .
docker run --detach --name churndeploy-local --publish 5000:5000 churndeploy:local
Invoke-RestMethod http://localhost:5000/health
Start-Process http://localhost:5000
docker rm --force churndeploy-local
```

The image runs Gunicorn as a non-root user and includes a Docker health check. Container logs go to stdout/stderr.

## Testing

```powershell
python -m pytest -q
ruff check app.py src tests
```

Tests cover ingestion, preprocessing with unknown categories, model artifacts/predictions, and Flask health and prediction routes. CI also builds the image, starts a temporary container, waits for its health check, and removes it.

## CI

Pull requests and pushes to `main` run Ruff, pytest, Docker build, and a health-gated container smoke test. CI does not authenticate to AWS or deploy the application.

## Planned AWS Deployment (Not Yet Implemented)

The intended future architecture is:

```text
GitHub Actions
→ GitHub OIDC
→ AWS IAM
→ Amazon ECR
→ AWS Systems Manager
→ EC2
→ Docker
→ /health
```

Before implementing this phase, create an ECR repository; an Ubuntu EC2 instance with Docker and SSM Agent; an EC2 instance profile with SSM and ECR pull permissions; the GitHub OIDC provider; and a GitHub IAM role restricted to this repository's `main` branch with ECR push and scoped SSM command permissions. Configure deployment values as GitHub variables, not committed credentials. Do not configure EC2 as a GitHub runner.

### ECR Image Retention

When deployment is implemented, consider retaining the newest 5–10 SHA-tagged images. ECR lifecycle rules cannot dynamically inspect which digest EC2 is running. For a strict guarantee that active and rollback images are retained, use a cleanup process that checks those digests before deleting images; do not use a broad automatic expiration rule.

## Repository Structure

```text
.github/workflows/   CI workflow only
artifacts/           fitted model and preprocessing artifacts
notebook/data/       local source dataset (not included in the image)
src/components/      ingestion, transformation, model training
src/pipeline/        training and inference pipelines
templates/           Flask views
tests/               ingestion, preprocessing, model, and app tests
app.py               Flask application
Dockerfile           production container image
```

## Future Improvements

Model monitoring, drift detection, blue/green deployment, automated retraining, and infrastructure as code are possible follow-ups. They are not implemented in this repository.