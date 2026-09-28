# ChurnGuard — Telco Customer Churn Prediction and AWS CI/CD

ChurnGuard predicts telecom customer churn from the IBM Telco Customer Churn dataset. The portfolio focus is the production path around a deliberately straightforward classifier: modular training, a Flask application, automated tests, containerization, and automated AWS deployment.

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

## Architecture

```text
Developer
↓
GitHub main
↓
GitHub Actions CI
├─ Ruff
├─ Pytest
├─ dataset validation
├─ model training
└─ Docker smoke test
↓
successful CI only
↓
GitHub Actions CD
↓
GitHub OIDC
↓
AWS IAM
↓
Amazon ECR
↓
AWS Systems Manager
↓
Amazon EC2
↓
Docker candidate container
↓
health validation
↓
production container
↓
public application
```

CI runs on pull requests and pushes to `main`. It runs Ruff and pytest, downloads and validates the dataset, trains and verifies model artifacts, then builds Docker and runs a health smoke test. CD automatically runs only after successful CI triggered by a push to `main`; manual `workflow_dispatch` is available as a fallback.

## Technology Stack

Python, Pandas, Scikit-learn, XGBoost, Flask, Gunicorn, Pytest, Ruff, Docker, GitHub Actions, AWS IAM, GitHub OIDC, Amazon ECR, Amazon EC2, and AWS Systems Manager.

## Deployment Security

- GitHub Actions authenticates to AWS through GitHub OIDC and receives temporary AWS credentials; permanent AWS access keys are not stored in GitHub.
- The IAM trust role is restricted to this repository and the `main` branch.
- The GitHub deploy role has scoped ECR push and SSM permissions.
- The EC2 instance role provides SSM and ECR image-read permissions.
- Deployment uses Systems Manager; SSH is not used.

## Image and Deployment Strategy

Images are tagged with the immutable Git commit SHA that passed CI. Production deploys that SHA-tagged image, not `latest`; Amazon ECR stores the deployment images.

The new image first runs as `churnguard-candidate`, bound only to `127.0.0.1:5001`. Production remains untouched until the candidate is healthy. After validation, the production container runs as `churnguard` with host port 80 mapped to container port 5000. Docker health checks and an EC2-local `/health` request verify the deployment.

Before replacing an existing production container, it is preserved as `churnguard-rollback`. If the new production container fails health validation, it is removed and the previous container is restored and checked. The GitHub Actions job still fails even when rollback succeeds, making the failed release visible.

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
docker build -t churnguard:local .
docker run --detach --name churnguard-local --publish 5000:5000 churnguard:local
Invoke-RestMethod http://localhost:5000/health
Start-Process http://localhost:5000
docker rm --force churnguard-local
```

The image runs Gunicorn as a non-root user and includes a Docker health check. Container logs go to stdout/stderr.

## Testing

```powershell
python -m pytest -q
ruff check app.py src tests
```

Tests cover ingestion, preprocessing with unknown categories, model artifacts/predictions, and Flask health and prediction routes. CI also builds the image, starts a temporary container, waits for its health check, and removes it.

## CI

Pull requests and pushes to `main` run CI: Ruff, pytest, dataset validation, model training, and a Docker health smoke test. CD runs automatically only after a successful CI run caused by a push to `main`; manual `workflow_dispatch` remains available as a fallback.

## AWS Deployment

CD uses GitHub OIDC for temporary AWS credentials, pushes the commit-SHA-tagged image to Amazon ECR, and uses Systems Manager to deploy it to Amazon EC2. Candidate and production health checks gate the deployment; an unhealthy production replacement triggers rollback to the preserved container. See [Image and Deployment Strategy](#image-and-deployment-strategy) for candidate and rollback behavior.

### ECR Image Retention

Retain enough SHA-tagged ECR images for the desired rollback window. ECR lifecycle rules cannot dynamically inspect which digest EC2 is running; preserve the deployed and rollback images when applying expiration rules.

## Repository Structure

```text
.github/workflows/
	ci.yml              CI validation and Docker smoke test
	deploy.yml          OIDC-based ECR push and SSM EC2 deployment
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