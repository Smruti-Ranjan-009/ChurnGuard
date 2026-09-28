import pandas as pd
import pytest


@pytest.fixture
def telco_data():
    rows = []
    for index in range(40):
        rows.append({
            "gender": "Female" if index % 2 else "Male",
            "SeniorCitizen": index % 2,
            "Partner": "Yes" if index % 3 else "No",
            "Dependents": "No" if index % 3 else "Yes",
            "tenure": index + 1,
            "PhoneService": "Yes",
            "MultipleLines": "No" if index % 2 else "Yes",
            "InternetService": "DSL" if index % 2 else "Fiber optic",
            "OnlineSecurity": "No" if index % 2 else "Yes",
            "OnlineBackup": "Yes" if index % 2 else "No",
            "DeviceProtection": "No" if index % 2 else "Yes",
            "TechSupport": "Yes" if index % 2 else "No",
            "StreamingTV": "No" if index % 2 else "Yes",
            "StreamingMovies": "Yes" if index % 2 else "No",
            "Contract": "Month-to-month" if index % 2 else "One year",
            "PaperlessBilling": "Yes" if index % 2 else "No",
            "PaymentMethod": "Electronic check" if index % 2 else "Mailed check",
            "MonthlyCharges": 35.0 + index,
            "TotalCharges": "" if index == 0 else str((35 + index) * (index + 1)),
            "Churn": "Yes" if index % 2 else "No",
        })
    return pd.DataFrame(rows)