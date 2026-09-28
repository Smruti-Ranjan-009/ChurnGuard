def test_health_form_and_prediction_routes(monkeypatch, telco_data):
    import app as app_module

    class StubPredictionPipeline:
        def predict(self, features):
            return {"prediction": 1, "churn_probability": 0.81, "risk_level": "HIGH"}

    monkeypatch.setattr(app_module, "PredictPipeline", StubPredictionPipeline)
    client = app_module.app.test_client()
    health_response = client.get("/health")
    form_response = client.get("/")
    prediction_response = client.post(
        "/predict", json=telco_data.drop(columns="Churn").iloc[0].to_dict()
    )

    assert health_response.status_code == 200
    assert health_response.get_json() == {"status": "healthy"}
    assert form_response.status_code == 200
    assert prediction_response.status_code == 200
    assert prediction_response.get_json() == {
        "prediction": 1, "churn_probability": 0.81, "risk_level": "HIGH"
    }