import os

from flask import Flask, jsonify, render_template, request

from src.exception import CustomException
from src.logger import logging
from src.pipeline.predict_pipeline import PredictPipeline, TelcoCustomerData

application = Flask(__name__)
app = application


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="healthy")


@app.route("/predict", methods=["POST"])
def predict():
    is_json_request = request.is_json
    customer_values = request.get_json(silent=True) if is_json_request else request.form.to_dict()
    try:
        customer = TelcoCustomerData(**(customer_values or {}))
        result = PredictPipeline().predict(customer.get_data_as_data_frame())
    except (TypeError, ValueError) as error:
        if is_json_request:
            return jsonify(error=str(error)), 400
        return render_template("home.html", error=str(error)), 400
    except CustomException:
        logging.exception("Churn prediction failed")
        if is_json_request:
            return jsonify(error="Prediction is temporarily unavailable."), 503
        return render_template("home.html", error="Prediction is temporarily unavailable."), 503
    if is_json_request:
        return jsonify(result)
    return render_template("home.html", result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))


