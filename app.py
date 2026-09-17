"""
Flask REST API for the Telco Customer Churn model.

Loads the trained pipeline saved by the notebook (model/churn_model.pkl) and exposes
a single POST /predict endpoint that accepts raw customer attributes as JSON, applies
the exact same preprocessing used during training (see preprocessing.py), and returns
the churn prediction and probability.
"""

import os

from flask import Flask, jsonify, render_template, request
import joblib

from preprocessing import validate_and_build_dataframe

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "churn_model.pkl")

app = Flask(__name__)
model = joblib.load(MODEL_PATH)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify(error="Request body must be valid JSON."), 400

    try:
        row_df = validate_and_build_dataframe(payload)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    prediction = model.predict(row_df)[0]
    probability = model.predict_proba(row_df)[0, 1]

    return jsonify(
        prediction="Yes" if prediction == 1 else "No",
        churn_probability=round(float(probability), 4),
    )


@app.errorhandler(404)
def not_found(_exc):
    return jsonify(error="Not found."), 404


@app.errorhandler(405)
def method_not_allowed(_exc):
    return jsonify(error="Method not allowed."), 405


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
