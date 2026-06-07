import pandas as pd
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
import os

# Initialize Flask App
app = Flask(__name__)

# --- Load Models at Startup ---
print("Loading price prediction model... Please wait.")

try:
    # ===========================
    # PRICE PREDICTION MODEL
    # ===========================
    PREDICTION_MODEL_DIR = 'saved_models/prediction'
    price_model = joblib.load(os.path.join(PREDICTION_MODEL_DIR, 'real_estate_price_predictor.joblib'))
    price_model_columns = joblib.load(os.path.join(PREDICTION_MODEL_DIR, 'model_training_columns.joblib'))

    print("✅ Price Prediction Model Loaded")

except Exception as e:
    print(f"ERROR loading price model: {e}")
    price_model = None


# ===========================
# HOME ROUTE → Show HTML Page
# ===========================
@app.route('/')
def home():
    return render_template('index.html')


# ===========================
# PRICE PREDICTION API
# ===========================
@app.route('/api/predict', methods=['POST'])
def predict_price():
    if price_model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])

        # Basic string cleaning
        for col in ['location', 'city', 'locality']:
            if col in input_df:
                input_df[col] = input_df[col].str.strip().str.lower()

        # One-hot encoding
        input_encoded = pd.get_dummies(input_df, columns=['location', 'city', 'locality'], dtype=int)
        input_aligned = input_encoded.reindex(columns=price_model_columns, fill_value=0)

        prediction = price_model.predict(input_aligned)
        return jsonify({"predicted_price": float(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)