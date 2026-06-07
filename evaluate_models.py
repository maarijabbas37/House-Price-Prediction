import os
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PREDICTION_MODEL_DIR = os.path.join("saved_models", "prediction")
RECOMMENDER_DIR = os.path.join("saved_models", "recommender_10k")


def load_price_model():
    model_filename = os.getenv("PRICE_MODEL_FILENAME", "real_estate_price_predictor.joblib")
    columns_filename = os.getenv("PRICE_MODEL_COLUMNS_FILENAME", "model_training_columns.joblib")

    model_path = os.path.join(PREDICTION_MODEL_DIR, model_filename)
    columns_path = os.path.join(PREDICTION_MODEL_DIR, columns_filename)

    model = joblib.load(model_path)
    feature_index = joblib.load(columns_path)
    return model, feature_index


def prepare_features(df: pd.DataFrame, feature_index: pd.Index) -> pd.DataFrame:
    working = df.copy()

    for col in ["location", "city", "locality"]:
        working[col] = (
            working[col]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

    encoded = pd.get_dummies(
        working,
        columns=["location", "city", "locality"],
        dtype=int,
    )

    aligned = encoded.reindex(columns=feature_index, fill_value=0)
    return aligned


def evaluate_price_model() -> None:
    model, feature_index = load_price_model()

    dataset_path = os.path.join(RECOMMENDER_DIR, "balanced_10k_properties.csv")
    dataset = pd.read_csv(dataset_path)

    required_columns: List[str] = ["baths", "bedrooms", "area_marla", "location", "city", "locality", "price"]
    missing_columns = [col for col in required_columns if col not in dataset.columns]
    if missing_columns:
        raise RuntimeError(f"Dataset is missing required columns: {missing_columns}")

    features = dataset[required_columns].dropna(subset=["price"]).reset_index(drop=True)
    feature_matrix = prepare_features(features.drop(columns=["price"]), feature_index)

    predictions = model.predict(feature_matrix)
    targets = features["price"].to_numpy()

    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(targets, predictions)

    print("=== Price Prediction Model Evaluation ===")
    print(f"Rows evaluated: {len(targets)}")
    print(f"MAE : {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R^2 : {r2:.4f}")
    print("\nSample predictions (actual vs predicted):")
    for idx in range(5):
        actual = targets[idx]
        predicted = predictions[idx]
        print(f"  Row {idx}: actual={actual:,.0f} — predicted={predicted:,.0f}")


def demo_recommendations(sample_size: int = 5) -> None:
    dataset_path = os.path.join(RECOMMENDER_DIR, "balanced_10k_properties.csv")
    similarity_path = os.path.join(RECOMMENDER_DIR, "cosine_similarity_matrix_10k.joblib")

    properties = pd.read_csv(dataset_path)
    similarity_matrix = joblib.load(similarity_path)
    id_to_index = pd.Series(properties.index, index=properties["property_id"])

    if properties.empty:
        print("\nNo properties available to demonstrate recommendations.")
        return

    property_id = int(os.getenv("EVAL_PROPERTY_ID", properties.iloc[0]["property_id"]))
    if property_id not in id_to_index:
        raise RuntimeError(f"Property ID {property_id} not found in dataset.")

    property_idx = id_to_index[property_id]
    similarity_scores = similarity_matrix[property_idx]
    ranked_indices = np.argsort(similarity_scores)[::-1]

    # skip self match at rank 0
    ranked_indices = ranked_indices[ranked_indices != property_idx][:sample_size]
    recommendations = properties.iloc[ranked_indices][
        ["property_id", "city", "locality", "price", "baths", "bedrooms", "area_marla"]
    ]

    print(f"\n=== Recommendations for property_id={property_id} ===")
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    evaluate_price_model()
    demo_recommendations()

