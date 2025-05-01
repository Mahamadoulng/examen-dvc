import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
import os

def evaluate_model(model_path, x_test_path, y_test_path, metrics_path="metrics/scores.json", output_path="data/predictions.csv"):
    # Chargement des données et du modèle
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()
    model = joblib.load(model_path)

    # Prédictions
    y_pred = model.predict(X_test)

    # Calcul des métriques
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    scores = {
        "mse": mse,
        "r2": r2,
        "mae": mae
    }

    # Sauvegarde des métriques
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(scores, f, indent=4)

    # Sauvegarde des prédictions
    pred_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred
    })
    pred_df.to_csv(output_path, index=False)

    print(f"Métriques enregistrées dans {metrics_path}")
    print(f"Prédictions sauvegardées dans {output_path}")

if __name__ == "__main__":
    evaluate_model(
        model_path="models/model.pkl",
        x_test_path="data/processed_data/X_test_scaled.csv",
        y_test_path="data/processed_data/y_test.csv"
    )
