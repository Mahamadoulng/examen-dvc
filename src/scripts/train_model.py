import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
import os

def train_model(x_train_path, y_train_path, best_params_path, model_output_path="models/model.pkl"):
    # Chargement des données
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()

    # Chargement des meilleurs paramètres
    best_params = joblib.load(best_params_path)
    print("Paramètres utilisés pour l'entraînement :", best_params)

    # Entraînement du modèle
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)

    # Sauvegarde du modèle
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Modèle entraîné sauvegardé dans {model_output_path}")

if __name__ == "__main__":
    train_model(
        x_train_path="data/processed_data/X_train_scaled.csv",
        y_train_path="data/processed_data/y_train.csv",
        best_params_path="models/best_params.pkl"
    )
