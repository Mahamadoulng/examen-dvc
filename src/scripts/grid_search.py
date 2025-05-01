import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def run_grid_search(x_train_path, y_train_path, output_path="models/best_params.pkl"):
    print("Chargement des données...")
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()  # .ravel() pour éviter une erreur avec y au format DataFrame

    print("Initialisation du modèle et des paramètres...")
    model = RandomForestRegressor(random_state=42)

    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )

    print("Lancement du GridSearch...")
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print("Meilleurs paramètres trouvés :", best_params)

    # Création du dossier de sortie
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Sauvegarde des meilleurs paramètres
    joblib.dump(best_params, output_path)
    print(f"Paramètres enregistrés dans {output_path}")

if __name__ == "__main__":
    run_grid_search(
        x_train_path="data/processed_data/X_train_scaled.csv",
        y_train_path="data/processed_data/y_train.csv"
    )
