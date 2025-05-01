import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def normalize_data(x_train_path, x_test_path, output_dir="data/processed_data"):
    print(f"Chargement de {x_train_path} et {x_test_path}")
    
    # Charger les datasets
    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)

    # Sélectionner uniquement les colonnes numériques
    numeric_columns = X_train.select_dtypes(include=['number']).columns
    print(f"Colonnes numériques sélectionnées : {list(numeric_columns)}")

    scaler = StandardScaler()

    # Normaliser uniquement les colonnes numériques
    X_train_scaled = scaler.fit_transform(X_train[numeric_columns])
    X_test_scaled = scaler.transform(X_test[numeric_columns])

    # Reconstruire les DataFrames avec les colonnes normalisées
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=numeric_columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=numeric_columns)

    # Créer le dossier de sortie si besoin
    os.makedirs(output_dir, exist_ok=True)

    # Sauvegarder les fichiers
    X_train_scaled_df.to_csv(f"{output_dir}/X_train_scaled.csv", index=False)
    X_test_scaled_df.to_csv(f"{output_dir}/X_test_scaled.csv", index=False)

    print("Normalisation terminée.")
    print(f"Fichiers sauvegardés dans : {output_dir}")

if __name__ == "__main__":
    normalize_data("data/processed_data/X_train.csv", "data/processed_data/X_test.csv")
