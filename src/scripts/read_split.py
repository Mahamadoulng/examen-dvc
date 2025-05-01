# src/scripts/read_split.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys

def preprocessing(data_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["silica_concentrate"])
    Y = data[["silica_concentrate"]]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    os.makedirs("data/processed_data", exist_ok=True)
    X_train.to_csv("data/processed_data/X_train.csv", index=False)
    X_test.to_csv("data/processed_data/X_test.csv", index=False)
    y_train.to_csv("data/processed_data/y_train.csv", index=False)
    y_test.to_csv("data/processed_data/y_test.csv", index=False)

if __name__ == "__main__":
    preprocessing(sys.argv[1])
