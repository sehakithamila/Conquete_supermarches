from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

GOLD_PRODUCTS = Path("data/gold/products")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "energy_kcal_100g", "sugars_100g", "salt_100g",
    "fat_100g", "saturated-fat_100g", "proteins_100g",
    "fiber_100g", "additives_count", "nova_group",
]
TARGET = "nutriscore_grade"


def train_random_forest():
    print("Chargement des données Gold...")
    # On ne lit que les colonnes utiles (économise la RAM)
    df = pd.read_parquet(GOLD_PRODUCTS, columns=FEATURES + [TARGET])

    # Lignes valides + uniquement les vrais grades A..E
    df_model = df.dropna()
    df_model = df_model[df_model[TARGET].isin(["a", "b", "c", "d", "e"])]
    print(f"{len(df_model):,} lignes utilisables")
    if df_model.empty:
        raise RuntimeError("Aucune ligne exploitable : vérifie l'ETL (02_etl_spark.py).")

    X = df_model[FEATURES]
    le = LabelEncoder()
    y = le.fit_transform(df_model[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Entraînement du Random Forest en cours...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n--- Rapport de classification ---")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    model_path = MODEL_DIR / "random_forest_nutriscore.joblib"
    encoder_path = MODEL_DIR / "label_encoder.joblib"
    joblib.dump(clf, model_path)
    joblib.dump(le, encoder_path)
    print(f"\nModèle sauvegardé avec succès dans : {model_path}")


if __name__ == "__main__":
    train_random_forest()