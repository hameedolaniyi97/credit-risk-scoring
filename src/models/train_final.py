import sys
sys.path.append(".")

import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.load_data import load_raw
from src.features.build_features import build_preprocessor

def main():
    df = load_raw()
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=32, stratify=y
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0
            ))
    ])


    pipeline.fit(X_train, y_train)

    # Save the whole pipeline (preprocessing + model together) as one
    # file. This is important: it means the Streamlit app later just
    # loads ONE object and calls .predict_proba() on raw input - it
    # never needs to know preprocessing even happened.
    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipeline, "models/best_model.joblib")

    # Also save the train/test split itself, so Step 7 (threshold
    # selection) and evaluation use the exact same held-out test set,
    # not a freshly random one each time.
    joblib.dump((X_train, X_test, y_train, y_test), "models/train_test_split.joblib")

    print("Model saved to models/best_model.joblib")
    print("Train/test split saved to models/train_test_split.joblib")


if __name__ == "__main__":
    main()