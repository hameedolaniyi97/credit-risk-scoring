import sys 
import mlflow
sys.path.append(".")

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.load_data import load_raw
from src.features.build_features import build_preprocessor

def main():
    df = load_raw()
    X = df.drop(columns=["target"])
    y = df["target"]

    # StratifiedKFold keeps the 70/30 class balance consistent across every one of the 5
    # folds same reasoning as stratify in train_test_split, just applied fold by fold.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=32)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=32),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0
            )
    }

    # Give this set of experiments a name MLflow group every run under it, so you can 
    # browse "all model comperison attempts" together in the UI later, rather than one
    # flat unorganized list.
    mlflow.set_experiment("credit-risk-model-comperison")

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", model)
        ])

        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
        print(f"{name}: mean AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")
        print(f" individual folds: {[round(s, 3) for s in scores]}")
        print()

        # Each "run" is one logged attempt here, one model type. We log what we used
        # (params) and what happened (metrics), so you can compare runs side by side
        # later without re running anything.
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_type", name)

            # Only log hyperparameters for XGBoost logistic regression's defaults
            # aren't meaningfully tunable in the same way here.
            if name == "XGBoost":
                mlflow.log_params(model.get_params())

            mlflow.log_metric("mean_auc", scores.mean())
            mlflow.log_metric("std_auc", scores.std())
            for i, score in enumerate(scores):
                mlflow.log_metric(f"fold_{i}_auc", score)

if __name__ == "__main__":
    main()