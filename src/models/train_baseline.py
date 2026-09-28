import sys
sys.path.append(".")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report

from src.data.load_data import load_raw
from src.features.build_features import build_preprocessor


def main():
    df = load_raw()
    X = df.drop(columns=["target"])
    y = df["target"]

    # stratify=y keeps the 70/30 class balance consistent between
    # train and test sets without it, one split could randomly end up 
    # with a different default rate than the other, skewing evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=32, stratify=y
    )

    # We bundle preprocessing and model into ONE pipeline object.
    # This matters a lot: it guarantees the exact same preprocessing steps (fitted only
    # on training data) get applied consistently to test data and later to a single
    # loan applicant in thr app with zero risk of forgetting a step or doing it
    # differently 
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=32)),
        ]
    )

    pipeline.fit(X_train, y_train)

    # predict proba gives propability of each class: [:, 1] takes the probability
    # of class 1 (default) what we actually care about.
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = pipeline.predict(X_test)

    print("ROC AUC:", roc_auc_score(y_test, probs))
    print()
    print(classification_report(y_test, preds, digits=4))


if __name__ == "__main__":
    main()


