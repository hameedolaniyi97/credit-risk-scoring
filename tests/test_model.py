import numpy as np 
import joblib


def test_saved_model_loads_and_predicts():
    # This test assumes train_final.py has already been run, since it loads the saved artifacts
    # rather than retrainig keeps the test fast, and checks the SAVED artifact is actually usable,
    # not just that training code runs 
    pipeline = joblib.load("models/best_model.joblib")
    X_train, X_test, y_train, y_test = joblib.load("models/train_test_split.joblib")

    probs = pipeline.predict_proba(X_test)[:, 1]

    # One probability per row in the test set no row dropped or dumpicated during prediction 

    assert len(probs) == len(X_test)

    # A sanity check on model quality itself if AUC ever dropped below this, something
    # is badly wrong (e.g. a broken retrain, corruption data or accidental label leak/reversal)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, probs)
    assert auc > 0.65