import sys 
sys.path.append(".")

import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    pipeline = joblib.load("models/best_model.joblib")
    X_train, X_test, y_train, y_test = joblib.load("models/train_test_split.joblib")

    # split the pipeline into its two parts: the preprocessor and the actual xgboost model.
    # SHAP's TreeExplainer needs the raw model itself, not the full pipeline and it needs 
    # data thats already been through preprocessing, since that's what the model 
    # actually sees
    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['classifier']

    X_test_transformed = preprocessor.transform(X_train)

    # Get the feature names AFTER one hot encoding, so SHAP labels each
    # bar with something meaningful (e.g. "cheaking_status_A14") instead of
    # generic "feature_0", "feature_1".
    feature_names = preprocessor.get_feature_names_out()

    # TreeExplainer is fast and exact for tree-based models like XGBoost -
    # this is one of the reasons we chose XGBoost over logistic regression
    # back in Step 6.
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test_transformed)
    shap_values.feature_names = feature_names

    # Explain the FIRST applicant in the test set as a demo - later,
    # the Streamlit app will do this on-demand for whichever applicant
    # the loan officer is looking at.
    Path("reports").mkdir(exist_ok=True)

    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig("reports/shap_waterfall_example.png")
    print("Saved SHAP waterfall to reports/shap_waterfall_example.png")

    # Also save a global summary plot - shows which features matter most
    # across ALL applicants, not just one. Useful for a model-level
    # "what does this model generally care about" story.
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig("reports/shap_summary.png")
    print("Saved SHAP summary to reports/shap_summary.png")


if __name__ == "__main__":
    main()