import sys 
sys.path.append(".")

import numpy as np 
import matplotlib.pyplot as plt
import joblib 
from sklearn.metrics import confusion_matrix
from pathlib import Path

# Business assusmption: missing a dafaulter costs 5x more than wrongly declining a good customer.
# This ratio is a judgement call your bank/shareholder would set not something the model 
# derives on its own.
COST_FALSE_NEGATIVE = 5       # approved someone who defaults
COST_FALSE_POSITIVE = 1       # declined someone who would've been fine


def main():
    pipeline = joblib.load ("models/best_model.joblib")
    X_train, X_test, y_train, y_test = joblib.load("models/train_test_split.joblib")

    # Probability of class 1 (default) for every applicant in the test set
    probs = pipeline.predict_proba(X_test)[:, 1]

    best_threshold = 0.5
    best_cost = np.inf
    results = []

    # Constrain the search: a real bank can't decline more than ~50% of
    # applicants and stay in business. This bounds the threshold sweep to
    # a realistic operating range, instead of letting pure cost minimization
    # push toward "decline almost everyone."
    for t in np.arange(0.20, 0.95, 0.01):
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

        total_cost = (fn * COST_FALSE_NEGATIVE) + (fp * COST_FALSE_POSITIVE)
        results.append((t, total_cost, fn, fp))

        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = t

    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Total cost at best threshold: {best_cost}")
    print()

    # Show the confusion matrix breakdown at the chosen threshold, so
    # you can see exactly what tradeoff you're accepting.
    preds = (probs >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    print(f"At threshold {best_threshold:.2f}:")
    print(f"  True Negatives (correctly approved):  {tn}")
    print(f"  False Positives (wrongly declined):   {fp}")
    print(f"  False Negatives (missed defaulters):  {fn}")
    print(f"  True Positives (correctly declined):  {tp}")

    # Save the threshold so the Streamlit app can load it, exactly like
    # it loads the model - never hardcode 0.36 (or whatever we get) directly
    # in the app.
    joblib.dump(best_threshold, "models/threshold.joblib")
    print()
    print("Threshold saved to models/threshold.joblib")

    plot_cost_curve(results, best_threshold)

def plot_cost_curve(results, best_threshold):
    thresholds = [r[0] for r in results]
    costs = [r[1] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, costs, color="steelblue")
    plt.axvline(best_threshold, color="crimson", linestyle="--",
                 label=f"Best Threshold = {best_threshold:.2f}")
    plt.xlabel("Decission Threshold")
    plt.ylabel("Total business cost")
    plt.title("cost vs. decision threshold")
    plt.legend()
    plt.tight_layout()

    Path("reports").mkdir(exist_ok=True)
    plt.savefig("reports/cost_vs_threshold.png")
    print("Saved plot to reports/cost_vs_threshold.png")


if __name__ == "__main__":
    main()