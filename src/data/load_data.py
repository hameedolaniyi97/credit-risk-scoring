import pandas as pd
from pathlib import Path

# The raw file has no header row, so we name the columns ourselves,
# matching the order documented on the UCI page.
COLUMN_NAMES = [
    "checking_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_status", "employment_since", "installment_rate",
    "personal_status_sex", "other_parties", "residence_since", "property_type",
    "age_years", "other_installment_plans", "housing", "existing_credits",
    "job", "num_dependents", "telephone", "foreign_worker", "target"
]

RAW_DATA_PATH = Path("data/raw/german.data")


def load_raw() -> pd.DataFrame:
    """Load the raw German Credit dataset and remap the target."""
    df = pd.read_csv(RAW_DATA_PATH, sep=" ", names=COLUMN_NAMES)

    # Original coding: 1 = good credit, 2 = bad credit.
    # We remap to 0/1 so "1" always means "default" (the thing we're
    # trying to predict), which is the convention scikit-learn/XGBoost
    # and every metric (precision, recall, etc.) expect.
    df["target"] = df["target"].map({1: 0, 2: 1})

    return df


if __name__ == "__main__":
    df = load_raw()
    print("Shape:", df.shape)
    print("Target distribution:\n", df["target"].value_counts(normalize=True))