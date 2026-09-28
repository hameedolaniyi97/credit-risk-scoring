from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_COLS = [
    "duration_months", "credit_amount", "installment_rate",
    "residence_since", "age_years", "existing_credits", "num_dependents"
]

CATEGORICAL_COLS = [
    "checking_status", "credit_history", "purpose", "savings_status",
    "employment_since", "personal_status_sex", "other_parties",
    "property_type", "other_installment_plans", "housing", "job",
    "telephone", "foreign_worker"
]


def build_preprocessor() -> ColumnTransformer:
    """Build the impute -> scale/encode pipeline for numeric and categorical columns."""

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_COLS),
        ("cat", categorical_pipeline, CATEGORICAL_COLS)
    ])

    return preprocessor


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data.load_data import load_raw

    df = load_raw()
    X = df.drop(columns=["target"])

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    print("Original shape:", X.shape)
    print("Transformed shape:", X_transformed.shape)