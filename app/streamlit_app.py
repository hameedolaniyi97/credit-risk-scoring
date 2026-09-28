import sys
sys.path.append(".")

import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Risk Scoring", layout="wide")


@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("models/best_model.joblib")
    threshold = joblib.load("models/threshold.joblib")
    return pipeline, threshold


pipeline, threshold = load_artifacts()

@st.cache_resource
def get_explainer(_pipeline):
    # The leading underscore on _pipeline tells Streamlit's cachenot to try 
    # hashing the pipeline object itself (it can't reliably) it just runs this 
    # this one and reuses the result 
    model = _pipeline.named_steps['classifier']
    return shap.TreeExplainer(model)

explainer = get_explainer(pipeline)
    
st.title("🏦 Credit Risk Scoring — Loan Officer Dashboard")
st.caption(f"Model: XGBoost · Decision threshold: {threshold:.2f}")

st.sidebar.header("Applicant details")

# Numeric fields - typed entry instead of sliders, so the applicant/officer
# can enter exact values directly. step= controls how much each up/down
# arrow click changes the value by.
duration_months = st.sidebar.number_input("Loan duration (months)", min_value=4, max_value=72, value=24, step=1)
credit_amount = st.sidebar.number_input("Credit amount", min_value=250, max_value=20000000, value=2000, step=50)
age_years = st.sidebar.number_input("Age", min_value=19, max_value=75, value=35, step=1)
installment_rate = st.sidebar.number_input("Installment rate", min_value=1, max_value=10, value=2, step=1)
residence_since = st.sidebar.number_input("Years at current residence", min_value=1, max_value=10, value=2, step=1)
existing_credits = st.sidebar.number_input("Existing credits", min_value=1, max_value=10, value=1, step=1)
num_dependents = st.sidebar.number_input("People financially dependent", min_value=1, max_value=10, value=1, step=1)

# Human-readable labels mapped to the raw UCI codes the model expects.
# The selectbox shows the readable text; we look up the code afterward.
CHECKING_STATUS = {
    "Overdrawn (< 0 DM)": "A11", "0 to 200 DM": "A12",
    "200+ DM": "A13", "No checking account": "A14"
}
CREDIT_HISTORY = {
    "No credits / all paid duly": "A30", "All credits here paid duly": "A31",
    "Existing credits paid duly": "A32", "Delay in past payments": "A33",
    "Critical account / other credits elsewhere": "A34"
}
PURPOSE = {
    "New car": "A40", "Used car": "A41", "Furniture/equipment": "A42",
    "Radio/TV": "A43", "Domestic appliances": "A44", "Repairs": "A45",
    "Education": "A46", "Retraining": "A48", "Business": "A49", "Other": "A410"
}
SAVINGS_STATUS = {
    "< 100 DM": "A61", "100-500 DM": "A62", "500-1000 DM": "A63",
    "1000+ DM": "A64", "Unknown / no savings account": "A65"
}
EMPLOYMENT_SINCE = {
    "Unemployed": "A71", "< 1 year": "A72", "1-4 years": "A73",
    "4-7 years": "A74", "7+ years": "A75"
}
OTHER_PARTIES = {"None": "A101", "Co-applicant": "A102", "Guarantor": "A103"}
PROPERTY_TYPE = {
    "Real estate": "A121", "Building society savings/life insurance": "A122",
    "Car or other": "A123", "Unknown / no property": "A124"
}
OTHER_INSTALLMENT_PLANS = {"Bank": "A141", "Stores": "A142", "None": "A143"}
HOUSING = {"Rent": "A151", "Own": "A152", "For free": "A153"}
JOB = {
    "Unemployed / unskilled non-resident": "A171", "Unskilled resident": "A172",
    "Skilled employee/official": "A173", "Management/self-employed": "A174"
}
FOREIGN_WORKER = {"Yes": "A201", "No": "A202"}

checking_status = CHECKING_STATUS[st.sidebar.selectbox("Checking account status", list(CHECKING_STATUS.keys()))]
credit_history = CREDIT_HISTORY[st.sidebar.selectbox("Credit history", list(CREDIT_HISTORY.keys()))]
purpose = PURPOSE[st.sidebar.selectbox("Purpose", list(PURPOSE.keys()))]
savings_status = SAVINGS_STATUS[st.sidebar.selectbox("Savings status", list(SAVINGS_STATUS.keys()))]
employment_since = EMPLOYMENT_SINCE[st.sidebar.selectbox("Employment since", list(EMPLOYMENT_SINCE.keys()))]

# Sex and marital status as two separate dropdowns. The dataset's original
# combined field doesn't cover every sex/status combination (no "female,
# single" or "female, widowed" code exists), so the marital-status options
# shown depend on which sex is selected - this keeps every possible pick
# mapped to a real, valid code.
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])

if sex == "Male":
    marital_status = st.sidebar.selectbox(
        "Marital status", ["Divorced/Separated", "Single", "Married/Widowed"]
    )
    male_status_map = {
        "Divorced/Separated": "A91", "Single": "A93", "Married/Widowed": "A94"
    }
    personal_status_sex = male_status_map[marital_status]
else:
    marital_status = st.sidebar.selectbox(
        "Marital status", ["Divorced/Separated/Married"]
    )
    personal_status_sex = "A92"

other_parties = OTHER_PARTIES[st.sidebar.selectbox("Other parties", list(OTHER_PARTIES.keys()))]
property_type = PROPERTY_TYPE[st.sidebar.selectbox("Property", list(PROPERTY_TYPE.keys()))]
other_installment_plans = OTHER_INSTALLMENT_PLANS[st.sidebar.selectbox("Other installment plans", list(OTHER_INSTALLMENT_PLANS.keys()))]
housing = HOUSING[st.sidebar.selectbox("Housing", list(HOUSING.keys()))]
job = JOB[st.sidebar.selectbox("Job", list(JOB.keys()))]

# Telephone is a yes/no field in this dataset (not an actual phone number -
# the UCI data never collected real numbers), so a checkbox reads more
# like a direct input than a dropdown did.
has_telephone = st.sidebar.checkbox("Has registered telephone", value=True)
telephone = "A192" if has_telephone else "A191"

foreign_worker = FOREIGN_WORKER[st.sidebar.selectbox("Foreign worker", list(FOREIGN_WORKER.keys()))]

if st.sidebar.button("Score applicant"):
    # Build a single-row DataFrame matching the exact column names and
    # order the pipeline was trained on - this MUST match load_data.py's
    # COLUMN_NAMES (minus "target"), or the preprocessor will misalign
    # values to the wrong columns.
    applicant = pd.DataFrame([{
        "checking_status": checking_status,
        "duration_months": duration_months,
        "credit_history": credit_history,
        "purpose": purpose,
        "credit_amount": credit_amount,
        "savings_status": savings_status,
        "employment_since": employment_since,
        "installment_rate": installment_rate,
        "personal_status_sex": personal_status_sex,
        "other_parties": other_parties,
        "residence_since": residence_since,
        "property_type": property_type,
        "age_years": age_years,
        "other_installment_plans": other_installment_plans,
        "housing": housing,
        "existing_credits": existing_credits,
        "job": job,
        "num_dependents": num_dependents,
        "telephone": telephone,
        "foreign_worker": foreign_worker,
    }])

    prob_default = pipeline.predict_proba(applicant)[0, 1]
    decision = "DECLINE" if prob_default >= threshold else "APPROVE"

    col1, col2 = st.columns(2)
    with col1:
        if decision == "APPROVE":
            st.success(f"✅ {decision}")
        else:
            st.error(f"❌ {decision}")
    with col2:
        st.metric("Probability of default", f"{prob_default:.1%}")

    st.subheader("why this decision SHAP explanation")

    # Run the applicant's data through the preprocessing only (not the full pipeline)
    # SHAP needs the transformed features since that's what the model itself actually 
    # sees and reasons over 
    preprocessor = pipeline.named_steps["preprocessor"]
    applicant_transformed = preprocessor.transform(applicant)
    feature_names = preprocessor.get_feature_names_out()

    # Strip the "num__" / "cat__" ColumnTransformer prefixes so a loan
    # officer sees "duration_months" instead of "num__duration_months".
    clean_feature_names = [
        name.split("__", 1)[1] if "__" in name else name
        for name in feature_names
    ]

    shap_values = explainer(applicant_transformed)
    shap_values.feature_names = clean_feature_names

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
    plt.close(fig)