import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Survival Predictor")
st.markdown("Enter passenger details below to predict survival using a **Logistic Regression** model.")
st.markdown("---")

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("titanic_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ Model file `titanic_model.pkl` not found. Please run the deployment cell in your notebook first.")
    st.stop()

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Passenger Details")

col1, col2 = st.columns(2)

with col1:
    pclass   = st.selectbox("Passenger Class (Pclass)", options=[1, 2, 3],
                             format_func=lambda x: f"{x} - {'First' if x==1 else 'Second' if x==2 else 'Third'} Class")
    sex      = st.selectbox("Sex", options=["male", "female"])
    age      = st.slider("Age", min_value=1, max_value=80, value=30)
    sibsp    = st.number_input("Siblings / Spouses aboard (SibSp)", min_value=0, max_value=8, value=0)

with col2:
    parch    = st.number_input("Parents / Children aboard (Parch)", min_value=0, max_value=6, value=0)
    fare     = st.number_input("Fare (£)", min_value=0.0, max_value=520.0, value=32.0, step=0.5)
    embarked = st.selectbox("Port of Embarkation", options=["S", "C", "Q"],
                             format_func=lambda x: {"S": "S - Southampton", "C": "C - Cherbourg", "Q": "Q - Queenstown"}[x])
    passenger_id = st.number_input("Passenger ID", min_value=1, value=1)

st.markdown("---")

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("🔮 Predict Survival", use_container_width=True):

    # Build input matching training columns exactly
    # After get_dummies on Sex and Embarked the columns are:
    # PassengerId, Pclass, Age, SibSp, Parch, Fare,
    # Sex_female, Sex_male, Embarked_C, Embarked_Q, Embarked_S
    input_dict = {
        "PassengerId":  passenger_id,
        "Pclass":       pclass,
        "Age":          age,
        "SibSp":        sibsp,
        "Parch":        parch,
        "Fare":         fare,
        "Sex_female":   1 if sex == "female" else 0,
        "Sex_male":     1 if sex == "male"   else 0,
        "Embarked_C":   1 if embarked == "C" else 0,
        "Embarked_Q":   1 if embarked == "Q" else 0,
        "Embarked_S":   1 if embarked == "S" else 0,
    }

    input_df = pd.DataFrame([input_dict])

    prediction   = model.predict(input_df)[0]
    probability  = model.predict_proba(input_df)[0]

    survived_prob = round(probability[1] * 100, 2)
    died_prob     = round(probability[0] * 100, 2)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"✅ **Survived** — Survival probability: **{survived_prob}%**")
    else:
        st.error(f"❌ **Did Not Survive** — Survival probability: **{survived_prob}%**")

    # Probability bar chart
    prob_df = pd.DataFrame({
        "Outcome":     ["Did Not Survive", "Survived"],
        "Probability": [died_prob, survived_prob]
    })
    st.bar_chart(prob_df.set_index("Outcome"))

    # Summary table
    st.markdown("#### Input Summary")
    summary = {
        "Passenger Class": f"Class {pclass}",
        "Sex":             sex.capitalize(),
        "Age":             age,
        "SibSp":           sibsp,
        "Parch":           parch,
        "Fare":            f"£{fare}",
        "Embarked":        {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[embarked],
    }
    st.table(pd.DataFrame(summary.items(), columns=["Feature", "Value"]))

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Model: Logistic Regression | Dataset: Titanic Train | Built with Streamlit")
