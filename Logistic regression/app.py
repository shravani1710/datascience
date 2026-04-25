import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
st.title("🚢 Titanic Survival Predictor")
st.markdown("Enter passenger details below to predict survival using a **Logistic Regression** model.")
st.markdown("---")

# Train model directly inside the app — no pickle needed
@st.cache_resource
def train_model():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    data = pd.read_csv(url)
    data.drop("Cabin", inplace=True, axis=1)
    data.dropna(inplace=True)
    data.drop(columns=["Name", "Ticket"], inplace=True)
    data = pd.get_dummies(data, columns=["Sex", "Embarked"], dtype=int)
    features = data.drop("Survived", axis=1)
    target   = data["Survived"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, train_size=0.8, random_state=11
    )
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model

model = train_model()

# Input form
st.subheader("Passenger Details")
col1, col2 = st.columns(2)

with col1:
    pclass   = st.selectbox("Passenger Class", [1, 2, 3],
                 format_func=lambda x: f"{x} - {'First' if x==1 else 'Second' if x==2 else 'Third'} Class")
    sex      = st.selectbox("Sex", ["male", "female"])
    age      = st.slider("Age", 1, 80, 30)
    sibsp    = st.number_input("Siblings / Spouses (SibSp)", 0, 8, 0)

with col2:
    parch    = st.number_input("Parents / Children (Parch)", 0, 6, 0)
    fare     = st.number_input("Fare (£)", 0.0, 520.0, 32.0)
    embarked = st.selectbox("Embarked", ["S", "C", "Q"],
                 format_func=lambda x: {"S":"S - Southampton","C":"C - Cherbourg","Q":"Q - Queenstown"}[x])

st.markdown("---")

if st.button("🔮 Predict Survival", use_container_width=True):
    input_df = pd.DataFrame([{
        "PassengerId": 1,
        "Pclass":      pclass,
        "Age":         float(age),
        "SibSp":       sibsp,
        "Parch":       parch,
        "Fare":        fare,
        "Sex_female":  1 if sex == "female" else 0,
        "Sex_male":    1 if sex == "male"   else 0,
        "Embarked_C":  1 if embarked == "C" else 0,
        "Embarked_Q":  1 if embarked == "Q" else 0,
        "Embarked_S":  1 if embarked == "S" else 0,
    }])

    pred  = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    if pred == 1:
        st.success(f"✅ Survived!  Probability: {proba[1]:.2%}")
    else:
        st.error(f"❌ Did Not Survive.  Probability: {proba[1]:.2%}")

    prob_df = pd.DataFrame({
        "Outcome":     ["Did Not Survive", "Survived"],
        "Probability": [round(proba[0]*100,2), round(proba[1]*100,2)]
    })
    st.bar_chart(prob_df.set_index("Outcome"))

st.markdown("---")
st.caption("Model: Logistic Regression | Dataset: Titanic | Built with Streamlit")