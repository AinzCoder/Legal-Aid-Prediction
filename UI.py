import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load your trained model
model = pickle.load(open('naive_bayes1.pkl', 'rb'))

st.title("Legal Aid Eligibility Predictor")
st.markdown("Enter applicant details to predict eligibility for legal aid.")

col1, col2 = st.columns(2)

with col1:
    sex         = st.selectbox("Sex", ["Male", "Female"])
    ethnicity   = st.selectbox("Ethnicity", ["African-American","Caucasian","Hispanic","Other","Asian","Native American"])
    age         = st.number_input("Age", min_value=18, max_value=80, value=30)
    marital     = st.selectbox("Marital Status", ["Single","Married","Divorced","Separated","Significant Other","Widowed"])
    legal       = st.selectbox("Legal Status", ["Pretrial","Post Sentence","Conditional Release","Probation Violator","Other"])

with col2:
    custody     = st.selectbox("Custody Status", ["Jail Inmate","Probation","Pretrial Defendant","Residential Program"])
    supervision = st.selectbox("Supervision Level", ["Low","Medium","Medium with Override Consideration","High"])
    sup_num     = st.slider("Supervision Level (numeric)", 1, 4, 1)
    decile      = st.slider("Decile Score", 1, 10, 5)
    raw_score   = st.number_input("Raw Score", min_value=-5.0, max_value=55.0, value=0.0)


if st.button("Predict Eligibility", type="primary"):

    # Encode categorical columns
    sex_map = {
        "Female": 0,
        "Male": 1
    }

    ethnicity_map = {
        "African-American": 0,
        "Asian": 1,
        "Caucasian": 2,
        "Hispanic": 3,
        "Native American": 4,
        "Other": 5
    }

    legal_map = {
        "Conditional Release": 0,
        "Other": 1,
        "Post Sentence": 2,
        "Pretrial": 3,
        "Probation Violator": 4
    }

    custody_map = {
        "Jail Inmate": 0,
        "Pretrial Defendant": 1,
        "Probation": 2,
        "Residential Program": 3
    }

    marital_map = {
        "Divorced": 0,
        "Married": 1,
        "Separated": 2,
        "Significant Other": 3,
        "Single": 4,
        "Widowed": 5
    }

    supervision_map = {
        "High": 0,
        "Low": 1,
        "Medium": 2,
        "Medium with Override Consideration": 3
    }

    # Build dataframe
    input_df = pd.DataFrame([{
        'Sex_Code_Text': sex_map[sex],
        'Ethnic_Code_Text': ethnicity_map[ethnicity],
        'LegalStatus': legal_map[legal],
        'CustodyStatus': custody_map[custody],
        'MaritalStatus': marital_map[marital],
        'RecSupervisionLevel': sup_num,
        'RecSupervisionLevelText': supervision_map[supervision],
        'RawScore': raw_score,
        'DecileScore': decile,
        'Age': age,
    }])

    # Match training column order
    input_df = input_df[model.feature_names_in_]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.divider()

    if prediction == 1:
        st.success("Eligible for Legal Aid")
    else:
        st.error("Not Eligible for Legal Aid")

    st.markdown("#### Prediction probabilities")
    prob_df = pd.DataFrame({
        'Class': ['Not Eligible (Low)', 'Eligible (Medium/High)'],
        'Probability': [
            f"{probability[0]*100:.2f}%",
            f"{probability[1]*100:.2f}%"
        ]
    })

    st.dataframe(prob_df, hide_index=True)