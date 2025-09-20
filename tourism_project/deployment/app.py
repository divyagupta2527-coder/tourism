import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="divyagupta2527-coder/tourism", filename="Tourism_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("TOURISM PACKAGE App")
st.write("""
This application predicts the product taken or not.
""")

# User input
TypeofContact = st.selectbox("TypeofContact", ["selfEnquiry","Company Invited"])
Age = st.number_input("Age", min_value=18.0, max_value=65.0, value=20.0, step=0.1)
CityTier = st.selectbox("CityTier", ['1','2','3'])
Occupation = st.selectbox("Occupation)", ['Salaried','Small Bussiness','Large Bussiness','FreeLancer'])
Gender = st.selectbox("Gender", ['male','Female'])
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisting", min_value=1, max_value=5, value=1)
PreferredPropertyStar = st.number_input("PreferredPropertyStar", min_value=1, max_value=5, value=1)
NumberOfTrips = st.number_input("NumberOfTrips", min_value=1, max_value=5, value=1)
Passport = st.selectbox("Passport", ['Yes','No'])
OwnCar = st.selectbox("OwnCar", ['Yes','No'])
NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=5, value=0)
NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=1, max_value=6, value=1)
PitchSatisfactionScore = st.number_input("PitchSatisfactionScore", min_value=1, max_value=5, value=1)
DurationOfPitch = st.number_input("DurationOfPitch", min_value=1, max_value=20, value=1)
Designation = st.selectbox("Designation", ['Executive','Manager','Senior Manager','AVP','VP'])
MonthlyIncome=st.number_input("MonthlyIncome",min_value=1000,max_value=100000,value=1000)
PitchedStatisfactionScore=st.number_input("PitchedStatisfactionScore",min_value=1,max_value=5,value=1)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'TypeofContact': TypeofContact,
    'Age': Age,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'DurationOfPitch': DurationOfPitch,
    'Designation': Designation,
    'MonthlyIncome':MonthlyIncome,
    'PitchedStatisfactionScore':
}])


if st.button("Predict Taken"):
    prediction = model.predict(input_data)[0]
    result = "taken" if prediction == 1 else "Not Taken"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
