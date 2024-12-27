import streamlit as st
import pandas as pd
import numpy as np
import joblib


model = joblib.load("db_model.pkl")
standard_scaler = joblib.load("db_standard_scaler.pkl")

st.set_page_config(page_title="Diabetes Checker", layout="wide")

st.title("Diabetes Checker")

st.write("Kindly answer the questions below:")

features = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age']

highbp_mapping = {"Yes":1, "No":0}
input_HighBP = st.selectbox("Do you have a high blood pressure?", ["Yes", "No"])
HighBP = highbp_mapping[input_HighBP]

highchol_mapping = {"Yes":1, "No":0}
input_HighChol = st.selectbox("Do you have high cholesterol?", ["Yes", "No"])
HighChol = highchol_mapping[input_HighChol]

cholcheck_mapping = {"Yes":1, "No":0}
input_CholCheck = st.selectbox("Have you checked your cholesterol levels in the last 5 years?", ["Yes", "No"])
CholCheck = cholcheck_mapping[input_CholCheck]

BMI = st.number_input("What is your BMI?", 10, 300)

smoker_mapping = {"Yes":1, "No":0}
input_Smoker = st.selectbox("Have you smoked at least 100 cigarettes in your life?", ["Yes", "No"])
Smoker = smoker_mapping[input_Smoker]

stroke_mapping = {"Yes":1, "No":0}
input_Stroke = st.selectbox("Have you ever had stroke before?", ["Yes", "No"])
Stroke = stroke_mapping[input_Stroke]

heartdis_mapping = {"Yes":1, "No":0}
input_HeartDiseaseorAttack = st.selectbox("Do you have coronary heart disease or myocardial infection?", ["Yes", "No"])
HeartDiseaseorAttack = heartdis_mapping[input_HeartDiseaseorAttack]

physact_mapping = {"Yes":1, "No":0}
input_PhysActivity = st.selectbox("Have you involved yourself in physical activity in the past 30 days?", ["Yes", "No"])
PhysActivity = physact_mapping[input_PhysActivity]

fruits_mapping = {"Yes":1, "No":0}
input_Fruits = st.selectbox("Do you consume fruits one or more times per day?", ["Yes", "No"])
Fruits = fruits_mapping[input_Fruits]

veggies_mapping = {"Yes":1, "No":0}
input_Veggies = st.selectbox("Do you consume vegetables one or more times per day?", ["Yes", "No"])
Veggies = veggies_mapping[input_Veggies]

hvyalcohol_mapping = {"Yes":1, "No":0}
input_HvyAlcoholConsump = st.selectbox("Do you consider yourself a heavy drinker?(adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)", ["Yes", "No"])
HvyAlcoholConsump = hvyalcohol_mapping[input_HvyAlcoholConsump]

genhlth_mapping = {"Excellent":1, "Very good":2, "Good":3, "Fair":4, "Poor":5}
input_GenHlth = st.selectbox("What would you say your general health is?", ["Excellent", "Very good", "Good", "Fair", "Poor"])
GenHlth = genhlth_mapping[input_GenHlth]

MentHlth = st.slider("Thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good? scale 1-30 days", 0, 30)

PhysHlth = st.slider("Thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? scale 1-30 days", 0, 30)

diffwalk_mapping = {"Yes":1, "No":0}
input_DiffWalk = st.selectbox("Do you have serious difficulty walking or climbing stairs?", ["Yes", "No"])
DiffWalk = diffwalk_mapping[input_DiffWalk]

sex_mapping = {"Male":1, "Female":0}
input_Sex = st.selectbox("What is your gender", ["Male", "Female"])
Sex = sex_mapping[input_Sex]

age_mapping = {"18-24":1, "25-29":2, "30-34":3, "35-39":4, "40-44":5, "45-49":6, "50-54":7, "55-59":8, "60-64":9, "65-69":10, "70-74":11, "75-79":12, "80 or older":13}
input_Age = st.selectbox("Which age category do you fall in?", ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"])
Age = age_mapping[input_Age]  

features = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age']

if st.button("Check"):
    input_data = np.array([[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age]])

    input_df = pd.DataFrame(input_data, columns=features)

    scaled_input = standard_scaler.transform(input_df)

    predict_mapping = {
        0 : "You are most likely not having diabetes", 
        1 : "You are most likely having diabetes. Go to the hospital for further checkup"}

    prediction = model.predict(scaled_input)
    encoded_prediction = predict_mapping[int(prediction[0])]

    st.write(encoded_prediction)
    st.write(f"Model is 86.5% accurate")