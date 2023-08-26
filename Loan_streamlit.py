import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split


from sklearn.linear_model import LogisticRegression


import seaborn as sns
import matplotlib.pyplot as plt


import streamlit as st
import joblib

st.title('Will you be accepted or rejected for the loan rent?')
Loan_LR_Model = joblib.load("Loan_LR_Model.pkl")

image = 'Admission.jpg'  # Replace 'path_to_image.jpg' with the actual path to your image file
st.image(image, caption='Image from https://www.indoindians.com')
#Image by Đỗ Thiệp from Pixabay
name = st.text_input('Enter your name', '')

if name:

    st.success(f" **Hi {name}!This prediction is just for fun. For prediction, Please provide job status, your annual income,loan amount,loan term and cibil scores**")
    

    dependents = st.number_input('Enter your dependents', min_value=0, max_value=5, value=0, step=1)
    # Create radio buttons
    education= st.radio("Are you graduated?", ["Graduate","Not Graduate"])
    edu = 0 if education =="No" else 1
    employed= st.radio("Are you self-employed?", ["Yes","No"])
    e = 0 if employed =="No" else 1
    
    
    income = st.number_input('Enter your annual income', min_value=2000, max_value=10000000, value=200000, step=10000)
    loan_amount = st.number_input('Enter your loan amount', min_value=6000, max_value=50000000, value=600000, step=10000)
    loan_term = st.number_input('Enter your loan term', min_value=2, max_value=20, value=2, step=1)
    score = st.number_input('Enter your credit score', min_value=300, max_value=1000, value=300, step=1)
    # resident = st.number_input('Enter your resident asset amount', min_value=600000, max_value=500000000, value=600000, step=10000)
    # commercial = st.number_input('Enter your commercial asset amount', min_value=600000, max_value=500000000, value=600000, step=10000)
    # luxury = st.number_input('Enter your luxury asset amount', min_value=600000, max_value=500000000, value=600000, step=10000)
    # bank = st.number_input('Enter your bank asset amount', min_value=600000, max_value=500000000, value=600000, step=10000)

    data=[dependents,edu,e,income,loan_amount,loan_term,score]
    
    result=Loan_LR_Model.predict([data])
    
    button_clicked = st.button("Predict")
    if button_clicked:
        
        if result==0:
            st.success(f"**Based on the available data, {name} !Your loan will be rejected **")
        else:
            st.success(f"**Based on the available data, {name} !Your loan will be accepted **")





    