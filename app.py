import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Function to load the trained model
@st.cache
def load_model():
    model_path = os.path.join(./models/decision_tree_model.pkl")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Load the model
model = load_model()

# Title of the app
st.title('Heart Disease Prediction')

# Input widgets for user input
st.sidebar.header('User Input Features')

# Function to get user input
def get_user_input():
    age = st.sidebar.slider('Age', 29, 77, 50)
    sex = st.sidebar.selectbox('Sex', ['male', 'female'])
    cp = st.sidebar.selectbox('Chest Pain Type', [0, 1, 2, 3])
    trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 94, 200, 130)
    chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 126, 564, 240)
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved (bpm)', 71, 202, 150)
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise', 0.0, 6.2, 1.0)
    ca = st.sidebar.selectbox('Number of Major Vessels', [0, 1, 2, 3])
    
    # Map sex to numerical values
    sex_mapping = {'male': 1, 'female': 0}
    sex_val = sex_mapping[sex]
    
    # Create a dictionary of user input features
    user_data = {'age': age,
                 'sex': sex_val,
                 'cp': cp,
                 'trestbps': trestbps,
                 'chol': chol,
                 'thalach': thalach,
                 'oldpeak': oldpeak,
                 'ca': ca}
    
    # Convert the dictionary to a Pandas DataFrame
    features = pd.DataFrame(user_data, index=[0])
    return features

# Get user input
user_input = get_user_input()

# Display the user input features
st.subheader('User Input Features')
st.write(user_input)

# Make predictions
prediction = model.predict(user_input)
prediction_proba = model.predict_proba(user_input)

# Display the prediction
st.subheader('Prediction')
target = np.array(['No Disease', 'Disease'])
st.write(target[prediction])

# Display the prediction probabilities
st.subheader('Prediction Probability')
st.write(prediction_proba)
