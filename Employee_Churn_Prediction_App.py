import streamlit as st
import pandas as pd
from joblib import load
import dill
from datetime import datetime

# Load the pretrained model
with open('pipeline.pkl', 'rb') as file:
    model = dill.load(file)

my_feature_dict = load('my_feature_dict.pkl')

# Function to predict churn
def predict_churn(data):
    prediction = model.predict(data)
    return prediction

# Styling function
def style_text(text, tag="h1", color="navy", align="center"):
    return f"<{tag} style='color:{color}; text-align:{align};'>{text}</{tag}>"

# Add custom CSS
st.markdown(
    """
    <style>
    .stButton > button {
        background-color: navy;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title and Subheader with Styling
st.markdown(style_text('Employee Churn Prediction App'), unsafe_allow_html=True)
st.markdown(style_text('Based on Employee Churn Dataset', tag="h3"), unsafe_allow_html=True)
st.markdown(style_text('Created by Hina Naz', tag="p", color="gray"), unsafe_allow_html=True)

st.divider()

st.markdown(style_text('Fill the form for the Prediction', tag="h3"), unsafe_allow_html=True)

# Display categorical features
categorical_input = my_feature_dict.get('CATEGORICAL')
categorical_input_vals = {}

for i, col in enumerate(categorical_input.get('Column Name').values()):
    # Add placeholder text as the first option
    options = ['KIndly select relevant option'] + categorical_input.get('Members')[i]
    selected_value = st.selectbox(col, options)

    # Ensure valid selection
    if selected_value == options[0]:
        categorical_input_vals[col] = None  # Or set a default value if needed
    else:
        categorical_input_vals[col] = selected_value

# Display numerical features
numerical_input = my_feature_dict.get('NUMERICAL')
numerical_input_vals = {}

# 1. Joining Year with slider
current_year = datetime.now().year
numerical_input_vals['JoiningYear'] = st.slider(
    'Joining Year', 
    min_value=2012, 
    max_value=current_year, 
    value=2015
)

# 2. Second numerical variable with dropdown (Assuming the second numerical variable is 'SecondVariableName')
second_var_name = numerical_input.get('Column Name')[1]  # Adjust the index if needed
numerical_input_vals[second_var_name] = st.selectbox(
    second_var_name, 
    ['Select your ' + second_var_name, '1', '2', '3']  # Replace with actual options
)

# 3. Age starting from 18
numerical_input_vals['Age'] = st.number_input(
    'Age', 
    min_value=18, 
    max_value=100,  # Set a reasonable upper limit
    value=25
)

# 4. Experience in Current Domain with slider
numerical_input_vals['ExperienceInCurrentDomain'] = st.slider(
    'Experience in Current Domain', 
    min_value=0, 
    max_value=10, 
    value=2
)

# Combine numerical and categorical input dicts
input_data = {**categorical_input_vals, **numerical_input_vals}

# Convert to DataFrame
input_data = pd.DataFrame([input_data])

# Optional: Display input data for debugging
st.write("Input Data:", input_data)

# Churn Prediction
if st.button('Predict'):
    try:
        # Ensure no placeholder values are passed to the model
        if None in input_data.values.flatten():
            st.error("Please fill in all fields before predicting.")
        else:
            prediction = predict_churn(input_data)[0]
            translation_dict = {"Yes": "Expected", "No": "Not Expected"}
            prediction_translate = translation_dict.get(prediction, "Unknown")
            st.write(f'The Prediction is **{prediction}**, Hence employee is **{prediction_translate}** to churn.')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
