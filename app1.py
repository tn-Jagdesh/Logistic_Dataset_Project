import joblib
import pandas as pd
import streamlit as st

# Set the page config
st.set_page_config(
    page_title="Time Duration Predictor",
    page_icon="🕒",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for theming
st.markdown(
    """
    <style>
    body {
        background-color: white;
        color: red;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom CSS for autoplant logo in top-left corner
st.markdown(
    """
    <style>
    .autoplant-logo {
        position: absolute;
        top: -175px;
        left: -300px;
        color: red;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

New_Logistic_dataset = pd.read_csv('New_Logistic_dataset_scaled.csv')
Stacked_Model = joblib.load('Logistic_Data_StackingRegressor.pkl')

st.title('Time Duration Predictor')

# Autoplant logo in top-left corner
st.markdown('<div class="autoplant-logo">autoplant</div>', unsafe_allow_html=True)

SOURCE = st.selectbox('Source', New_Logistic_dataset.SOURCE.unique())

filtered_df = New_Logistic_dataset[New_Logistic_dataset.SOURCE == SOURCE]

DESTINATION = st.selectbox('Destination', filtered_df['DESTINATION'])

if st.button('Predict Time Duration'):
    def convert_duration(duration_seconds):
        hours = duration_seconds // 3600
        minutes = (duration_seconds % 3600) // 60
        return f'{int(hours)} hrs {int(minutes)} min'

    def Stacked_Model1(A, B):
        Input = New_Logistic_dataset[(New_Logistic_dataset['SOURCE'] == A) & (
                New_Logistic_dataset['DESTINATION'] == B)].iloc[:, :10]
        Predict_Duration = Stacked_Model.predict(Input)
        return Predict_Duration[0]

    predicted_duration = Stacked_Model1(SOURCE, DESTINATION)
    predicted_duration_formatted = convert_duration(predicted_duration)

    # Font size in pixels
    font_size = 24

    # Adjust the font size using Markdown syntax
    st.markdown(
        f"<h1 style='font-size: {font_size}px;'>The Time taken to travel from {SOURCE} to {DESTINATION} is {predicted_duration_formatted}</h1>",
        unsafe_allow_html=True)
