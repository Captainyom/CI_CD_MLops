import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the app
def main():
    st.title("Wine Quality Prediction")
    st.write("Enter the features to get the wine quality prediction.")

    # Input features
    feature_names = [
        'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
        'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
        'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'
    ]

    user_input = {}
    for feature in feature_names:
        user_input[feature] = st.number_input(f'Enter {feature}', value=0.0, format="%.2f")

    # Convert user input into a dataframe
    input_df = pd.DataFrame([user_input])

    # Make prediction
    if st.button("Predict"):
        prediction = model.predict(input_df)
        st.write(f"The predicted wine quality is: {prediction[0]}")

# Run the app
if __name__ == '__main__':
    main()
