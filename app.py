import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
try:
    with open('best_model_ML_Classification.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'best_model_ML_Classification.pkl' not found. Please ensure the model file is in the same directory as the script.")
    st.stop()

# Define the app
def main():
    st.title("Mule Account Prediction")
    st.write("Enter the features to get the Mule Account prediction.")

    # Define numerical features
    numerical_features = [
        'BAL_ACCT_MIN_REQD', 'N_TXN_AMT', 'BAL_LAST_STMNT',
        'COUNT_TXN_PAST_3_DAYS_DEBIT', 'COUNT_TXN_PAST_4_DAYS_CREDIT',
        'SUM_TXN_PAST_6_DAYS_DEBIT', 'SUM_ATM_TXN_PAST_6_DAYS',
        'COUNT_ATM_TXN_PAST_7TO30DAYS', 'COUNT_TXN_PAST_HOURS',
        'SUM_TXN_PAST_HOURS', 'COUNT_TXN_PAST_HOURS_DEBIT',
        'SUM_TXN_PAST_HOURS_DEBIT', 'COUNT_TXN_PAST_HOURS_CREDIT',
        'SUM_TXN_PAST_HOURS_CREDIT', 'DIGIT_SUM', 'AVG_DIGIT_SUM', 'NUM_DIGIT',
        'AVG_NUM_DIGIT', 'AGE_OF_ACCT', 'AGE_OF_CUSTOMER',
        '7D_1D_CR_1T_PARTIES', '14D_7D_CR_1T_PARTIES', '7D_1D_CR_5T_PARTIES'
    ]
    
    # Define categorical features and their respective values
    categorical_features = {
        'TYPE_OF_TXN': ['CREDIT', 'DEBIT'],
        'YR_OF_JOINING': ['2024', '2023', '2022', '2021', '2020'],
        'CHEQ_ENABLED': ['N', 'Y'],
        'PASSBOOK': ['N', 'Y'],
        'BHIM_QR': ['N', 'Y'],
        'IB_REG': ['N', 'Y'],
        'DB_FLG': ['N', 'Y'],
        'VALID_MB': ['N', 'Y'],
        'AGE_OF_ACCOUNT': ['0', '1', '2', '3', '4', '5', '6', '7', 'REST'],
        'STATES': ['Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana',
                   'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 
                   'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana',
                   'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal']
    }

    # Create input fields for numerical features
    user_input = {}
    for feature in numerical_features:
        user_input[feature] = st.number_input(f'Enter {feature}', value=0.0)

    # Create dropdowns for categorical features
    for feature, options in categorical_features.items():
        user_input[feature] = st.selectbox(f'Select {feature}', options)

    # Convert user input into a dataframe
    input_df = pd.DataFrame([user_input])

    # Make prediction
    if st.button("Predict"):
        try:
            # Convert categorical data to numerical data using one-hot encoding
            input_df_encoded = pd.get_dummies(input_df)

            # Define model columns (this should match the columns used when training the model)
            model_columns = [
                'BAL_ACCT_MIN_REQD', 'N_TXN_AMT', 'BAL_LAST_STMNT',
                'COUNT_TXN_PAST_3_DAYS_DEBIT', 'COUNT_TXN_PAST_4_DAYS_CREDIT',
                'SUM_TXN_PAST_6_DAYS_DEBIT', 'SUM_ATM_TXN_PAST_6_DAYS',
                'COUNT_ATM_TXN_PAST_7TO30DAYS', 'COUNT_TXN_PAST_HOURS',
                'SUM_TXN_PAST_HOURS', 'COUNT_TXN_PAST_HOURS_DEBIT',
                'SUM_TXN_PAST_HOURS_DEBIT', 'COUNT_TXN_PAST_HOURS_CREDIT',
                'SUM_TXN_PAST_HOURS_CREDIT', 'DIGIT_SUM', 'AVG_DIGIT_SUM', 'NUM_DIGIT',
                'AVG_NUM_DIGIT', 'AGE_OF_ACCT', 'AGE_OF_CUSTOMER',
                '7D_1D_CR_1T_PARTIES', '14D_7D_CR_1T_PARTIES', '7D_1D_CR_5T_PARTIES',
                'TYPE_OF_TXN_CREDIT', 'TYPE_OF_TXN_DEBIT', 'YR_OF_JOINING_2020',
                'YR_OF_JOINING_2021', 'YR_OF_JOINING_2022', 'YR_OF_JOINING_2023',
                'YR_OF_JOINING_2024', 'CHEQ_ENABLED_N', 'CHEQ_ENABLED_Y', 'PASSBOOK_N',
                'PASSBOOK_Y', 'BHIM_QR_N', 'BHIM_QR_Y', 'IB_REG_N', 'IB_REG_Y',
                'DB_FLG_N', 'DB_FLG_Y', 'VALID_MB_N', 'VALID_MB_Y', 'AGE_OF_ACCOUNT_0',
                'AGE_OF_ACCOUNT_1', 'AGE_OF_ACCOUNT_2', 'AGE_OF_ACCOUNT_3',
                'AGE_OF_ACCOUNT_4', 'AGE_OF_ACCOUNT_5', 'AGE_OF_ACCOUNT_6',
                'AGE_OF_ACCOUNT_7', 'AGE_OF_ACCOUNT_REST', 'STATES_Andhra Pradesh',
                'STATES_Arunachal Pradesh', 'STATES_Assam', 'STATES_Bihar',
                'STATES_Chhattisgarh', 'STATES_Goa', 'STATES_Gujarat', 'STATES_Haryana',
                'STATES_Himachal Pradesh', 'STATES_Jharkhand', 'STATES_Karnataka',
                'STATES_Kerala', 'STATES_Madhya Pradesh', 'STATES_Maharashtra',
                'STATES_Manipur', 'STATES_Meghalaya', 'STATES_Mizoram',
                'STATES_Nagaland', 'STATES_Odisha', 'STATES_Punjab', 'STATES_Rajasthan',
                'STATES_Sikkim', 'STATES_Tamil Nadu', 'STATES_Telangana',
                'STATES_Tripura', 'STATES_Uttar Pradesh', 'STATES_Uttarakhand',
                'STATES_West Bengal'
            ]

            # Ensure the encoded input has the same columns as the training data
            input_df_encoded = input_df_encoded.reindex(columns=model_columns, fill_value=0)

            # Predict using encoded data
            prediction = model.predict(input_df_encoded)
            st.write(f"The predicted Mule Account is: {prediction[0]}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Run the app
if __name__ == '__main__':
    main()
