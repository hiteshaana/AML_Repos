
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="wide"
)

# ==========================================================
# Load Model
# ==========================================================

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id="hiteshsharma/tourism-prediction-model",
        filename="best_tourism_prediction_model_v1.joblib",
        repo_type="model"
    )

    return joblib.load(model_path)

model = load_model()

# ==========================================================
# Application Title
# ==========================================================

st.title("✈️ Tourism Package Prediction")

st.write(
    """
Predict whether a customer is likely to purchase a tourism package.
Enter the customer details below and click **Predict Package Purchase**.
"""
)

# ==========================================================
# Customer Inputs
# ==========================================================

col1, col2 = st.columns(2)

with col1:

    age = st.number_input("Age",18,100,35)

    type_contact = st.selectbox(
        "Type of Contact",
        ["Self Enquiry","Company Invited"]
    )

    city_tier = st.selectbox(
        "City Tier",
        [1,2,3]
    )

    duration_pitch = st.number_input(
        "Duration Of Pitch",
        1,
        100,
        15
    )

    occupation = st.selectbox(
        "Occupation",
        [
            "Salaried",
            "Small Business",
            "Large Business",
            "Free Lancer"
        ]
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

    persons = st.number_input(
        "Number Of Persons Visiting",
        1,
        10,
        2
    )

    followups = st.number_input(
        "Number Of Followups",
        0,
        10,
        2
    )

    product = st.selectbox(
        "Product Pitched",
        [
            "Basic",
            "Standard",
            "Deluxe",
            "Super Deluxe",
            "King"
        ]
    )

with col2:

    property_star = st.selectbox(
        "Preferred Property Star",
        [1,2,3,4,5]
    )

    marital = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )

    trips = st.number_input(
        "Number Of Trips",
        0,
        30,
        2
    )

    passport = st.selectbox(
        "Passport",
        [0,1]
    )

    pitch_score = st.slider(
        "Pitch Satisfaction Score",
        1,
        5,
        3
    )

    own_car = st.selectbox(
        "Own Car",
        [0,1]
    )

    children = st.number_input(
        "Number Of Children Visiting",
        0,
        10,
        0
    )

    designation = st.selectbox(
        "Designation",
        [
            "Executive",
            "Manager",
            "Senior Manager",
            "AVP",
            "VP"
        ]
    )

    income = st.number_input(
        "Monthly Income",
        1000,
        1000000,
        50000
    )

# ==========================================================
# Prediction
# ==========================================================

if st.button("Predict Package Purchase"):

    input_df = pd.DataFrame({

        "Age":[age],
        "TypeofContact":[type_contact],
        "CityTier":[city_tier],
        "DurationOfPitch":[duration_pitch],
        "Occupation":[occupation],
        "Gender":[gender],
        "NumberOfPersonVisiting":[persons],
        "NumberOfFollowups":[followups],
        "ProductPitched":[product],
        "PreferredPropertyStar":[property_star],
        "MaritalStatus":[marital],
        "NumberOfTrips":[trips],
        "Passport":[passport],
        "PitchSatisfactionScore":[pitch_score],
        "OwnCar":[own_car],
        "NumberOfChildrenVisiting":[children],
        "Designation":[designation],
        "MonthlyIncome":[income]

    })

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.success(
            f"""
            Customer is likely to purchase the tourism package.

            Probability : {probability:.2%}
            """
        )

    else:

        st.error(
            f"""
            Customer is unlikely to purchase the tourism package.

            Probability : {probability:.2%}
            """
        )

    st.subheader("Input Summary")

    st.dataframe(input_df)
