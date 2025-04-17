import streamlit as st
import joblib
import numpy as np

# Load the trained XGBClassifier model
model = joblib.load('xgbclassifier_model.pkl')

def main():
    st.title('🏨 Hotel Booking Cancellation Prediction')

    st.write("### Booking Details")

    # Basic numerical inputs
    no_of_adults = st.slider('Number of Adults', 1, 10, 1)
    no_of_children = st.slider('Number of Children', 0, 10, 0)
    no_of_weekend_nights = st.slider('Weekend Nights', 0, 10, 0)
    no_of_week_nights = st.slider('Week Nights', 0, 20, 0)
    meal_plan = st.selectbox("Meal Plan", ['Not Selected', 'Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3'])
    
    meal_one_hot = [
        1 if meal_plan == 'Meal Plan 1' else 0,
        1 if meal_plan == 'Meal Plan 2' else 0,
        1 if meal_plan == 'Meal Plan 3' else 0,
        1 if meal_plan == 'Not Selected' else 0
    ]    
    
    required_car_parking_space = st.selectbox('Car Parking Required?', [0, 1])
    
    room_type = st.selectbox("Room Type Reserved", [
        'Room_Type 1', 'Room_Type 2', 'Room_Type 3',
        'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'
    ])
    room_one_hot = [1 if room_type == f'Room_Type {i}' else 0 for i in range(1, 8)]
    
    lead_time = st.slider('Lead Time (days)', 0, 700, 0)
    arrival_year = st.selectbox('Arrival Year', [2017, 2018], index=0)
    arrival_month = st.slider('Arrival Month', 1, 12, 1)
    arrival_date = st.slider('Arrival Date', 1, 31, 1)    
    
    market_segment = st.selectbox("Market Segment", [
        'Offline', 'Online', 'Corporate', 'Complementary', 'Aviation'
    ])
    market_one_hot = [
        1 if market_segment == 'Offline' else 0,
        1 if market_segment == 'Online' else 0,
        1 if market_segment == 'Corporate' else 0,
        1 if market_segment == 'Complementary' else 0,
        1 if market_segment == 'Aviation' else 0
    ]
    
    repeated_guest = st.selectbox('Repeated Guest?', [0, 1])
    no_of_previous_cancellations = st.slider('Previous Cancellations', 0, 10, 0)
    no_of_previous_bookings_not_canceled = st.slider('Previous Non-Canceled Bookings', 0, 50, 0)
    avg_price_per_room = st.slider('Average Price per Room', 0.0, 500.0, 0.0)
    no_of_special_requests = st.slider('Number of Special Requests', 0, 5, 0)

    if st.button('Make Prediction'):
        features = [
            no_of_adults,
            no_of_children,
            no_of_weekend_nights,
            no_of_week_nights,
            *meal_one_hot,
            required_car_parking_space,
            *room_one_hot,
            lead_time,
            arrival_year,
            arrival_month,
            arrival_date,
            *market_one_hot,
            repeated_guest,
            no_of_previous_cancellations,
            no_of_previous_bookings_not_canceled,
            avg_price_per_room,
            no_of_special_requests
        ]

        prediction = model.predict(np.array(features).reshape(1, -1))[0]
        proba = model.predict_proba(np.array(features).reshape(1, -1))[0]
        threshold = 0.3  # Try 0.3 or 0.4

        label = "Cancelled" if proba[1] > threshold else "Not Cancelled"

        st.write(f"🔍 Probabilities: Cancelled = {proba[1]:.2f}, Not Cancelled = {proba[0]:.2f}")
        st.success(f'🔍 Prediction (threshold={threshold}): **{label}**')

if __name__ == '__main__':
    main()
