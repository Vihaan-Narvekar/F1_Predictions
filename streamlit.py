import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from predictions import F1RacePredictor
import os
import fastf1

# Set page configuration
st.set_page_config(page_title="F1 Race Winner Predictor", layout="wide")
st.title("🏎️ F1 2025 Race Winner Prediction")
st.markdown("Predict the winner of upcoming races using 2024 qualifying data and weather conditions.")

# Initialize cache directory and predictor
cache_dir = "f1_cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Initialize the predictor
@st.cache_resource
def get_predictor():
    predictor = F1RacePredictor(cache_dir=cache_dir)
    return predictor

predictor = get_predictor()

# Define method to get qualifying times using FastF1 API
def get_qualifying_times(year, gp_name):
    fastf1.Cache.enable_cache("./f1_cache")
    schedule = fastf1.get_event_schedule(year)
    gp_round = schedule[schedule['EventName'].str.contains(gp_name, case=False, na=False)]['RoundNumber']
    
    if gp_round.empty:
        return pd.DataFrame(columns=["Driver", "QualifyingTime"])
    
    session = fastf1.get_session(year, int(gp_round.values[0]), 'Q')
    session.load()
    
    results = []
    for drv in session.drivers:
        driver_laps = session.laps.pick_driver(drv)
        best_lap = driver_laps.pick_fastest()
        if best_lap is not None:
            results.append({"Driver": session.get_driver(drv)["LastName"], "QualifyingTime": best_lap['LapTime'].total_seconds()})
    
    return pd.DataFrame(results)

# Sidebar for user input with better organization
st.sidebar.header("User Input Parameters")
selected_gp = st.sidebar.selectbox(
    "Select Grand Prix", 
    ["Australian", "Bahrain", "Saudi Arabian", "Japanese", "Chinese", "Miami", 
     "Emilia Romagna", "Monaco", "Canadian", "Spanish", "Austrian", "British", 
     "Hungarian", "Belgian", "Dutch", "Italian", "Azerbaijan", "Singapore", 
     "United States", "Mexican", "Brazilian", "Las Vegas", "Qatar", "Abu Dhabi"]
)

# Weather conditions
st.sidebar.subheader("Weather Conditions")
temp = st.sidebar.slider("Temperature (°C)", min_value=-10, max_value=50, value=25)
humidity = st.sidebar.slider("Humidity (%)", min_value=0, max_value=100, value=50)
wind_speed = st.sidebar.slider("Wind Speed (km/h)", min_value=0, max_value=100, value=10)

# Display user selections in a formatted way
st.subheader("Race Details")
st.write(f"**Grand Prix:** {selected_gp}")
st.write(f"**Weather Conditions:** Temperature: {temp}°C, Humidity: {humidity}%, Wind Speed: {wind_speed} km/h")

# Create qualifying data input section
st.subheader("Driver Qualifying Times")
st.markdown("Adjust the qualifying times for each driver using the sliders below.")

# Load historical qualifying data
historical_qualifying_data = get_qualifying_times(2024, selected_gp)

drivers = historical_qualifying_data["Driver"].tolist()


st.markdown("""
<style>
    .qual-time {
        font-size: 24px;
        font-weight: bold;
        color: #1E1E1E;
        background-color: #F0F2F6;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


# Create columns for better layout
cols = st.columns(2)
qual_times = {}

for i, driver in enumerate(drivers):
    col_idx = i % 2
    base_time = historical_qualifying_data.loc[historical_qualifying_data["Driver"] == driver, "QualifyingTime"].values[0]
    percentage_change = cols[col_idx].slider(
        f"Qualifying time change (%)", 
            min_value=-10.0, 
            max_value=10.0, 
            value=0.0,
            step=0.1,
            format="%.1f%%",
            key=f"qual_time_{driver}"
        )
    
    # Calculate the actual qualifying time based on the percentage change
    qual_times[driver] = base_time * (1 + percentage_change / 100)

    # Display the calculated qualifying time
    cols[col_idx].write(f"Qualifying time: {qual_times[driver]:.3f}s")


# Load model button
if st.sidebar.button("Load Saved Model"):
    if predictor.load_model():
        st.sidebar.success("Model loaded successfully!")
    else:
        st.sidebar.error("Failed to load model. Please train a model first.")

st.sidebar.markdown("---")
st.sidebar.subheader("Model Training")
train_model = st.sidebar.checkbox("Train a new model")

if train_model:
    training_gps = st.sidebar.multiselect(
        "Select Grand Prix for training (3-5 recommended)",
        ["Australian", "Bahrain", "Saudi Arabian", "Monaco", "Spanish"],
        default=["Australian", "Bahrain", "Monaco"]
    )
    
    if st.sidebar.button("Train Model"):
        with st.spinner("Training model... This may take a few minutes."):
            predictor.create_driver_mapping(2024, training_gps[0])
            historical_data = predictor.get_historical_data(training_gps)
            
            if historical_data is not None and not historical_data.empty:
                historical_data = predictor.add_weather_data(historical_data)
                model, metrics = predictor.train_model(historical_data)
                
                if predictor.save_model():
                    st.sidebar.success("Model trained and saved successfully!")
                    st.sidebar.write(f"Training MAE: {metrics['train_mae']:.3f} seconds")
                    st.sidebar.write(f"Test MAE: {metrics['test_mae']:.3f} seconds")
                else:
                    st.sidebar.error("Failed to save model")
            else:
                st.sidebar.error("Failed to collect historical data")

# Prediction button
if st.button("🔮 Predict Race Results"):
    if predictor.model is None:
        st.error("Please load a model first using the sidebar button.")
    else:
        qualifying_data = pd.DataFrame({
            "Driver": drivers,
            "QualifyingTime": [float(qual_times[driver]) for driver in drivers]
        })
        
        weather_data = pd.DataFrame({
            "AirTemp": [temp],
            "Humidity": [humidity],
            "WindSpeed": [wind_speed],
            "Pressure": [1013.25]
        })
        
        with st.spinner("Generating predictions..."):
            predictions = predictor.predict_race_results(qualifying_data, weather_data)
        
        if predictions is not None:
            winner = predictions.iloc[0]['Driver']
            st.success(f"🏆 Predicted Race Winner: **{winner}**")
            
            st.subheader("Full Race Predictions")
            st.dataframe(predictions[['PredictedPosition', 'Driver', 'QualifyingTime', 'PredictedLapTime']]
                .rename(columns={
                    'PredictedPosition': 'Position',
                    'QualifyingTime': 'Qualifying Time (s)',
                    'PredictedLapTime': 'Predicted Lap Time (s)'
                }))
            
            # Visualization
            st.subheader("Predicted Lap Times by Driver")
            plt.figure(figsize=(10, 5))
            sns.barplot(x=predictions['Driver'], y=predictions['PredictedLapTime'], palette='coolwarm')
            plt.xticks(rotation=45)
            plt.ylabel("Predicted Lap Time (s)")
            plt.xlabel("Driver")
            st.pyplot(plt)

st.markdown("---")
st.markdown("### About this Model")
st.info("This model uses historical qualifying times and weather conditions from the 2024 season to predict race results for 2025.")
