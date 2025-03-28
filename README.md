F1 Race Predictor

The F1 Race Predictor is a machine learning-based tool designed to predict race outcomes based on historical data, qualifying results, and weather conditions. The model leverages the FastF1 API to fetch race data and uses Histogram-based Gradient Boosting Regression to make predictions.

Features
- Fetches and caches F1 race data using the FastF1 API.
- Processes qualifying times, lap times, and weather data.
- Trains a Histogram-based Gradient Boosting Model to predict lap times.
- Saves and loads trained models for future predictions.
- Provides a driver name mapping for ease of use.
- Evaluates model performance using metrics like MAE, MSE, and R².

The model is based on historical F1 data, which means predictions depend on the accuracy and completeness of the data.
Performance can improve with additional data sources, better feature engineering, and model tuning.

**Note: The project will require you to locally load data before you create models and predictions.**

