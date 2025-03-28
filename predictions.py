import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class F1RacePredictor:
    def __init__(self, cache_dir="f1_cache"):
        """Initialize the F1 race predictor with caching enabled."""
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        fastf1.Cache.enable_cache(cache_dir)
        self.model = None
        self.driver_mapping = {}
        self.reverse_mapping = {}
        self.model_metrics = {}
    
    def create_driver_mapping(self, year, gp, session_type="Q"):
        """Create mapping between driver names and codes."""
        try:
            session = fastf1.get_session(year, gp, session_type)
            session.load()
            
            # Create mapping from full names to driver codes
            drivers_info = session.results[['FullName', 'Abbreviation']].copy()
            self.driver_mapping = dict(zip(drivers_info['FullName'], drivers_info['Abbreviation']))
            self.reverse_mapping = dict(zip(drivers_info['Abbreviation'], drivers_info['FullName']))
            return True
        except Exception as e:
            print(f"Error creating driver mapping: {e}")
            return False
    
    def get_historical_data(self, gps, year=2024, qualifying_session="Q", race_session="R"):
        """Collect historical data from specified races in a single year."""
        all_qualifying_data = []
        all_race_data = []
        
        for gp in gps:
            try:
                # Get qualifying data
                qual_session = fastf1.get_session(year, gp, qualifying_session)
                qual_session.load()
                qualifying_results = qual_session.results[['FullName', 'Abbreviation', 'Q1', 'Q2', 'Q3']].copy()
                
                # Convert time strings to seconds
                for col in ['Q1', 'Q2', 'Q3']:
                    qualifying_results[f'{col}_seconds'] = qualifying_results[col].apply(
                        lambda x: x.total_seconds() if pd.notna(x) else np.nan
                    )
                
                # Get best qualifying time
                qualifying_results['BestQualTime'] = qualifying_results[['Q1_seconds', 'Q2_seconds', 'Q3_seconds']].min(axis=1)
                qualifying_results['Year'] = year
                qualifying_results['GP'] = gp
                
                # Get race data
                race_session_obj = fastf1.get_session(year, gp, race_session)
                race_session_obj.load()
                
                # Get lap times
                laps = race_session_obj.laps.copy()
                laps = laps[['Driver', 'LapTime', 'LapNumber', 'Stint', 'Compound']]
                laps.dropna(subset=['LapTime'], inplace=True)
                laps['LapTime_seconds'] = laps['LapTime'].dt.total_seconds()
                
                # Calculate median lap time for each driver (excluding outliers)
                driver_lap_stats = laps.groupby('Driver').agg({
                    'LapTime_seconds': lambda x: np.median(x[x < np.percentile(x, 95)]),  # Exclude top 5% slowest laps
                    'Compound': lambda x: x.mode().iloc[0] if not x.mode().empty else None
                }).reset_index()
                
                driver_lap_stats['Year'] = year
                driver_lap_stats['GP'] = gp
                


                all_qualifying_data.append(qualifying_results)
                all_race_data.append(driver_lap_stats)
                
                print(f"Successfully processed {year} {gp}")
            except Exception as e:
                print(f"Error processing {year} {gp}: {e}")
        print(qualifying_results.columns)
        
        # Combine all data
        if all_qualifying_data and all_race_data:
            qualifying_df = pd.concat(all_qualifying_data, ignore_index=True)
            race_df = pd.concat(all_race_data, ignore_index=True)
            
            # Merge qualifying and race data
            merged_data = qualifying_df.merge(
                race_df,
                left_on=['Abbreviation', 'Year', 'GP'],
                right_on=['Driver', 'Year', 'GP'],
                how='inner'
            )
            
            return merged_data
        else:
            return None
    
    def add_weather_data(self, data):
        """Add weather data to the dataset if available."""
        try:
            for idx, row in data.iterrows():
                year, gp = row['Year'], row['GP']
                session = fastf1.get_session(year, gp, 'R')
                session.load()
                
                if hasattr(session, 'weather_data'):
                    weather = session.weather_data
                    if not weather.empty:
                        # Get average weather conditions
                        avg_temp = weather['AirTemp'].mean()
                        avg_humidity = weather['Humidity'].mean()
                        avg_pressure = weather['Pressure'].mean()
                        avg_wind_speed = weather['WindSpeed'].mean()
                        
                        # Add to dataset
                        data.loc[idx, 'AirTemp'] = avg_temp
                        data.loc[idx, 'Humidity'] = avg_humidity
                        data.loc[idx, 'Pressure'] = avg_pressure
                        data.loc[idx, 'WindSpeed'] = avg_wind_speed
            
            # Fill missing weather data with medians
            for col in ['AirTemp', 'Humidity', 'Pressure', 'WindSpeed']:
                if col in data.columns:
                    data[col].fillna(data[col].median(), inplace=True)
            
            return data
        except Exception as e:
            print(f"Error adding weather data: {e}")
            return data
    
    def train_model(self, data, features=None, target='LapTime_seconds', test_size=0.2, random_state=42):
        """Train a histogram-based gradient boosting model on the provided data."""
        if features is None:
            features = ['BestQualTime']
            
            # Add weather features if available
            weather_features = ['AirTemp', 'Humidity', 'Pressure', 'WindSpeed']
            for feature in weather_features:
                if feature in data.columns:
                    features.append(feature)
        
        # Prepare data
        X = data[features].copy()
        y = data[target].copy()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Train model
        model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=2,
            l2_regularization=0.1,
            max_bins=255,
            random_state=random_state
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Store model and metrics
        self.model = model
        self.model_metrics = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'features': features
        }
        
        return model, self.model_metrics
    
    def save_model(self, filepath='f1_race_predictor_model.pkl'):
        """Save the trained model to a file."""
        if self.model is not None:
            model_data = {
                'model': self.model,
                'metrics': self.model_metrics,
                'driver_mapping': self.driver_mapping,
                'reverse_mapping': self.reverse_mapping,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            return True
        return False
    
    def load_model(self, filepath='f1_race_predictor_model.pkl'):
        """Load a trained model from a file."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.model_metrics = model_data['metrics']
            self.driver_mapping = model_data['driver_mapping']
            self.reverse_mapping = model_data['reverse_mapping']
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_race_results(self, qualifying_data, weather_data=None):
        """Predict race results based on qualifying data and optional weather data."""
        if self.model is None:
            print("Model not trained or loaded. Please train or load a model first.")
            return None
        
        # Prepare features for prediction
        features = self.model_metrics['features']
        X_pred = pd.DataFrame()
        
        # Add qualifying time
        if 'BestQualTime' in features:
            X_pred['BestQualTime'] = qualifying_data['QualifyingTime']
        
        # Add weather features if available and required
        weather_features = ['AirTemp', 'Humidity', 'Pressure', 'WindSpeed']
        if weather_data is not None:
            for feature in weather_features:
                if feature in features and feature in weather_data:
                    X_pred[feature] = weather_data[feature]
        
        # Make predictions
        predicted_lap_times = self.model.predict(X_pred)
        
        # Create results dataframe
        results = qualifying_data.copy()
        results['PredictedLapTime'] = predicted_lap_times
        
        # Calculate predicted race time (assuming 58 laps for a typical race)
        # This is simplified - real races have varying lap counts
        results['PredictedRaceTime'] = results['PredictedLapTime'] * 58
        
        # Sort by predicted race time
        results = results.sort_values('PredictedRaceTime')
        
        # Add positions
        results['PredictedPosition'] = range(1, len(results) + 1)
        
        return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='F1 Race Predictor')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--predict', action='store_true', help='Make predictions using a trained model')
    parser.add_argument('--gps', nargs='+', type=str, default=['Australian', 'Bahrain', 'Saudi Arabian', 'Japanese', 'Chinese', 'Miami', 'Emilia Romagna', 'Monaco', 'Canadian', 'Spanish', 'Austrian', 'British', 'Hungarian', 'Belgian', 'Dutch', 'Italian', 'Azerbaijan', 'Singapore', 'United States', 'Mexican', 'Brazilian', 'Las Vegas', 'Qatar', 'Abu Dhabi'], 
                        help='Grand Prix to use for training')
    parser.add_argument('--model_path', type=str, default='f1_race_predictor_model.pkl', 
                        help='Path to save/load model')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = F1RacePredictor()
    
    if args.train:
        print(f"Training model using data from 2024 and GPs {args.gps}")
        
        # Create driver mapping
        predictor.create_driver_mapping(2024, args.gps[0])
        
        # Get historical data
        historical_data = predictor.get_historical_data(args.gps)
        historical_data.dropna(inplace=True)
        
        if historical_data is not None and not historical_data.empty:
            # Add weather data
            historical_data = predictor.add_weather_data(historical_data)
            
            # Train model
            model, metrics = predictor.train_model(historical_data)
            
            # Print metrics
            print("\nModel Training Results:")
            print(f"Training MAE: {metrics['train_mae']:.3f} seconds")
            print(f"Test MAE: {metrics['test_mae']:.3f} seconds")
            print(f"R² Score: {metrics['test_r2']:.3f}")
            
            # Save model
            if predictor.save_model(args.model_path):
                print(f"Model saved to {args.model_path}")
            else:
                print("Failed to save model")
        else:
            print("Failed to collect historical data")
    
    if args.predict:
        # Load model
        if predictor.load_model(args.model_path):
            print(f"Model loaded from {args.model_path}")
            
            # Create sample qualifying data
            sample_drivers = [
                "Max Verstappen", "Lewis Hamilton", "Charles Leclerc", "Lando Norris", 
                "Carlos Sainz Jr.", "George Russell", "Sergio Perez", "Fernando Alonso", 
                "Oscar Piastri", "Pierre Gasly"
            ]
            
            sample_times = [90.5, 90.7, 90.8, 90.9, 91.0, 91.1, 91.2, 91.3, 91.4, 91.5]
            
            qualifying_data = pd.DataFrame({
                "Driver": sample_drivers,
                "QualifyingTime": sample_times
            })
            
            # Sample weather data
            weather_data = pd.DataFrame({
                "AirTemp": [25],
                "Humidity": [60],
                "WindSpeed": [10]
            })
            
            # Make predictions
            predictions = predictor.predict_race_results(qualifying_data, weather_data)
            
            if predictions is not None:
                print("\nPredicted Race Results:")
                print(predictions[['PredictedPosition', 'Driver', 'QualifyingTime', 'PredictedLapTime']]
                      .rename(columns={
                          'PredictedPosition': 'Position',
                          'QualifyingTime': 'Qualifying Time (s)',
                          'PredictedLapTime': 'Predicted Lap Time (s)'
                      }))
                
                # Calculate time gaps
                winner_time = predictions.iloc[0]['PredictedRaceTime']
                predictions['Gap to Winner (s)'] = predictions['PredictedRaceTime'] - winner_time
                
                print("\nTime Gaps to Winner:")
                print(predictions[['PredictedPosition', 'Driver', 'Gap to Winner (s)']]
                      .rename(columns={'PredictedPosition': 'Position'}))
            else:
                print("Failed to make predictions")
        else:
            print(f"Failed to load model from {args.model_path}")
    
    # If no arguments provided, show usage
    if not (args.train or args.predict):
        parser.print_help()
        print("\nExample usage:")
        print("  Train a model: python predictions.py --train --gps Australian Bahrain Monaco")
        print("  Make predictions: python predictions.py --predict --model_path f1_race_predictor_model.pkl")
        print("  Train and predict: python predictions.py --train --predict")
