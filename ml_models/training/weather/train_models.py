#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import argparse

def load_data():
    """Load the cleaned weather and climate datasets"""
    try:
        rainfall_data = pd.read_csv('data/cleaned/rainfall_cleaned.csv')
        print("Successfully loaded rainfall data.")
        return rainfall_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def train_rainfall_prediction_model(rainfall_data, output_dir='models/weather'):
    """Train a model to predict rainfall based on district and month"""
    
    rainfall_model_file = f'{output_dir}/rainfall_prediction_model.joblib'
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if rainfall_data is None:
        print("No data available for training. Please load data first.")
        return None
    
    print("Training rainfall prediction model...")
    
    try:
        # Prepare features and target
        # Extract year, month as features
        rainfall_data['Year'] = rainfall_data['Year'].astype(int)
        
        # Create binary columns for each month
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Reshape data for modeling
        model_data = []
        
        for district in rainfall_data['District Name'].unique():
            district_data = rainfall_data[rainfall_data['District Name'] == district]
            
            for year in district_data['Year'].unique():
                year_data = district_data[district_data['Year'] == year]
                
                if len(year_data) > 0:
                    for month in months:
                        if month in year_data.columns:
                            model_data.append({
                                'District': district,
                                'Year': year,
                                'Month': month,
                                'Rainfall': year_data[month].values[0]
                            })
        
        model_df = pd.DataFrame(model_data)
        
        # Create dummy variables for district and month
        X = pd.get_dummies(model_df[['District', 'Month', 'Year']], 
                          columns=['District', 'Month'], 
                          drop_first=False)
        y = model_df['Rainfall']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Rainfall Prediction Model - MSE: {mse:.4f}, R²: {r2:.4f}")
        
        # Save feature names for inference
        feature_names = X.columns.tolist()
        
        # Create model dictionary with the model and feature information
        model_dict = {
            'model': model,
            'feature_names': feature_names,
            'districts': model_df['District'].unique().tolist(),
            'months': months
        }
        
        # Save the model
        joblib.dump(model_dict, rainfall_model_file)
        print(f"Rainfall prediction model trained and saved to {rainfall_model_file}")
        
        # Save model performance
        with open(f'{output_dir}/rainfall_model_performance.json', 'w') as f:
            json.dump({
                "mse": float(mse),
                "r2": float(r2)
            }, f, indent=4)
        
        return model_dict
        
    except Exception as e:
        print(f"Error training rainfall prediction model: {e}")
        return None

def train_temperature_prediction_model(rainfall_data, output_dir='models/weather'):
    """
    Train a simplified temperature prediction model using rainfall data as a proxy.
    In a real system, you'd use actual temperature data.
    """
    
    temperature_model_file = f'{output_dir}/temperature_prediction_model.joblib'
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if rainfall_data is None:
        print("No data available for training. Please load data first.")
        return None
    
    print("Training temperature prediction model (simplified approach)...")
    
    try:
        # For this example, we'll create a simplified temperature dataset based on rainfall
        # In a real application, you would use actual temperature data
        
        # Create a mapping of months to seasons and typical temperatures in India
        month_to_temp = {
            'Jan': {'mean': 20, 'std': 3},  # Winter
            'Feb': {'mean': 22, 'std': 3},  # Winter to Spring
            'Mar': {'mean': 26, 'std': 3},  # Spring
            'Apr': {'mean': 30, 'std': 4},  # Spring to Summer
            'May': {'mean': 35, 'std': 5},  # Summer
            'Jun': {'mean': 34, 'std': 5},  # Summer to Monsoon
            'Jul': {'mean': 32, 'std': 4},  # Monsoon
            'Aug': {'mean': 31, 'std': 4},  # Monsoon
            'Sep': {'mean': 30, 'std': 4},  # Monsoon to Autumn
            'Oct': {'mean': 28, 'std': 3},  # Autumn
            'Nov': {'mean': 24, 'std': 3},  # Autumn to Winter
            'Dec': {'mean': 21, 'std': 3}   # Winter
        }
        
        # Prepare dataset
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Create synthetic temperature data with inverse relationship to rainfall
        model_data = []
        
        for district in rainfall_data['District Name'].unique():
            district_data = rainfall_data[rainfall_data['District Name'] == district]
            
            for year in district_data['Year'].unique():
                year_data = district_data[district_data['Year'] == year]
                
                if len(year_data) > 0:
                    for month in months:
                        if month in year_data.columns:
                            # Get base temperature for the month
                            base_temp = month_to_temp[month]['mean']
                            std_temp = month_to_temp[month]['std']
                            
                            # Adjust temperature based on rainfall (more rain = lower temp)
                            rainfall = year_data[month].values[0]
                            
                            # Inverse relationship with some randomness
                            rainfall_effect = min(5, rainfall / 100)  # Cap the effect
                            
                            # Generate temperature with some randomness
                            temperature = base_temp - rainfall_effect + np.random.normal(0, std_temp/2)
                            
                            model_data.append({
                                'District': district,
                                'Year': year,
                                'Month': month,
                                'Temperature': max(15, min(45, temperature))  # Bound temperatures
                            })
        
        model_df = pd.DataFrame(model_data)
        
        # Create dummy variables for district and month
        X = pd.get_dummies(model_df[['District', 'Month', 'Year']], 
                          columns=['District', 'Month'], 
                          drop_first=False)
        y = model_df['Temperature']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Temperature Prediction Model - MSE: {mse:.4f}, R²: {r2:.4f}")
        
        # Save feature names for inference
        feature_names = X.columns.tolist()
        
        # Create model dictionary with the model and feature information
        model_dict = {
            'model': model,
            'feature_names': feature_names,
            'districts': model_df['District'].unique().tolist(),
            'months': months
        }
        
        # Save the model
        joblib.dump(model_dict, temperature_model_file)
        print(f"Temperature prediction model trained and saved to {temperature_model_file}")
        
        # Save model performance
        with open(f'{output_dir}/temperature_model_performance.json', 'w') as f:
            json.dump({
                "mse": float(mse),
                "r2": float(r2)
            }, f, indent=4)
        
        return model_dict
        
    except Exception as e:
        print(f"Error training temperature prediction model: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Train weather models')
    parser.add_argument('--output-dir', type=str, default='models/weather', 
                        help='Directory to save trained models')
    parser.add_argument('--model', type=str, choices=['all', 'rainfall', 'temperature'], default='all',
                        help='Which model to train (default: all)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    rainfall_data = load_data()
    
    # Train models based on argument
    if args.model in ['all', 'rainfall']:
        train_rainfall_prediction_model(rainfall_data, args.output_dir)
    
    if args.model in ['all', 'temperature']:
        train_temperature_prediction_model(rainfall_data, args.output_dir)

if __name__ == "__main__":
    main() 