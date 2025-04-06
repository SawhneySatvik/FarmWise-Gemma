#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
import argparse
from datetime import datetime

def load_data():
    """Load the cleaned crop price dataset"""
    try:
        crop_price_data = pd.read_csv('data/cleaned/crop_price_cleaned.csv')
        print("Successfully loaded crop price data.")
        return crop_price_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def train_price_prediction_models(crop_price_data, output_dir='models/market', compare_models=True):
    """Train machine learning models for crop price prediction with model comparison"""
    
    price_model_file = f'{output_dir}/price_prediction_models.joblib'
    model_comparison_file = f'{output_dir}/price_model_comparison.json'
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if crop_price_data is None:
        print("No crop price data available for training.")
        return None
    
    print("Training crop price prediction models...")
    
    # Get unique crops from the dataset
    crop_columns = [col for col in crop_price_data.columns 
                   if col not in ['State Name', 'Dist Name', 'Year']]
    
    # Prepare features
    X = crop_price_data[['State Name', 'Dist Name', 'Year']]
    
    # Dictionary to store models
    price_models = {}
    model_comparison_results = {}
    
    # Train a separate model for each crop
    for crop in crop_columns:
        print(f"Training price prediction model for {crop}...")
        
        # Check if we have enough non-NaN data for this crop
        y = crop_price_data[crop]
        valid_indices = ~y.isna() & (y != -1.0)  # Filter out NaN and -1.0 values
        
        if valid_indices.sum() < 50:  # Skip crops with too little data
            print(f"Insufficient price data for {crop}, skipping...")
            continue
        
        # Filter out rows with invalid price values
        X_crop = X[valid_indices].copy()
        y_crop = y[valid_indices].copy()
        
        # One-hot encode categorical features
        X_encoded = pd.get_dummies(X_crop, columns=['State Name', 'Dist Name'], drop_first=False)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_crop, test_size=0.2, random_state=42
        )
        
        if compare_models:
            # Define models to compare
            models = {
                'RandomForest': RandomForestRegressor(random_state=42),
                'GradientBoosting': GradientBoostingRegressor(random_state=42),
                'Ridge': Ridge(random_state=42),
                'Lasso': Lasso(random_state=42),
                'ElasticNet': ElasticNet(random_state=42),
                'SVR': SVR(kernel='rbf'),
                'AdaBoost': AdaBoostRegressor(random_state=42)
            }
            
            # Compare models
            print(f"Comparing models for {crop}...")
            model_results = {}
            
            for name, model in models.items():
                # Train and evaluate model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate various metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Calculate MAPE safely (avoid division by zero)
                non_zero_indices = y_test != 0
                if np.any(non_zero_indices):
                    mape = np.mean(np.abs((y_test[non_zero_indices] - y_pred[non_zero_indices]) / y_test[non_zero_indices])) * 100
                else:
                    mape = float('inf')
                
                print(f"  {name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%, R²: {r2:.2f}")
                
                model_results[name] = {
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'mape': float(mape),
                    'r2': float(r2),
                    'model': model
                }
            
            # Select best model based on RMSE (or R² if you prefer)
            best_model_name = min(model_results.items(), key=lambda x: x[1]['rmse'])[0]
            best_model = model_results[best_model_name]['model']
            best_metrics = {k: v for k, v in model_results[best_model_name].items() if k != 'model'}
            
            print(f"Best model for {crop}: {best_model_name} with RMSE: {best_metrics['rmse']:.2f}")
            
            # Save model comparison results
            model_comparison_results[crop] = {
                'best_model': best_model_name,
                'metrics': {name: {k: v for k, v in results.items() if k != 'model'} 
                           for name, results in model_results.items()}
            }
        else:
            # Just use RandomForest without comparison
            best_model = RandomForestRegressor(n_estimators=100, random_state=42)
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Calculate MAPE safely
            non_zero_indices = y_test != 0
            if np.any(non_zero_indices):
                mape = np.mean(np.abs((y_test[non_zero_indices] - y_pred[non_zero_indices]) / y_test[non_zero_indices])) * 100
            else:
                mape = float('inf')
            
            best_metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape),
                'r2': float(r2)
            }
            
            print(f"{crop} Model - RMSE: {rmse:.2f}, R²: {r2:.2f}")
        
        # Store the best model
        price_models[crop] = {
            'model': best_model,
            'feature_names': X_encoded.columns.tolist(),
            'metrics': best_metrics
        }
    
    # Save the models and comparison results
    joblib.dump(price_models, price_model_file)
    
    if compare_models:
        with open(model_comparison_file, 'w') as f:
            json.dump(model_comparison_results, f, indent=4)
    
    print(f"Trained and saved price prediction models for {len(price_models)} crops.")
    return price_models

def train_seasonality_models(crop_price_data, output_dir='models/market'):
    """Train models for seasonal price patterns"""
    
    seasonality_model_file = f'{output_dir}/price_seasonality_models.joblib'
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if crop_price_data is None:
        print("No crop price data available for training.")
        return None
    
    print("Analyzing crop price seasonality...")
    
    # Get list of crops (excluding non-crop columns)
    crops = [col for col in crop_price_data.columns 
            if col not in ['State Name', 'Dist Name', 'Year']]
    
    # Check if we have monthly price data
    month_columns = [f'Month_{i}' for i in range(1, 13)]
    has_monthly_data = all(col in crop_price_data.columns for col in month_columns)
    
    if not has_monthly_data:
        print("No monthly price data available for seasonality analysis.")
        return None
    
    # Dictionary to store seasonality patterns
    seasonality_models = {}
    
    # Process each crop
    for crop in crops:
        print(f"Analyzing seasonality for {crop}...")
        
        # Filter out invalid data
        valid_data = crop_price_data[crop_price_data[crop] != -1.0]
        
        if len(valid_data) < 50:
            print(f"Insufficient data for {crop}, skipping...")
            continue
        
        # Extract monthly patterns
        monthly_patterns = {}
        
        # Group by state and district to calculate local patterns
        locations = valid_data.groupby(['State Name', 'Dist Name'])
        
        for (state, district), location_data in locations:
            location_monthly = {}
            
            for i, month in enumerate(month_columns, 1):
                if month in location_data.columns:
                    # Filter out invalid monthly prices
                    valid_prices = location_data[location_data[month] != -1.0][month]
                    
                    if len(valid_prices) >= 5:  # Require at least 5 valid data points
                        location_monthly[i] = {
                            'avg_price': float(valid_prices.mean()),
                            'min_price': float(valid_prices.min()),
                            'max_price': float(valid_prices.max()),
                            'std_dev': float(valid_prices.std())
                        }
            
            if location_monthly:
                # Calculate price indices relative to annual average
                annual_avg = np.mean([m['avg_price'] for m in location_monthly.values()])
                
                for month, data in location_monthly.items():
                    data['price_index'] = data['avg_price'] / annual_avg if annual_avg > 0 else 1.0
                
                # Calculate best selling and buying months
                sorted_months = sorted(location_monthly.items(), key=lambda x: x[1]['price_index'], reverse=True)
                best_selling = [month for month, _ in sorted_months[:3]]  # Top 3 months
                best_buying = [month for month, _ in sorted_months[-3:]]  # Bottom 3 months
                
                location_key = f"{state}_{district}".replace(" ", "_")
                monthly_patterns[location_key] = {
                    'monthly_data': location_monthly,
                    'best_selling_months': best_selling,
                    'best_buying_months': best_buying,
                    'price_variation': (max([d['price_index'] for d in location_monthly.values()]) - 
                                       min([d['price_index'] for d in location_monthly.values()])) * 100
                }
        
        # Calculate national pattern (if enough locations)
        if len(monthly_patterns) >= 3:
            # Average the price indices across locations
            national_pattern = {}
            
            for month in range(1, 13):
                indices = [loc_data['monthly_data'].get(month, {}).get('price_index', None) 
                          for loc_data in monthly_patterns.values()]
                indices = [idx for idx in indices if idx is not None]
                
                if indices:
                    national_pattern[month] = {
                        'avg_price_index': float(np.mean(indices)),
                        'locations_count': len(indices)
                    }
            
            # Calculate national best months
            if national_pattern:
                sorted_months = sorted(national_pattern.items(), key=lambda x: x[1]['avg_price_index'], reverse=True)
                national_best_selling = [month for month, _ in sorted_months[:3]]  # Top 3 months
                national_best_buying = [month for month, _ in sorted_months[-3:]]  # Bottom 3 months
                
                monthly_patterns['national'] = {
                    'monthly_data': national_pattern,
                    'best_selling_months': national_best_selling,
                    'best_buying_months': national_best_buying,
                    'price_variation': (max([d['avg_price_index'] for d in national_pattern.values()]) - 
                                       min([d['avg_price_index'] for d in national_pattern.values()])) * 100
                }
        
        seasonality_models[crop] = monthly_patterns
    
    # Save the seasonality models
    joblib.dump(seasonality_models, seasonality_model_file)
    print(f"Analyzed and saved price seasonality models for {len(seasonality_models)} crops.")
    
    return seasonality_models

def main():
    """Main function to handle command-line arguments and train models"""
    parser = argparse.ArgumentParser(description='Train market prediction models')
    parser.add_argument('--output-dir', default='models/market', help='Directory to save trained models')
    parser.add_argument('--compare-models', action='store_true', help='Compare different model types')
    parser.add_argument('--skip-seasonality', action='store_true', help='Skip training seasonality models')
    args = parser.parse_args()
    
    # Load data
    crop_price_data = load_data()
    
    if crop_price_data is None:
        print("Error: Could not load crop price data. Exiting.")
        return
    
    # Train price prediction models
    price_models = train_price_prediction_models(
        crop_price_data, 
        output_dir=args.output_dir,
        compare_models=args.compare_models
    )
    
    # Train seasonality models (if not skipped)
    if not args.skip_seasonality:
        seasonality_models = train_seasonality_models(
            crop_price_data,
            output_dir=args.output_dir
        )
    
    print("Market model training completed successfully.")

if __name__ == "__main__":
    main() 