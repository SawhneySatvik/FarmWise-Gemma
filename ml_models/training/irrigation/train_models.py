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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import argparse

def load_data():
    """Load the cleaned irrigation and rainfall datasets"""
    try:
        irrigation_data = pd.read_csv('data/cleaned/irrigation_cleaned.csv')
        rainfall_data = pd.read_csv('data/cleaned/rainfall_cleaned.csv')
        print("Successfully loaded irrigation and rainfall data.")
        return irrigation_data, rainfall_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def train_irrigation_models(irrigation_data, rainfall_data, output_dir='models/irrigation', compare_models=True):
    """Train machine learning models for irrigation recommendation with model comparison"""
    
    irrigation_model_file = f'{output_dir}/irrigation_recommendation_models.joblib'
    model_comparison_file = f'{output_dir}/irrigation_model_comparison.json'
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if irrigation_data is None or rainfall_data is None:
        print("No irrigation or rainfall data available for training.")
        return None
    
    print("Training irrigation recommendation models...")
    
    # Dictionary to store models
    irrigation_models = {}
    model_comparison_results = {}
    
    try:
        # Get crop columns from irrigation data
        crop_columns = [col for col in irrigation_data.columns 
                       if col not in ['State Name', 'Dist Name', 'Year']]
        
        # Prepare for training one model per crop
        for crop in crop_columns:
            print(f"Training irrigation model for {crop}...")
            
            # Merge irrigation and rainfall data
            merged_data = pd.merge(
                irrigation_data[['State Name', 'Dist Name', 'Year', crop]],
                rainfall_data[['State Name', 'Dist Name', 'Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual']],
                on=['State Name', 'Dist Name', 'Year'],
                how='inner'
            )
            
            # Check if we have enough data
            if len(merged_data) < 50:
                print(f"Insufficient data for {crop}, skipping...")
                continue
            
            # Filter out rows with NaN values
            merged_data = merged_data.dropna(subset=[crop])
            
            if len(merged_data) < 50:
                print(f"Insufficient data after filtering NaNs for {crop}, skipping...")
                continue
            
            # Prepare features and target
            X = merged_data[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual']]
            
            # Also include state and district as one-hot encoded features
            X_with_location = pd.get_dummies(merged_data[['State Name', 'Dist Name']], drop_first=False)
            X = pd.concat([X, X_with_location], axis=1)
            
            y = merged_data[crop]
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if compare_models:
                # Define models to compare
                models = {
                    'RandomForest': RandomForestRegressor(random_state=42),
                    'GradientBoosting': GradientBoostingRegressor(random_state=42),
                    'Ridge': Ridge(random_state=42),
                    'Lasso': Lasso(random_state=42, alpha=0.01),
                    'ElasticNet': ElasticNet(random_state=42, alpha=0.01, l1_ratio=0.5),
                    'SVR': SVR(kernel='rbf', C=10.0, gamma='scale'),
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
                    
                    print(f"  {name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
                    
                    model_results[name] = {
                        'mse': float(mse),
                        'rmse': float(rmse),
                        'mae': float(mae),
                        'r2': float(r2),
                        'model': model
                    }
                
                # Select best model based on RMSE
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
                
                best_metrics = {
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'r2': float(r2)
                }
                
                print(f"{crop} Model - RMSE: {rmse:.2f}, R²: {r2:.4f}")
            
            # Store the best model
            irrigation_models[crop] = {
                'model': best_model,
                'feature_names': X.columns.tolist(),
                'metrics': best_metrics
            }
        
        # Save the models and comparison results
        joblib.dump(irrigation_models, irrigation_model_file)
        
        if compare_models:
            with open(model_comparison_file, 'w') as f:
                json.dump(model_comparison_results, f, indent=4)
        
        print(f"Trained and saved irrigation models for {len(irrigation_models)} crops.")
        return irrigation_models
        
    except Exception as e:
        print(f"Error training irrigation models: {e}")
        return None

def train_water_requirement_models(irrigation_data, rainfall_data, output_dir='models/irrigation'):
    """Train models to predict crop water requirements based on climate data"""
    
    water_req_model_file = f'{output_dir}/water_requirement_models.joblib'
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if irrigation_data is None or rainfall_data is None:
        print("No irrigation or rainfall data available for training.")
        return None
    
    print("Training crop water requirement models...")
    
    # Define base water requirements for crops (in mm per growing season)
    # These will be adjusted based on climate data
    base_water_requirements = {
        "Rice": 1200,
        "Wheat": 600,
        "Maize": 500,
        "Jowar": 450,
        "Bajra": 400,
        "Barley": 500,
        "Small millets": 400,
        "Ragi": 450,
        "Gram": 350,
        "Tur": 400,
        "Other pulses": 400,
        "Groundnut": 500,
        "Rapeseed &Mustard": 400,
        "Cotton": 700,
        "Jute": 500,
        "Sugarcane": 1500,
        "Tobacco": 400,
        "Potato": 500,
        "Onion": 400,
        "Other Condiments & Spices": 500
    }
    
    # Get crop columns from irrigation data
    crop_columns = [col for col in irrigation_data.columns 
                   if col not in ['State Name', 'Dist Name', 'Year']]
    
    # Dictionary to store models and adjustments
    water_requirement_models = {}
    
    # Create a climate-based adjustment model for each crop
    for crop in crop_columns:
        if crop not in base_water_requirements:
            continue
            
        print(f"Creating water requirement model for {crop}...")
        
        # Merge irrigation and climate data
        merged_data = pd.merge(
            irrigation_data[['State Name', 'Dist Name', 'Year', crop]],
            rainfall_data[['State Name', 'Dist Name', 'Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual']],
            on=['State Name', 'Dist Name', 'Year'],
            how='inner'
        )
        
        if len(merged_data) < 30:
            print(f"Insufficient data for {crop}, using base requirements only...")
            water_requirement_models[crop] = {
                'base_requirement': base_water_requirements[crop],
                'has_model': False
            }
            continue
        
        # Calculate climate factors for adjustments
        merged_data['avg_monthly_rainfall'] = merged_data[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']].mean(axis=1)
        
        # Define seasons
        merged_data['winter_rainfall'] = merged_data[['Jan', 'Feb']].mean(axis=1)
        merged_data['summer_rainfall'] = merged_data[['Mar', 'Apr', 'May']].mean(axis=1)
        merged_data['monsoon_rainfall'] = merged_data[['Jun', 'Jul', 'Aug', 'Sep']].mean(axis=1)
        merged_data['post_monsoon_rainfall'] = merged_data[['Oct', 'Nov', 'Dec']].mean(axis=1)
        
        # Calculate adjustment factor based on climate
        base_req = base_water_requirements[crop]
        
        # Calculate water requirements by region
        state_district_reqs = {}
        for (state, district), group in merged_data.groupby(['State Name', 'Dist Name']):
            if len(group) < 5:
                continue
                
            # Create an adjustment model based on:
            # 1. Base water requirement
            # 2. Average annual rainfall
            # 3. Seasonal distribution of rainfall
            # 4. Irrigation area (as a proxy for how much irrigation is typically needed)
            
            avg_annual_rainfall = group['Annual'].mean()
            avg_irrigation_area = group[crop].mean()
            
            # Simple adjustment formula: more rainfall = less irrigation needed
            # But with diminishing returns based on crop characteristics
            if avg_annual_rainfall > 0:
                rainfall_factor = min(1.0, base_req / (1.5 * avg_annual_rainfall))
            else:
                rainfall_factor = 1.0
                
            # Seasonal adjustments
            # Some crops need water in specific seasons
            winter_importance = 0.2
            summer_importance = 0.3
            monsoon_importance = 0.4
            post_monsoon_importance = 0.1
            
            # Customize seasonal importance by crop
            if crop == "Rice":
                monsoon_importance = 0.6
                summer_importance = 0.2
            elif crop == "Wheat":
                winter_importance = 0.5
                post_monsoon_importance = 0.3
            
            # Calculate seasonal rainfall adequacy
            seasonal_factor = (
                winter_importance * (1.0 - min(1.0, group['winter_rainfall'].mean() / (base_req * winter_importance * 3))) +
                summer_importance * (1.0 - min(1.0, group['summer_rainfall'].mean() / (base_req * summer_importance * 3))) +
                monsoon_importance * (1.0 - min(1.0, group['monsoon_rainfall'].mean() / (base_req * monsoon_importance * 3))) +
                post_monsoon_importance * (1.0 - min(1.0, group['post_monsoon_rainfall'].mean() / (base_req * post_monsoon_importance * 3)))
            )
            
            # Final adjustment combines rainfall and seasonal factors
            adjustment_factor = (rainfall_factor + seasonal_factor) / 2
            
            # Calculate region-specific water requirement
            adjusted_req = base_req * adjustment_factor
            
            state_district_reqs[(state, district)] = {
                'adjusted_requirement': float(adjusted_req),
                'annual_rainfall': float(avg_annual_rainfall),
                'irrigation_area': float(avg_irrigation_area),
                'adjustment_factor': float(adjustment_factor)
            }
        
        # Store model
        water_requirement_models[crop] = {
            'base_requirement': base_water_requirements[crop],
            'regional_requirements': state_district_reqs,
            'has_model': True
        }
    
    # Save the models
    joblib.dump(water_requirement_models, water_req_model_file)
    print(f"Created water requirement models for {len(water_requirement_models)} crops.")
    
    return water_requirement_models

def main():
    """Main function to handle command-line arguments and train models"""
    parser = argparse.ArgumentParser(description='Train irrigation models')
    parser.add_argument('--output-dir', default='models/irrigation', help='Directory to save trained models')
    parser.add_argument('--compare-models', action='store_true', help='Compare different model types')
    parser.add_argument('--skip-water-req', action='store_true', help='Skip training water requirement models')
    args = parser.parse_args()
    
    # Load data
    irrigation_data, rainfall_data = load_data()
    
    if irrigation_data is None or rainfall_data is None:
        print("Error: Could not load required data. Exiting.")
        return
    
    # Train irrigation recommendation models
    irrigation_models = train_irrigation_models(
        irrigation_data, 
        rainfall_data,
        output_dir=args.output_dir,
        compare_models=args.compare_models
    )
    
    # Train water requirement models (if not skipped)
    if not args.skip_water_req:
        water_req_models = train_water_requirement_models(
            irrigation_data,
            rainfall_data,
            output_dir=args.output_dir
        )
    
    print("Irrigation model training completed successfully.")

if __name__ == "__main__":
    main() 