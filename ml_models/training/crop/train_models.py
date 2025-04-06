#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import argparse

def load_data():
    """Load the cleaned crop datasets"""
    try:
        crop_yield_data = pd.read_csv('data/cleaned/crop_yield_cleaned.csv')
        crop_recommendation_data = pd.read_csv('data/cleaned/crop_recommendation_cleaned.csv')
        print("Successfully loaded crop yield and recommendation data.")
        return crop_yield_data, crop_recommendation_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def train_yield_prediction_models(crop_yield_data, output_dir='models/crop', compare_models=True):
    """Train machine learning models for crop yield prediction with model comparison"""
    
    yield_model_file = f'{output_dir}/yield_prediction_models.joblib'
    model_comparison_file = f'{output_dir}/yield_model_comparison.json'
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if crop_yield_data is None:
        print("No crop yield data available for training.")
        return None
    
    print("Training crop yield prediction models...")
    
    # Get unique crops from the dataset
    crop_columns = [col for col in crop_yield_data.columns 
                   if col not in ['State Name', 'Dist Name', 'Year']]
    
    # Prepare features
    X = crop_yield_data[['State Name', 'Dist Name', 'Year']]
    
    # Dictionary to store models
    yield_models = {}
    model_comparison_results = {}
    
    # Train a separate model for each crop
    for crop in crop_columns:
        print(f"Training yield prediction model for {crop}...")
        
        # Check if we have enough non-NaN data for this crop
        y = crop_yield_data[crop]
        valid_indices = ~y.isna()
        
        if valid_indices.sum() < 50:  # Skip crops with too little data
            print(f"Insufficient data for {crop}, skipping...")
            continue
        
        # Filter out rows with NaN yield values
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
                'ElasticNet': ElasticNet(random_state=42),
                'AdaBoost': AdaBoostRegressor(random_state=42)
            }
            
            # Compare models
            print(f"Comparing models for {crop}...")
            model_results = {}
            
            for name, model in models.items():
                # Train and evaluate model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                print(f"  {name} - MSE: {mse:.2f}, R²: {r2:.2f}")
                
                model_results[name] = {
                    'mse': float(mse),
                    'r2': float(r2),
                    'model': model
                }
            
            # Select best model based on R² score
            best_model_name = max(model_results.items(), key=lambda x: x[1]['r2'])[0]
            best_model = model_results[best_model_name]['model']
            best_mse = model_results[best_model_name]['mse']
            best_r2 = model_results[best_model_name]['r2']
            
            print(f"Best model for {crop}: {best_model_name} with R²: {best_r2:.2f}")
            
            # Save model comparison results
            model_comparison_results[crop] = {
                'best_model': best_model_name,
                'metrics': {name: {'mse': results['mse'], 'r2': results['r2']} 
                           for name, results in model_results.items()}
            }
        else:
            # Just use RandomForest without comparison
            best_model = RandomForestRegressor(n_estimators=100, random_state=42)
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            best_mse = mean_squared_error(y_test, y_pred)
            best_r2 = r2_score(y_test, y_pred)
            
            print(f"{crop} Model - MSE: {best_mse:.2f}, R²: {best_r2:.2f}")
        
        # Store the best model
        yield_models[crop] = {
            'model': best_model,
            'feature_names': X_encoded.columns.tolist(),
            'metrics': {
                'mse': float(best_mse),
                'r2': float(best_r2)
            }
        }
    
    # Save the models and comparison results
    joblib.dump(yield_models, yield_model_file)
    
    if compare_models:
        with open(model_comparison_file, 'w') as f:
            json.dump(model_comparison_results, f, indent=4)
    
    print(f"Trained and saved yield prediction models for {len(yield_models)} crops.")
    return yield_models

def train_crop_recommendation_model(crop_recommendation_data, output_dir='models/crop', compare_models=True):
    """Train a model to recommend suitable crops based on soil and climate parameters with model comparison"""
    
    recommender_model_file = f'{output_dir}/crop_recommendation_model.joblib'
    model_comparison_file = f'{output_dir}/recommendation_model_comparison.json'
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if crop_recommendation_data is None:
        print("No crop recommendation data available for training.")
        return None
    
    print("Training crop recommendation model...")
    
    # Prepare features and target
    X = crop_recommendation_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = crop_recommendation_data['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    if compare_models:
        # Define models to compare
        models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'GradientBoosting': GradientBoostingRegressor(random_state=42),
            'SVM': SVR(),
            'AdaBoost': AdaBoostRegressor(random_state=42)
        }
        
        # Compare models
        print("Comparing crop recommendation models...")
        model_results = {}
        
        for name, model in models.items():
            # For SVM and other non-classifier models, use a different approach
            if name in ['SVM', 'AdaBoost', 'GradientBoosting']:
                # Train separate models for each class (one-vs-rest approach)
                class_models = []
                for class_idx in range(len(label_encoder.classes_)):
                    # Create binary target (1 for current class, 0 for others)
                    binary_target = np.zeros_like(y_train_encoded)
                    binary_target[y_train_encoded == class_idx] = 1
                    
                    # Train model for this class
                    class_model = model.__class__(**{k: v for k, v in model.get_params().items()
                                                  if k != 'random_state'})
                    class_model.fit(X_train_scaled, binary_target)
                    class_models.append(class_model)
                
                # Predict using all class models
                class_predictions = []
                for class_idx, class_model in enumerate(class_models):
                    pred = class_model.predict(X_test_scaled)
                    class_predictions.append(pred)
                
                # For each sample, pick the class with highest prediction
                y_pred_encoded = np.argmax(np.array(class_predictions).T, axis=1)
                
            else:
                # For classifiers, use standard approach
                model.fit(X_train_scaled, y_train_encoded)
                y_pred_encoded = model.predict(X_test_scaled)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
            print(f"  {name} - Accuracy: {accuracy:.4f}")
            
            model_results[name] = {
                'accuracy': float(accuracy),
                'model': model if name not in ['SVM', 'AdaBoost', 'GradientBoosting'] else class_models
            }
        
        # Select best model based on accuracy
        best_model_name = max(model_results.items(), key=lambda x: x[1]['accuracy'])[0]
        if best_model_name in ['SVM', 'AdaBoost', 'GradientBoosting']:
            best_model = model_results[best_model_name]['model']  # List of models
            best_type = 'multi'
        else:
            best_model = model_results[best_model_name]['model']
            best_type = 'single'
        
        best_accuracy = model_results[best_model_name]['accuracy']
        
        print(f"Best model for crop recommendation: {best_model_name} with accuracy: {best_accuracy:.4f}")
        
        # Save model comparison results
        model_comparison = {
            'best_model': best_model_name,
            'metrics': {name: {'accuracy': results['accuracy']} 
                       for name, results in model_results.items() if name != 'model'}
        }
        
        with open(model_comparison_file, 'w') as f:
            json.dump(model_comparison, f, indent=4)
    else:
        # Just use RandomForest without comparison
        best_model = RandomForestClassifier(n_estimators=100, random_state=42)
        best_model.fit(X_train_scaled, y_train_encoded)
        y_pred_encoded = best_model.predict(X_test_scaled)
        best_accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
        best_type = 'single'
        
        print(f"Crop Recommendation Model - Accuracy: {best_accuracy:.4f}")
    
    # Create model bundle
    if best_type == 'single':
        model_bundle = {
            'model': best_model,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'feature_names': X.columns.tolist(),
            'classes': label_encoder.classes_.tolist(),
            'model_type': 'single',
            'metrics': {'accuracy': float(best_accuracy)}
        }
    else:
        model_bundle = {
            'models': best_model,  # List of models
            'scaler': scaler,
            'label_encoder': label_encoder,
            'feature_names': X.columns.tolist(),
            'classes': label_encoder.classes_.tolist(),
            'model_type': 'multi',
            'metrics': {'accuracy': float(best_accuracy)}
        }
    
    # Save the model
    joblib.dump(model_bundle, recommender_model_file)
    print("Crop recommendation model trained and saved.")
    return model_bundle

def main():
    parser = argparse.ArgumentParser(description='Train crop models')
    parser.add_argument('--output-dir', type=str, default='models/crop', 
                        help='Directory to save trained models')
    parser.add_argument('--model', type=str, choices=['all', 'yield', 'recommendation'], default='all',
                        help='Which model to train (default: all)')
    parser.add_argument('--compare', action='store_true', 
                        help='Compare different model algorithms and select the best one')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    crop_yield_data, crop_recommendation_data = load_data()
    
    # Train models based on argument
    if args.model in ['all', 'yield']:
        train_yield_prediction_models(crop_yield_data, args.output_dir, args.compare)
    
    if args.model in ['all', 'recommendation']:
        train_crop_recommendation_model(crop_recommendation_data, args.output_dir, args.compare)

if __name__ == "__main__":
    main() 
    