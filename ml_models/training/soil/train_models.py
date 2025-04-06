#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import json
import argparse

def load_data():
    """Load the cleaned soil datasets"""
    try:
        soil_nutrients_data = pd.read_csv('data/cleaned/soil_nutrients_cleaned.csv')
        soil_types_data = pd.read_csv('data/cleaned/soil_types_cleaned.csv')
        crop_recommendation_data = pd.read_csv('data/cleaned/crop_recommendation_cleaned.csv')
        print("Successfully loaded soil and crop recommendation data.")
        return soil_nutrients_data, soil_types_data, crop_recommendation_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def train_nutrient_prediction_model(soil_nutrients_data, soil_types_data, output_dir='models/soil'):
    """Train a model to predict soil nutrient levels based on location and soil type"""
    
    nutrient_model_file = f'{output_dir}/nutrient_prediction_model.joblib'
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if soil_nutrients_data is None or soil_types_data is None:
        print("No data available for training. Please load data first.")
        return None
    
    print("Training soil nutrient prediction model...")
    
    # Merge soil nutrients and soil types data on district
    try:
        merged_data = pd.merge(
            soil_nutrients_data,
            soil_types_data,
            on=['State Name', 'Dist Name'],
            how='inner'
        )
        
        if len(merged_data) == 0:
            print("No matching records found after merging datasets.")
            return None
        
        # Prepare features and target
        # Using district, state, and soil type percentages to predict nutrient levels
        feature_cols = ['State Name', 'Dist Name'] + [col for col in soil_types_data.columns 
                                                  if col not in ['State Name', 'Dist Name']]
        target_cols = ['Zn', 'Fe', 'Cu', 'Mn', 'B', 'S']
        
        # One-hot encode categorical variables
        X = pd.get_dummies(merged_data[feature_cols], columns=['State Name', 'Dist Name'], drop_first=True)
        
        # Create a multi-output regression model
        models = {}
        
        for nutrient in target_cols:
            print(f"Training model for {nutrient}...")
            y = merged_data[nutrient]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"{nutrient} Model - MSE: {mse:.4f}, R²: {r2:.4f}")
            
            # Store model
            models[nutrient] = model
        
        # Save the model
        joblib.dump(models, nutrient_model_file)
        print(f"Nutrient prediction model trained and saved to {nutrient_model_file}")
        
        # Create a model performance report
        performance = {}
        for nutrient in target_cols:
            y = merged_data[nutrient]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            y_pred = models[nutrient].predict(X_test)
            performance[nutrient] = {
                "mse": float(mean_squared_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred))
            }
        
        # Save performance metrics
        with open(f'{output_dir}/nutrient_model_performance.json', 'w') as f:
            json.dump(performance, f, indent=4)
        
        return models
        
    except Exception as e:
        print(f"Error training nutrient prediction model: {e}")
        return None

def train_soil_classifier(soil_nutrients_data, soil_types_data, output_dir='models/soil'):
    """Train a model to classify soil types based on soil nutrient levels"""
    
    soil_classifier_file = f'{output_dir}/soil_type_classifier.joblib'
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if soil_nutrients_data is None or soil_types_data is None:
        print("No data available for training. Please load data first.")
        return None
    
    print("Training soil type classifier...")
    
    try:
        # Merge soil nutrients and soil types data
        merged_data = pd.merge(
            soil_nutrients_data,
            soil_types_data,
            on=['State Name', 'Dist Name'],
            how='inner'
        )
        
        if len(merged_data) == 0:
            print("No matching records found after merging datasets.")
            return None
        
        # Identify the dominant soil type for each district
        soil_type_cols = [col for col in soil_types_data.columns 
                         if col not in ['State Name', 'Dist Name']]
        
        # Create a new column indicating the dominant soil type
        merged_data['dominant_soil_type'] = merged_data[soil_type_cols].idxmax(axis=1)
        
        # Prepare features and target
        X = merged_data[['Zn', 'Fe', 'Cu', 'Mn', 'B', 'S']]
        y = merged_data['dominant_soil_type']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"Soil Classifier - Accuracy: {accuracy:.4f}")
        print(report)
        
        # Create a model dictionary with the model and scaler
        model_dict = {
            'model': model,
            'scaler': scaler,
            'classes': model.classes_.tolist()
        }
        
        # Save the model
        joblib.dump(model_dict, soil_classifier_file)
        print(f"Soil type classifier trained and saved to {soil_classifier_file}")
        
        # Save model performance
        with open(f'{output_dir}/soil_classifier_performance.json', 'w') as f:
            json.dump({
                "accuracy": float(accuracy),
                "classes": model.classes_.tolist()
            }, f, indent=4)
        
        return model_dict
        
    except Exception as e:
        print(f"Error training soil classifier: {e}")
        return None

def train_crop_recommendation_model(crop_recommendation_data, output_dir='models/soil'):
    """Train a model to recommend crops based on soil parameters"""
    
    crop_model_file = f'{output_dir}/crop_recommendation_model.joblib'
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if crop_recommendation_data is None:
        print("No crop recommendation data available for training.")
        return None
    
    print("Training crop recommendation model...")
    
    try:
        # Prepare features and target
        X = crop_recommendation_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        y = crop_recommendation_data['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"Crop Recommendation Model - Accuracy: {accuracy:.4f}")
        print(report)
        
        # Create a model dictionary with the model and scaler
        model_dict = {
            'model': model,
            'scaler': scaler,
            'classes': model.classes_.tolist()
        }
        
        # Save the model
        joblib.dump(model_dict, crop_model_file)
        print(f"Crop recommendation model trained and saved to {crop_model_file}")
        
        # Save model performance
        with open(f'{output_dir}/crop_model_performance.json', 'w') as f:
            json.dump({
                "accuracy": float(accuracy),
                "classes": model.classes_.tolist()
            }, f, indent=4)
        
        return model_dict
        
    except Exception as e:
        print(f"Error training crop recommendation model: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Train soil models')
    parser.add_argument('--output-dir', type=str, default='models/soil', 
                        help='Directory to save trained models')
    parser.add_argument('--model', type=str, choices=['all', 'nutrient', 'soil', 'crop'], default='all',
                        help='Which model to train (default: all)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    soil_nutrients_data, soil_types_data, crop_recommendation_data = load_data()
    
    # Train models based on argument
    if args.model in ['all', 'nutrient']:
        train_nutrient_prediction_model(soil_nutrients_data, soil_types_data, args.output_dir)
    
    if args.model in ['all', 'soil']:
        train_soil_classifier(soil_nutrients_data, soil_types_data, args.output_dir)
    
    if args.model in ['all', 'crop']:
        train_crop_recommendation_model(crop_recommendation_data, args.output_dir)

if __name__ == "__main__":
    main() 