#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import StandardScaler
from datetime import datetime

def ensure_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def prepare_rainfall_data(source_file='data/RainfallDataset.csv', output_file='data/cleaned/rainfall_cleaned.csv'):
    """Clean and prepare the rainfall dataset"""
    print(f"Preparing rainfall data from {source_file}...")
    
    try:
        # Load data
        data = pd.read_csv(source_file)
        
        # Basic cleaning
        data.columns = data.columns.str.strip()
        
        # Handle missing values
        data = data.fillna(0)
        
        # Rename columns for consistency across datasets
        column_mapping = {
            'STATE_NAME': 'State Name',
            'DISTRICT': 'Dist Name',
            'YEAR': 'Year'
        }
        
        data = data.rename(columns={col: new_col for col, new_col in column_mapping.items() 
                                   if col in data.columns})
        
        # Ensure directory exists
        ensure_directory(os.path.dirname(output_file))
        
        # Save cleaned data
        data.to_csv(output_file, index=False)
        
        print(f"Rainfall data cleaned and saved to {output_file}")
        return data
        
    except Exception as e:
        print(f"Error preparing rainfall data: {e}")
        return None

def prepare_irrigation_data(source_file='data/Irrigation.csv', output_file='data/cleaned/irrigation_cleaned.csv'):
    """Clean and prepare the irrigation dataset"""
    print(f"Preparing irrigation data from {source_file}...")
    
    try:
        # Load data
        data = pd.read_csv(source_file)
        
        # Basic cleaning
        data.columns = data.columns.str.strip()
        
        # Handle missing values - use 0 for missing irrigation data
        data = data.fillna(0)
        
        # Rename columns for consistency if needed
        column_mapping = {
            'STATE_NAME': 'State Name',
            'DISTRICT': 'Dist Name',
            'YEAR': 'Year'
        }
        
        data = data.rename(columns={col: new_col for col, new_col in column_mapping.items() 
                                   if col in data.columns})
        
        # Ensure directory exists
        ensure_directory(os.path.dirname(output_file))
        
        # Save cleaned data
        data.to_csv(output_file, index=False)
        
        print(f"Irrigation data cleaned and saved to {output_file}")
        return data
        
    except Exception as e:
        print(f"Error preparing irrigation data: {e}")
        return None

def prepare_crop_price_data(source_file='data/CropPrice.csv', output_file='data/cleaned/crop_price_cleaned.csv'):
    """Clean and prepare the crop price dataset"""
    print(f"Preparing crop price data from {source_file}...")
    
    try:
        # Load data
        data = pd.read_csv(source_file)
        
        # Basic cleaning
        data.columns = data.columns.str.strip()
        
        # Handle missing values - for prices, we'll use forward fill then backward fill
        # For prices marked as -1, replace with NaN first
        data = data.replace(-1, np.nan)
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # If still have NaNs, replace with zeros
        data = data.fillna(0)
        
        # Rename columns for consistency
        column_mapping = {
            'STATE_NAME': 'State Name',
            'DISTRICT': 'Dist Name',
            'YEAR': 'Year'
        }
        
        data = data.rename(columns={col: new_col for col, new_col in column_mapping.items() 
                                   if col in data.columns})
        
        # Convert all crop names to lowercase for consistency
        crop_columns = [col for col in data.columns if col not in ['State Name', 'Dist Name', 'Year']]
        data.columns = [col.lower() if col in crop_columns else col for col in data.columns]
        
        # Ensure directory exists
        ensure_directory(os.path.dirname(output_file))
        
        # Save cleaned data
        data.to_csv(output_file, index=False)
        
        print(f"Crop price data cleaned and saved to {output_file}")
        return data
        
    except Exception as e:
        print(f"Error preparing crop price data: {e}")
        return None

def prepare_crop_yield_data(source_file='data/CropYield.csv', output_file='data/cleaned/crop_yield_cleaned.csv'):
    """Clean and prepare the crop yield dataset"""
    print(f"Preparing crop yield data from {source_file}...")
    
    try:
        # Load data
        data = pd.read_csv(source_file)
        
        # Basic cleaning
        data.columns = data.columns.str.strip()
        
        # Handle missing values
        data = data.fillna(0)
        
        # Rename columns for consistency
        column_mapping = {
            'STATE_NAME': 'State Name',
            'DISTRICT': 'Dist Name',
            'YEAR': 'Year'
        }
        
        data = data.rename(columns={col: new_col for col, new_col in column_mapping.items() 
                                   if col in data.columns})
        
        # Convert all crop names to lowercase for consistency
        crop_columns = [col for col in data.columns if col not in ['State Name', 'Dist Name', 'Year']]
        
        # Create a new DataFrame with standardized column names
        renamed_data = data[['State Name', 'Dist Name', 'Year']].copy()
        
        for col in crop_columns:
            renamed_data[col.lower()] = data[col]
        
        # Ensure directory exists
        ensure_directory(os.path.dirname(output_file))
        
        # Save cleaned data
        renamed_data.to_csv(output_file, index=False)
        
        print(f"Crop yield data cleaned and saved to {output_file}")
        return renamed_data
        
    except Exception as e:
        print(f"Error preparing crop yield data: {e}")
        return None

def prepare_soil_data(source_file='data/soil.csv', output_file='data/cleaned/soil_cleaned.csv'):
    """Clean and prepare the soil dataset"""
    print(f"Preparing soil data from {source_file}...")
    
    try:
        # Load data
        data = pd.read_csv(source_file)
        
        # Basic cleaning
        data.columns = data.columns.str.strip()
        
        # Handle missing values
        # First replace any negative values with NaN (assuming they're error codes)
        for col in data.columns:
            if data[col].dtype.kind in 'ifc':  # integer, float, complex
                data[col] = data[col].apply(lambda x: np.nan if x < 0 else x)
        
        # Fill NaN values with the median of each column
        for col in data.columns:
            if data[col].dtype.kind in 'ifc':  # integer, float, complex
                data[col] = data[col].fillna(data[col].median())
        
        # Rename columns for consistency
        column_mapping = {
            'STATE_NAME': 'State Name',
            'DISTRICT': 'Dist Name'
        }
        
        data = data.rename(columns={col: new_col for col, new_col in column_mapping.items() 
                                   if col in data.columns})
        
        # Ensure directory exists
        ensure_directory(os.path.dirname(output_file))
        
        # Save cleaned data
        data.to_csv(output_file, index=False)
        
        print(f"Soil data cleaned and saved to {output_file}")
        return data
        
    except Exception as e:
        print(f"Error preparing soil data: {e}")
        return None

def prepare_crop_recommendation_data(source_file='data/Crop_recommendation.csv', 
                                    output_file='data/cleaned/crop_recommendation_cleaned.csv'):
    """Clean and prepare the crop recommendation dataset"""
    print(f"Preparing crop recommendation data from {source_file}...")
    
    try:
        # Load data
        data = pd.read_csv(source_file)
        
        # Basic cleaning
        data.columns = data.columns.str.strip()
        
        # Convert all labels to lowercase for consistency
        if 'label' in data.columns:
            data['label'] = data['label'].str.lower()
        
        # No missing values typically in this dataset, but we'll handle just in case
        # Fill numeric columns with their median
        numeric_cols = data.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].median())
        
        # Fill categorical columns with their mode
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            data[col] = data[col].fillna(data[col].mode()[0])
        
        # Ensure directory exists
        ensure_directory(os.path.dirname(output_file))
        
        # Save cleaned data
        data.to_csv(output_file, index=False)
        
        print(f"Crop recommendation data cleaned and saved to {output_file}")
        return data
        
    except Exception as e:
        print(f"Error preparing crop recommendation data: {e}")
        return None

def generate_sample_data(output_dir='data/samples'):
    """Generate sample data for testing when real data is insufficient"""
    print("Generating sample datasets for testing...")
    
    ensure_directory(output_dir)
    
    # Sample crop data
    crops = ['rice', 'wheat', 'maize', 'potato', 'tomato', 'cotton', 'sugarcane']
    
    # Sample states and districts
    locations = [
        ('Bihar', 'Patna'),
        ('Uttar Pradesh', 'Lucknow'),
        ('Punjab', 'Amritsar'),
        ('Karnataka', 'Bangalore'),
        ('Tamil Nadu', 'Chennai')
    ]
    
    # Generate sample crop price data
    def generate_price_data():
        rows = []
        years = list(range(2010, 2023))
        
        for state, district in locations:
            for year in years:
                row = {'State Name': state, 'Dist Name': district, 'Year': year}
                
                for crop in crops:
                    # Generate prices with some randomness but following a trend
                    base_price = {
                        'rice': 2000, 'wheat': 1800, 'maize': 1500, 
                        'potato': 1200, 'tomato': 2500, 'cotton': 5000, 'sugarcane': 300
                    }[crop]
                    
                    # Add yearly inflation of 3-7%
                    inflation_factor = 1.0 + sum([np.random.uniform(0.03, 0.07) for _ in range(year - 2010)])
                    
                    # Add seasonal and regional variation
                    seasonal_factor = np.random.uniform(0.8, 1.2)
                    regional_factor = {'Bihar': 0.9, 'Uttar Pradesh': 1.0, 'Punjab': 1.1, 
                                      'Karnataka': 1.05, 'Tamil Nadu': 0.95}.get(state, 1.0)
                    
                    final_price = base_price * inflation_factor * seasonal_factor * regional_factor
                    row[crop] = round(final_price, 2)
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(f"{output_dir}/sample_crop_prices.csv", index=False)
        print(f"Sample crop price data saved to {output_dir}/sample_crop_prices.csv")
    
    # Generate sample crop yield data
    def generate_yield_data():
        rows = []
        years = list(range(2010, 2023))
        
        for state, district in locations:
            for year in years:
                row = {'State Name': state, 'Dist Name': district, 'Year': year}
                
                for crop in crops:
                    # Generate yields with some randomness but following a trend
                    base_yield = {
                        'rice': 4.0, 'wheat': 3.5, 'maize': 4.2, 
                        'potato': 20.0, 'tomato': 25.0, 'cotton': 1.5, 'sugarcane': 70.0
                    }[crop]
                    
                    # Add yearly improvement of 1-3%
                    improvement_factor = 1.0 + sum([np.random.uniform(0.01, 0.03) for _ in range(year - 2010)])
                    
                    # Add seasonal and regional variation
                    seasonal_factor = np.random.uniform(0.85, 1.15)
                    regional_factor = {'Bihar': 0.95, 'Uttar Pradesh': 1.0, 'Punjab': 1.15, 
                                      'Karnataka': 1.05, 'Tamil Nadu': 1.0}.get(state, 1.0)
                    
                    final_yield = base_yield * improvement_factor * seasonal_factor * regional_factor
                    row[crop] = round(final_yield, 2)
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(f"{output_dir}/sample_crop_yields.csv", index=False)
        print(f"Sample crop yield data saved to {output_dir}/sample_crop_yields.csv")
    
    # Generate sample irrigation data
    def generate_irrigation_data():
        rows = []
        years = list(range(2010, 2023))
        
        for state, district in locations:
            for year in years:
                row = {'State Name': state, 'Dist Name': district, 'Year': year}
                
                for crop in crops:
                    # Generate irrigation area
                    base_area = {
                        'rice': 50.0, 'wheat': 40.0, 'maize': 30.0, 
                        'potato': 15.0, 'tomato': 10.0, 'cotton': 25.0, 'sugarcane': 35.0
                    }[crop]
                    
                    # Add yearly variation
                    yearly_factor = np.random.uniform(0.9, 1.1)
                    
                    # Add regional variation
                    regional_factor = {'Bihar': 0.9, 'Uttar Pradesh': 1.0, 'Punjab': 1.2, 
                                      'Karnataka': 0.8, 'Tamil Nadu': 0.7}.get(state, 1.0)
                    
                    final_area = base_area * yearly_factor * regional_factor
                    row[crop] = round(final_area, 2)
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(f"{output_dir}/sample_irrigation.csv", index=False)
        print(f"Sample irrigation data saved to {output_dir}/sample_irrigation.csv")
    
    # Generate sample rainfall data
    def generate_rainfall_data():
        rows = []
        years = list(range(2010, 2023))
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for state, district in locations:
            for year in years:
                row = {'State Name': state, 'Dist Name': district, 'Year': year}
                
                # Define rainfall pattern based on region
                if state in ['Bihar', 'Uttar Pradesh']:
                    pattern = [10, 15, 20, 25, 40, 150, 300, 250, 150, 40, 15, 10]
                elif state in ['Punjab']:
                    pattern = [20, 30, 40, 35, 25, 80, 200, 180, 100, 30, 20, 15]
                else:
                    pattern = [50, 30, 20, 30, 70, 100, 120, 110, 150, 180, 120, 60]
                
                # Add yearly variation
                yearly_factor = np.random.uniform(0.85, 1.15)
                
                # Set monthly rainfall
                for i, month in enumerate(months):
                    base_rainfall = pattern[i]
                    monthly_factor = np.random.uniform(0.8, 1.2)
                    row[month] = round(base_rainfall * yearly_factor * monthly_factor, 1)
                
                # Calculate annual rainfall
                row['Annual'] = sum([row[month] for month in months])
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(f"{output_dir}/sample_rainfall.csv", index=False)
        print(f"Sample rainfall data saved to {output_dir}/sample_rainfall.csv")
    
    # Execute sample data generation
    generate_price_data()
    generate_yield_data()
    generate_irrigation_data()
    generate_rainfall_data()
    
    print("Sample data generation complete.")

def main():
    """Main function to handle command-line arguments and perform data preparation"""
    parser = argparse.ArgumentParser(description='Prepare datasets for farming agents')
    parser.add_argument('--dataset', choices=['all', 'rainfall', 'irrigation', 'crop_price', 
                                             'crop_yield', 'soil', 'crop_recommendation', 'sample'], 
                        default='all', help='Which dataset to process')
    parser.add_argument('--output-dir', default='data/cleaned', help='Directory to save cleaned data')
    parser.add_argument('--source-dir', default='data', help='Directory containing source data')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    ensure_directory(args.output_dir)
    
    # Process selected dataset(s)
    if args.dataset in ['all', 'rainfall']:
        prepare_rainfall_data(
            source_file=f"{args.source_dir}/RainfallDataset.csv",
            output_file=f"{args.output_dir}/rainfall_cleaned.csv"
        )
    
    if args.dataset in ['all', 'irrigation']:
        prepare_irrigation_data(
            source_file=f"{args.source_dir}/Irrigation.csv",
            output_file=f"{args.output_dir}/irrigation_cleaned.csv"
        )
    
    if args.dataset in ['all', 'crop_price']:
        prepare_crop_price_data(
            source_file=f"{args.source_dir}/CropPrice.csv",
            output_file=f"{args.output_dir}/crop_price_cleaned.csv"
        )
    
    if args.dataset in ['all', 'crop_yield']:
        prepare_crop_yield_data(
            source_file=f"{args.source_dir}/CropYield.csv",
            output_file=f"{args.output_dir}/crop_yield_cleaned.csv"
        )
    
    if args.dataset in ['all', 'soil']:
        prepare_soil_data(
            source_file=f"{args.source_dir}/soil.csv",
            output_file=f"{args.output_dir}/soil_cleaned.csv"
        )
    
    if args.dataset in ['all', 'crop_recommendation']:
        prepare_crop_recommendation_data(
            source_file=f"{args.source_dir}/Crop_recommendation.csv",
            output_file=f"{args.output_dir}/crop_recommendation_cleaned.csv"
        )
    
    if args.dataset in ['sample']:
        generate_sample_data(output_dir=f"{args.output_dir}/samples")
    
    print("Data preparation complete.")

if __name__ == "__main__":
    main() 