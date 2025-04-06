import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create the cleaned directory if it doesn't exist
os.makedirs('data/cleaned', exist_ok=True)
os.makedirs('images', exist_ok=True)

# Load crop yield dataset
print("Loading CropYield.csv...")
crop_yield_df = pd.read_csv('data/CropYield.csv', quotechar='"')

# Data preprocessing for crop yield dataset
print("\nPreprocessing crop yield data...")

# Basic information about the dataset
print("\nCrop Yield Dataset Info:")
print(f"Shape: {crop_yield_df.shape}")
print(f"Columns: {crop_yield_df.columns.tolist()}")

# Convert -1.0 values to NaN
crop_yield_df = crop_yield_df.replace(-1.0, np.nan)

# Summary statistics for crop yield
print("\nCrop Yield Summary Statistics:")
yield_summary = crop_yield_df.describe()
print(yield_summary)

# Check missing values
missing_values = crop_yield_df.isnull().sum()
print("\nMissing Values in Crop Yield Dataset:")
print(missing_values)

# Get list of crops
crop_cols = [col for col in crop_yield_df.columns if col not in ['State Name', 'Dist Name', 'Year']]
print("\nList of crops:", crop_cols)

# Clean the crop yield dataset
crop_yield_cleaned = crop_yield_df.copy()

# Fill missing values with median for each crop
for crop in crop_cols:
    if missing_values[crop] > 0:
        crop_yield_cleaned[crop] = crop_yield_cleaned[crop].fillna(crop_yield_cleaned[crop].median())

# Save the cleaned crop yield dataset
print("Saving cleaned crop yield dataset...")
crop_yield_cleaned.to_csv('data/cleaned/crop_yield_cleaned.csv', index=False)

# Load rainfall data for correlation analysis
print("Loading cleaned rainfall data for correlation analysis...")
try:
    rainfall_cleaned = pd.read_csv('data/cleaned/rainfall_cleaned.csv')
    have_rainfall = True
except:
    print("Rainfall data not found, skipping rainfall-yield correlation analysis")
    have_rainfall = False

# Data visualization
print("\nGenerating visualizations...")

# Set the style for plots
sns.set(style="whitegrid")

# Plot 1: Average yield by crop
plt.figure(figsize=(14, 8))
crop_avg_yield = crop_yield_df[crop_cols].mean().sort_values(ascending=False)
sns.barplot(x=crop_avg_yield.index, y=crop_avg_yield.values)
plt.title('Average Yield by Crop')
plt.xlabel('Crop')
plt.ylabel('Average Yield (Kg/Hectare)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('images/average_yield_by_crop.png')

# Plot 2: Yield trends over time
plt.figure(figsize=(15, 10))

# Select top 5 crops by average yield
top_crops = crop_avg_yield.head(5).index.tolist()

# Calculate yearly average for top crops
yearly_avg = crop_yield_df.groupby('Year')[top_crops].mean()

# Plot trends
for crop in top_crops:
    plt.plot(yearly_avg.index, yearly_avg[crop], marker='o', label=crop)

plt.title('Yield Trends Over Time (Top 5 Crops)')
plt.xlabel('Year')
plt.ylabel('Average Yield (Kg/Hectare)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('images/yield_trends_over_time.png')

# Plot 3: State-wise comparison for top crop
plt.figure(figsize=(14, 8))
top_crop = crop_avg_yield.index[0]
state_crop_yield = crop_yield_df.groupby('State Name')[top_crop].mean().sort_values(ascending=False)
sns.barplot(x=state_crop_yield.index, y=state_crop_yield.values)
plt.title(f'Average {top_crop} Yield by State')
plt.xlabel('State')
plt.ylabel(f'{top_crop} Yield (Kg/Hectare)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('images/top_crop_yield_by_state.png')

# Plot 4: Yield distribution for top crops
plt.figure(figsize=(14, 8))
yield_data = pd.melt(crop_yield_df, id_vars=['State Name', 'Dist Name', 'Year'], 
                     value_vars=top_crops, var_name='Crop', value_name='Yield')
sns.boxplot(x='Crop', y='Yield', data=yield_data)
plt.title('Yield Distribution for Top 5 Crops')
plt.xlabel('Crop')
plt.ylabel('Yield (Kg/Hectare)')
plt.tight_layout()
plt.savefig('images/yield_distribution_top_crops.png')

# Plot 5: Correlation between rainfall and yield (if rainfall data available)
if have_rainfall:
    # Merge rainfall and yield data
    # Both datasets have State Name and Dist Name columns
    rainfall_year_district = rainfall_cleaned[['State Name', 'Dist Name', 'ANNUAL RAINFALL (Millimeters)']]
    
    # Melt the crop yield data to long format
    yield_long = pd.melt(
        crop_yield_cleaned, 
        id_vars=['State Name', 'Dist Name', 'Year'],
        value_vars=crop_cols,
        var_name='Crop', 
        value_name='Yield'
    )
    
    # Merge with rainfall data
    yield_rainfall = pd.merge(
        yield_long,
        rainfall_year_district,
        on=['State Name', 'Dist Name'],
        how='inner'
    )
    
    # Plot correlation for top crops
    plt.figure(figsize=(15, 10))
    for i, crop in enumerate(top_crops[:min(4, len(top_crops))]):
        plt.subplot(2, 2, i+1)
        crop_data = yield_rainfall[yield_rainfall['Crop'] == crop]
        sns.scatterplot(
            x='ANNUAL RAINFALL (Millimeters)', 
            y='Yield', 
            data=crop_data,
            alpha=0.6
        )
        plt.title(f'{crop} Yield vs Rainfall')
        plt.xlabel('Annual Rainfall (mm)')
        plt.ylabel('Yield (Kg/Hectare)')
    
    plt.tight_layout()
    plt.savefig('images/rainfall_vs_yield.png')
    
    # Calculate correlation coefficients
    crop_rain_corr = {}
    for crop in top_crops:
        crop_data = yield_rainfall[yield_rainfall['Crop'] == crop]
        corr = crop_data[['ANNUAL RAINFALL (Millimeters)', 'Yield']].corr().iloc[0, 1]
        crop_rain_corr[crop] = corr
    
    # Plot correlation coefficients
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(crop_rain_corr.keys()), y=list(crop_rain_corr.values()))
    plt.title('Correlation between Rainfall and Crop Yield')
    plt.xlabel('Crop')
    plt.ylabel('Correlation Coefficient')
    plt.xticks(rotation=90)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.tight_layout()
    plt.savefig('images/rainfall_yield_correlation.png')

# Load crop recommendation data for comprehensive analysis
print("Loading crop recommendation data for comprehensive analysis...")
try:
    crop_recommendation = pd.read_csv('data/Crop_recommendation.csv')
    have_recommendation = True
except:
    print("Crop recommendation data not found, skipping combined analysis")
    have_recommendation = False

# Combine yield data with recommendation data for comprehensive insights
if have_recommendation:
    # Create a mapping of typical yield to optimal conditions
    crop_mapping = {}
    for crop in crop_recommendation['label'].unique():
        # Find the closest match in our yield data
        matched_crop = None
        for yield_crop in crop_cols:
            if crop.lower() in yield_crop.lower() or yield_crop.lower() in crop.lower():
                matched_crop = yield_crop
                break
        
        if matched_crop:
            crop_mapping[crop] = matched_crop
    
    # Print the mapping
    print("\nCrop name mapping between datasets:")
    print(crop_mapping)
    
    # For crops that have a match, create a comprehensive profile
    comprehensive_data = []
    
    for rec_crop, yield_crop in crop_mapping.items():
        # Get recommendation data
        rec_data = crop_recommendation[crop_recommendation['label'] == rec_crop]
        avg_N = rec_data['N'].mean()
        avg_P = rec_data['P'].mean()
        avg_K = rec_data['K'].mean()
        avg_temp = rec_data['temperature'].mean()
        avg_humidity = rec_data['humidity'].mean()
        avg_ph = rec_data['ph'].mean()
        avg_rainfall = rec_data['rainfall'].mean()
        
        # Get yield data
        avg_yield = crop_yield_df[yield_crop].mean()
        
        # Combine into a profile
        crop_profile = {
            'Crop': rec_crop,
            'Average Yield (Kg/Hectare)': avg_yield,
            'N': avg_N,
            'P': avg_P,
            'K': avg_K,
            'Temperature': avg_temp,
            'Humidity': avg_humidity,
            'pH': avg_ph,
            'Rainfall': avg_rainfall
        }
        
        comprehensive_data.append(crop_profile)
    
    # Create comprehensive dataframe
    if comprehensive_data:
        comprehensive_df = pd.DataFrame(comprehensive_data)
        
        # Save the comprehensive crop profile
        print("Saving comprehensive crop profile...")
        comprehensive_df.to_csv('data/cleaned/crop_comprehensive_profile.csv', index=False)
        
        # Visualize the comprehensive profile
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x='Rainfall', 
            y='Average Yield (Kg/Hectare)',
            size='N',
            hue='Temperature',
            data=comprehensive_df
        )
        plt.title('Crop Yield vs. Rainfall, N Content and Temperature')
        plt.xlabel('Rainfall (mm)')
        plt.ylabel('Average Yield (Kg/Hectare)')
        for i, row in comprehensive_df.iterrows():
            plt.annotate(row['Crop'], (row['Rainfall'], row['Average Yield (Kg/Hectare)']))
        plt.tight_layout()
        plt.savefig('images/comprehensive_crop_analysis.png')

print("\nCrop Yield analysis completed successfully!")
print("Cleaned datasets saved to data/cleaned/ directory")
print("Visualization plots saved to images/ directory")

# Print conclusions
print("\nKey Findings:")
print(f"1. Top 3 crops by yield: {', '.join(crop_avg_yield.head(3).index.tolist())}")
print(f"2. Bottom 3 crops by yield: {', '.join(crop_avg_yield.tail(3).index.tolist())}")

# Calculate year-over-year growth for each crop
yearly_avg = crop_yield_df.groupby('Year')[crop_cols].mean()
growth_rates = {}
for crop in crop_cols:
    if len(yearly_avg) > 1:
        first_year = yearly_avg[crop].iloc[0]
        last_year = yearly_avg[crop].iloc[-1]
        if first_year > 0:
            growth = ((last_year - first_year) / first_year) * 100
            growth_rates[crop] = growth

# Print growth rates
if growth_rates:
    sorted_growth = sorted(growth_rates.items(), key=lambda x: x[1], reverse=True)
    print("3. Crop yield growth rates over the period:")
    for crop, growth in sorted_growth[:5]:
        print(f"   - {crop}: {growth:.2f}%")
    
# If we have rainfall correlation data, print that too
if have_rainfall and crop_rain_corr:
    print("4. Rainfall impact on crops:")
    for crop, corr in sorted(crop_rain_corr.items(), key=lambda x: abs(x[1]), reverse=True)[:3]:
        direction = "positive" if corr > 0 else "negative"
        print(f"   - {crop}: {direction} correlation ({corr:.2f})")

# Print most productive states
top_producers = {}
for crop in crop_cols:
    state_crop = crop_yield_df.groupby('State Name')[crop].mean().sort_values(ascending=False)
    if not state_crop.empty:
        top_producers[crop] = state_crop.index[0]

print("5. Top producing states for major crops:")
for crop, state in list(top_producers.items())[:5]:
    print(f"   - {crop}: {state}") 