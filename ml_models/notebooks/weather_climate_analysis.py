import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create the cleaned directory if it doesn't exist
os.makedirs('data/cleaned', exist_ok=True)
os.makedirs('images', exist_ok=True)

# Load rainfall dataset
print("Loading RainfallDataset.csv...")
rainfall_df = pd.read_csv('data/RainfallDataset.csv', quotechar='"')

# Load growing period dataset
print("Loading LengthOfGrowingPeriod.csv...")
growing_period_df = pd.read_csv('data/LengthOfGrowingPeriod.csv', quotechar='"')

# Data preprocessing for rainfall dataset
print("Preprocessing rainfall data...")

# Basic information about the dataset
print("\nRainfall Dataset Info:")
print(f"Shape: {rainfall_df.shape}")
print(f"Columns: {rainfall_df.columns.tolist()}")

# Convert -1.0 values to NaN
rainfall_df = rainfall_df.replace(-1.0, np.nan)

# Summary statistics for rainfall
print("\nRainfall Summary Statistics:")
rainfall_summary = rainfall_df.describe()
print(rainfall_summary)

# Check missing values
missing_values = rainfall_df.isnull().sum()
print("\nMissing Values in Rainfall Dataset:")
print(missing_values)

# Calculate mean monthly rainfall per state/district
monthly_cols = [col for col in rainfall_df.columns if 'RAINFALL' in col and 'ANNUAL' not in col]
annual_col = 'ANNUAL RAINFALL (Millimeters)'

# Group by State and District and calculate the mean rainfall
state_district_rainfall = rainfall_df.groupby(['State Name', 'Dist Name'])[monthly_cols + [annual_col]].mean().reset_index()

# Create a cleaned dataset with state, district, and monthly/annual rainfall
rainfall_cleaned = state_district_rainfall.copy()

# Save the cleaned rainfall dataset
print("Saving cleaned rainfall dataset...")
rainfall_cleaned.to_csv('data/cleaned/rainfall_cleaned.csv', index=False)

# Basic information about growing period dataset
print("\nGrowing Period Dataset Info:")
print(f"Shape: {growing_period_df.shape}")
print(f"Columns: {growing_period_df.columns.tolist()}")

# Clean growing period dataset
growing_period_cleaned = growing_period_df[['State Name', 'Dist Name', 'LENGTH OF GROWING PERIOD DAYS (Number)']].copy()

# Merge rainfall and growing period datasets
print("\nMerging rainfall and growing period datasets...")
weather_climate_df = pd.merge(
    rainfall_cleaned,
    growing_period_cleaned,
    on=['State Name', 'Dist Name'],
    how='outer'
)

# Save the merged weather and climate dataset
print("Saving merged weather and climate dataset...")
weather_climate_df.to_csv('data/cleaned/weather_climate_combined.csv', index=False)

# Data visualization
print("\nGenerating visualizations...")

# Set the style for plots
sns.set(style="whitegrid")

# Plot 1: Average annual rainfall by state
plt.figure(figsize=(12, 8))
state_rainfall = rainfall_df.groupby('State Name')[annual_col].mean().sort_values(ascending=False)
ax = sns.barplot(x=state_rainfall.index, y=state_rainfall.values)
plt.title('Average Annual Rainfall by State')
plt.xlabel('State')
plt.ylabel('Average Annual Rainfall (mm)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('images/annual_rainfall_by_state.png')

# Plot 2: Monthly rainfall patterns
plt.figure(figsize=(14, 8))
monthly_avg = rainfall_df[monthly_cols].mean()
sns.lineplot(x=range(len(monthly_avg)), y=monthly_avg.values)
plt.title('Average Monthly Rainfall Pattern')
plt.xlabel('Month')
plt.ylabel('Average Rainfall (mm)')
plt.xticks(range(len(monthly_avg)), [col.split()[0].capitalize() for col in monthly_cols], rotation=45)
plt.tight_layout()
plt.savefig('images/monthly_rainfall_pattern.png')

# Plot 3: Growing period distribution
plt.figure(figsize=(12, 8))
sns.histplot(growing_period_df['LENGTH OF GROWING PERIOD DAYS (Number)'], bins=20, kde=True)
plt.title('Distribution of Growing Period Length')
plt.xlabel('Growing Period (days)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('images/growing_period_distribution.png')

# Plot 4: Correlation between annual rainfall and growing period
plt.figure(figsize=(10, 7))
merged_data = pd.merge(
    rainfall_df[['State Name', 'Dist Name', annual_col]],
    growing_period_df[['State Name', 'Dist Name', 'LENGTH OF GROWING PERIOD DAYS (Number)']],
    on=['State Name', 'Dist Name']
).drop_duplicates()

sns.scatterplot(
    data=merged_data,
    x=annual_col,
    y='LENGTH OF GROWING PERIOD DAYS (Number)'
)
plt.title('Rainfall vs Growing Period')
plt.xlabel('Annual Rainfall (mm)')
plt.ylabel('Growing Period (days)')
plt.tight_layout()
plt.savefig('images/rainfall_vs_growing_period.png')

print("\nWeather and Climate analysis completed successfully!")
print("Cleaned datasets saved to data/cleaned/ directory")
print("Visualization plots saved to images/ directory")

# Print conclusions
print("\nKey Findings:")
print(f"1. Average annual rainfall across all regions: {rainfall_df[annual_col].mean():.2f} mm")
print(f"2. Month with highest average rainfall: {monthly_cols[monthly_avg.argmax()].split()[0]}")
print(f"3. Month with lowest average rainfall: {monthly_cols[monthly_avg.argmin()].split()[0]}")
print(f"4. Average growing period: {growing_period_df['LENGTH OF GROWING PERIOD DAYS (Number)'].mean():.2f} days")
print(f"5. States with highest rainfall: {', '.join(state_rainfall.head(3).index.tolist())}")
print(f"6. States with lowest rainfall: {', '.join(state_rainfall.tail(3).index.tolist())}") 