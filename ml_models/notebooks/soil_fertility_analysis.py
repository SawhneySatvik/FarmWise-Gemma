import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create the cleaned directory if it doesn't exist
os.makedirs('data/cleaned', exist_ok=True)
os.makedirs('images', exist_ok=True)

# Load soil datasets
print("Loading soil datasets...")
soil_dataset = pd.read_csv('data/SoilDataset.csv', quotechar='"')
soil_nutrients = pd.read_csv('data/soil.csv', quotechar='"')
crop_recommendation = pd.read_csv('data/Crop_recommendation.csv', quotechar='"')

# Data preprocessing for SoilDataset.csv
print("\nPreprocessing soil type dataset...")
print(f"SoilDataset shape: {soil_dataset.shape}")
print(f"SoilDataset columns: {soil_dataset.columns.tolist()}")

# Convert percentage columns to numeric
for col in soil_dataset.columns:
    if '%' in col:
        soil_dataset[col] = soil_dataset[col].str.replace('%', '').astype(float)

# Check missing values
soil_missing = soil_dataset.isnull().sum()
print("\nMissing values in SoilDataset:")
print(soil_missing)

# Clean the soil type dataset
soil_cleaned = soil_dataset.copy()
# Replace any NaN values with 0
soil_cleaned = soil_cleaned.fillna(0)

# Save the cleaned soil type dataset
print("Saving cleaned soil type dataset...")
soil_cleaned.to_csv('data/cleaned/soil_types_cleaned.csv', index=False)

# Data preprocessing for soil.csv (nutrients)
print("\nPreprocessing soil nutrients dataset...")
print(f"Soil nutrients shape: {soil_nutrients.shape}")
print(f"Soil nutrients columns: {soil_nutrients.columns.tolist()}")

# Check missing values
nutrients_missing = soil_nutrients.isnull().sum()
print("\nMissing values in soil nutrients dataset:")
print(nutrients_missing)

# Clean the soil nutrients dataset
nutrients_cleaned = soil_nutrients.copy()
# Fill missing values with median values
for col in nutrients_cleaned.columns:
    if nutrients_cleaned[col].dtype != 'object' and nutrients_missing[col] > 0:
        nutrients_cleaned[col] = nutrients_cleaned[col].fillna(nutrients_cleaned[col].median())

# Save the cleaned soil nutrients dataset
print("Saving cleaned soil nutrients dataset...")
nutrients_cleaned.to_csv('data/cleaned/soil_nutrients_cleaned.csv', index=False)

# Data preprocessing for Crop_recommendation.csv
print("\nPreprocessing crop recommendation dataset...")
print(f"Crop recommendation shape: {crop_recommendation.shape}")
print(f"Crop recommendation columns: {crop_recommendation.columns.tolist()}")

# Check missing values
crop_missing = crop_recommendation.isnull().sum()
print("\nMissing values in crop recommendation dataset:")
print(crop_missing)

# Summary statistics for crop recommendation
print("\nCrop recommendation summary statistics:")
print(crop_recommendation.describe())

# Count of each crop type
crop_counts = crop_recommendation['label'].value_counts()
print("\nCrop counts:")
print(crop_counts)

# Clean the crop recommendation dataset
crop_recommendation_cleaned = crop_recommendation.copy()
# No specific cleaning needed based on the data, but we'll perform a check
# Check for outliers using IQR
numeric_cols = crop_recommendation_cleaned.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    Q1 = crop_recommendation_cleaned[col].quantile(0.25)
    Q3 = crop_recommendation_cleaned[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((crop_recommendation_cleaned[col] < lower_bound) | 
                (crop_recommendation_cleaned[col] > upper_bound)).sum()
    print(f"Column {col} has {outliers} outliers")

# Save the cleaned crop recommendation dataset
print("Saving cleaned crop recommendation dataset...")
crop_recommendation_cleaned.to_csv('data/cleaned/crop_recommendation_cleaned.csv', index=False)

# Create a combined soil and crop recommendation dataset
# We can't directly merge these datasets as they have different structures
# But we can create a comprehensive soil fertility dataset for reference

# Data visualization
print("\nGenerating visualizations...")

# Set the style for plots
sns.set(style="whitegrid")

# Plot 1: Distribution of soil types
plt.figure(figsize=(14, 10))
soil_type_cols = [col for col in soil_dataset.columns if '%' in col]
soil_types_avg = soil_dataset[soil_type_cols].mean().sort_values(ascending=False)
sns.barplot(x=soil_types_avg.index, y=soil_types_avg.values)
plt.title('Average Percentage of Soil Types')
plt.xlabel('Soil Type')
plt.ylabel('Average Percentage')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('images/soil_types_distribution.png')

# Plot 2: Nutrient levels distribution
plt.figure(figsize=(14, 8))
nutrient_cols = ['Zn', 'Fe', 'Cu', 'Mn', 'B', 'S']
nutrient_data = pd.melt(soil_nutrients, id_vars=['DistrictName'], value_vars=nutrient_cols)
sns.boxplot(x='variable', y='value', data=nutrient_data)
plt.title('Distribution of Soil Nutrient Levels')
plt.xlabel('Nutrient')
plt.ylabel('Percentage')
plt.tight_layout()
plt.savefig('images/soil_nutrient_distribution.png')

# Plot 3: Optimal soil conditions by crop
plt.figure(figsize=(16, 10))
crop_soil_data = crop_recommendation.groupby('label')[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].mean()
crop_soil_data = crop_soil_data.reset_index()

# Plot N, P, K values by crop
plt.subplot(2, 2, 1)
crop_npk = pd.melt(crop_soil_data, id_vars=['label'], value_vars=['N', 'P', 'K'])
sns.barplot(x='label', y='value', hue='variable', data=crop_npk)
plt.title('Average N, P, K Requirements by Crop')
plt.xlabel('Crop')
plt.ylabel('Value')
plt.xticks(rotation=90)
plt.legend(title='Nutrient')

# Plot temperature by crop
plt.subplot(2, 2, 2)
sns.barplot(x='label', y='temperature', data=crop_soil_data)
plt.title('Average Temperature Requirements by Crop')
plt.xlabel('Crop')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=90)

# Plot humidity by crop
plt.subplot(2, 2, 3)
sns.barplot(x='label', y='humidity', data=crop_soil_data)
plt.title('Average Humidity Requirements by Crop')
plt.xlabel('Crop')
plt.ylabel('Humidity (%)')
plt.xticks(rotation=90)

# Plot rainfall by crop
plt.subplot(2, 2, 4)
sns.barplot(x='label', y='rainfall', data=crop_soil_data)
plt.title('Average Rainfall Requirements by Crop')
plt.xlabel('Crop')
plt.ylabel('Rainfall (mm)')
plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig('images/crop_soil_requirements.png')

# Plot 4: Correlation heatmap for crop recommendation features
plt.figure(figsize=(12, 10))
crop_corr = crop_recommendation.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(crop_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation between Soil and Climate Features')
plt.tight_layout()
plt.savefig('images/soil_climate_correlation.png')

print("\nSoil and Fertility analysis completed successfully!")
print("Cleaned datasets saved to data/cleaned/ directory")
print("Visualization plots saved to images/ directory")

# Print conclusions
print("\nKey Findings:")
print(f"1. Most common soil types: {', '.join(soil_types_avg.head(3).index.tolist())}")
print(f"2. Average NPK levels in crop recommendation dataset: N={crop_recommendation['N'].mean():.2f}, "
      f"P={crop_recommendation['P'].mean():.2f}, K={crop_recommendation['K'].mean():.2f}")
print(f"3. Top 3 crops in recommendation dataset: {', '.join(crop_counts.head(3).index.tolist())}")
print("4. Important soil nutrient correlations:")
for i, col1 in enumerate(crop_corr.columns[:-1]):
    for col2 in crop_corr.columns[i+1:]:
        if abs(crop_corr.loc[col1, col2]) > 0.5:
            print(f"   - {col1} and {col2}: {crop_corr.loc[col1, col2]:.2f}") 