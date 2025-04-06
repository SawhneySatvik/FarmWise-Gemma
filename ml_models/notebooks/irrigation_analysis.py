import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create the cleaned directory if it doesn't exist
os.makedirs('data/cleaned', exist_ok=True)
os.makedirs('images', exist_ok=True)

# Load irrigation dataset
print("Loading Irrigation.csv...")
irrigation_df = pd.read_csv('data/Irrigation.csv', quotechar='"')

# Data preprocessing for irrigation dataset
print("\nPreprocessing irrigation data...")

# Basic information about the dataset
print("\nIrrigation Dataset Info:")
print(f"Shape: {irrigation_df.shape}")
print(f"Columns: {irrigation_df.columns.tolist()}")

# Convert -1.0 values to NaN
irrigation_df = irrigation_df.replace(-1.0, np.nan)

# Summary statistics for irrigation
print("\nIrrigation Summary Statistics:")
irrigation_summary = irrigation_df.describe()
print(irrigation_summary)

# Check missing values
missing_values = irrigation_df.isnull().sum()
print("\nMissing Values in Irrigation Dataset:")
print(missing_values)

# Get list of crops
crop_cols = [col for col in irrigation_df.columns if col not in ['State Name', 'Dist Name', 'Year']]
print("\nList of crops:", crop_cols)

# Clean the irrigation dataset
irrigation_cleaned = irrigation_df.copy()

# Fill missing values with median for each crop
for crop in crop_cols:
    if missing_values[crop] > 0:
        irrigation_cleaned[crop] = irrigation_cleaned[crop].fillna(irrigation_cleaned[crop].median())

# Save the cleaned irrigation dataset
print("Saving cleaned irrigation dataset...")
irrigation_cleaned.to_csv('data/cleaned/irrigation_cleaned.csv', index=False)

# Load rainfall data for correlation analysis
print("Loading cleaned rainfall data for correlation analysis...")
try:
    rainfall_cleaned = pd.read_csv('data/cleaned/rainfall_cleaned.csv')
    have_rainfall = True
except:
    print("Rainfall data not found, skipping rainfall-irrigation correlation analysis")
    have_rainfall = False

# Data visualization
print("\nGenerating visualizations...")

# Set the style for plots
sns.set(style="whitegrid")

# Plot 1: Total irrigated area by crop
plt.figure(figsize=(14, 8))
crop_total_irrigation = irrigation_df[crop_cols].sum().sort_values(ascending=False)
sns.barplot(x=crop_total_irrigation.index, y=crop_total_irrigation.values)
plt.title('Total Irrigated Area by Crop')
plt.xlabel('Crop')
plt.ylabel('Total Irrigated Area (Thousand Hectares)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('images/total_irrigation_by_crop.png')

# Plot 2: Irrigation trends over time
plt.figure(figsize=(15, 10))

# Select top 5 crops by total irrigation
top_crops = crop_total_irrigation.head(5).index.tolist()

# Calculate yearly total for top crops
yearly_total = irrigation_df.groupby('Year')[top_crops].sum()

# Plot trends
for crop in top_crops:
    plt.plot(yearly_total.index, yearly_total[crop], marker='o', label=crop)

plt.title('Irrigation Trends Over Time (Top 5 Crops)')
plt.xlabel('Year')
plt.ylabel('Irrigated Area (Thousand Hectares)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('images/irrigation_trends_over_time.png')

# Plot 3: State-wise comparison for top crop
plt.figure(figsize=(14, 8))
top_crop = crop_total_irrigation.index[0]
state_crop_irrigation = irrigation_df.groupby('State Name')[top_crop].sum().sort_values(ascending=False)
sns.barplot(x=state_crop_irrigation.index, y=state_crop_irrigation.values)
plt.title(f'Total {top_crop} Irrigated Area by State')
plt.xlabel('State')
plt.ylabel(f'{top_crop} Irrigated Area (Thousand Hectares)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('images/top_crop_irrigation_by_state.png')

# Plot 4: Distribution of irrigated area for top crops
plt.figure(figsize=(14, 8))
irrigation_data = pd.melt(irrigation_df, id_vars=['State Name', 'Dist Name', 'Year'], 
                     value_vars=top_crops, var_name='Crop', value_name='Irrigated Area')
sns.boxplot(x='Crop', y='Irrigated Area', data=irrigation_data)
plt.title('Distribution of Irrigated Area for Top 5 Crops')
plt.xlabel('Crop')
plt.ylabel('Irrigated Area (Thousand Hectares)')
plt.tight_layout()
plt.savefig('images/irrigation_distribution_top_crops.png')

# Plot 5: Year-on-year growth in irrigated area
plt.figure(figsize=(15, 10))
# Calculate year-on-year percentage change
yearly_pct_change = yearly_total.pct_change() * 100

# Plot for top crops
for crop in top_crops:
    plt.plot(yearly_pct_change.index[1:], yearly_pct_change[crop][1:], marker='o', label=crop)

plt.title('Year-on-Year Change in Irrigated Area (Top 5 Crops)')
plt.xlabel('Year')
plt.ylabel('Percentage Change (%)')
plt.legend()
plt.grid(True)
plt.axhline(y=0, color='r', linestyle='--')
plt.tight_layout()
plt.savefig('images/irrigation_year_on_year_change.png')

# Plot 6: Proportion of total agricultural area that is irrigated per state
plt.figure(figsize=(14, 8))
# Calculate total irrigated area per state
state_total_irrigation = irrigation_df.groupby('State Name')[crop_cols].sum().sum(axis=1)

# Calculate proportion of top crops in total irrigation per state
state_crop_proportion = pd.DataFrame(index=state_total_irrigation.index)
for crop in top_crops:
    state_crop_sum = irrigation_df.groupby('State Name')[crop].sum()
    state_crop_proportion[crop] = state_crop_sum / state_total_irrigation * 100

# Plot the proportion
state_crop_proportion = state_crop_proportion.sort_values(by=top_crops[0], ascending=False)
state_crop_proportion.plot(kind='bar', stacked=True, figsize=(14, 8))
plt.title('Proportion of Top Crops in Total Irrigation by State')
plt.xlabel('State')
plt.ylabel('Percentage of Total Irrigated Area (%)')
plt.legend(title='Crop')
plt.tight_layout()
plt.savefig('images/crop_irrigation_proportion_by_state.png')

# Plot 7: Correlation between rainfall and irrigation (if rainfall data available)
if have_rainfall:
    # Merge rainfall and irrigation data
    # Both datasets have State Name and Dist Name columns
    rainfall_state_district = rainfall_cleaned[['State Name', 'Dist Name', 'ANNUAL RAINFALL (Millimeters)']]
    
    # Calculate total irrigation per district
    irrigation_total = irrigation_df.groupby(['State Name', 'Dist Name', 'Year'])[crop_cols].sum().sum(axis=1).reset_index()
    irrigation_total = irrigation_total.rename(columns={0: 'Total Irrigation'})
    
    # Group by State and District (average across years)
    irrigation_avg = irrigation_total.groupby(['State Name', 'Dist Name'])['Total Irrigation'].mean().reset_index()
    
    # Merge with rainfall data
    irrigation_rainfall = pd.merge(
        irrigation_avg,
        rainfall_state_district,
        on=['State Name', 'Dist Name'],
        how='inner'
    )
    
    # Save the merged irrigation-rainfall dataset
    print("Saving irrigation-rainfall correlation dataset...")
    irrigation_rainfall.to_csv('data/cleaned/irrigation_rainfall_correlation.csv', index=False)
    
    # Plot correlation
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='ANNUAL RAINFALL (Millimeters)', 
        y='Total Irrigation', 
        data=irrigation_rainfall,
        alpha=0.6
    )
    
    # Add regression line
    from scipy import stats
    if len(irrigation_rainfall) >= 3:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            irrigation_rainfall['ANNUAL RAINFALL (Millimeters)'], 
            irrigation_rainfall['Total Irrigation']
        )
        
        # Add trendline and correlation info
        x = irrigation_rainfall['ANNUAL RAINFALL (Millimeters)']
        plt.plot(x, intercept + slope * x, 'r', 
                label=f'r = {r_value:.2f}, p = {p_value:.3f}')
    
    plt.title('Irrigation vs Rainfall')
    plt.xlabel('Annual Rainfall (mm)')
    plt.ylabel('Total Irrigated Area (Thousand Hectares)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('images/irrigation_vs_rainfall.png')
    
    # Plot 8: Irrigation intensity by rainfall category
    plt.figure(figsize=(12, 8))
    
    # Create rainfall categories
    irrigation_rainfall['Rainfall Category'] = pd.cut(
        irrigation_rainfall['ANNUAL RAINFALL (Millimeters)'],
        bins=[0, 800, 1200, 1600, 3000],
        labels=['Low (<800mm)', 'Medium (800-1200mm)', 'High (1200-1600mm)', 'Very High (>1600mm)']
    )
    
    # Calculate average irrigation by rainfall category
    category_irrigation = irrigation_rainfall.groupby('Rainfall Category')['Total Irrigation'].mean().reset_index()
    
    sns.barplot(x='Rainfall Category', y='Total Irrigation', data=category_irrigation)
    plt.title('Average Irrigation by Rainfall Category')
    plt.xlabel('Rainfall Category')
    plt.ylabel('Average Irrigated Area (Thousand Hectares)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('images/irrigation_by_rainfall_category.png')

# Load crop yield data for efficiency analysis
print("Loading cleaned crop yield data for irrigation efficiency analysis...")
try:
    crop_yield_cleaned = pd.read_csv('data/cleaned/crop_yield_cleaned.csv')
    have_yield = True
except:
    print("Crop yield data not found, skipping irrigation efficiency analysis")
    have_yield = False

# Calculate irrigation efficiency (yield per unit of irrigation)
if have_yield:
    # Find common crops between irrigation and yield datasets
    yield_crop_cols = [col for col in crop_yield_cleaned.columns 
                      if col not in ['State Name', 'Dist Name', 'Year']]
    
    common_crops = []
    crop_mapping = {}
    
    for irr_crop in crop_cols:
        for yield_crop in yield_crop_cols:
            # Check if the crop names match or are similar
            if (irr_crop.lower() in yield_crop.lower() or 
                yield_crop.lower() in irr_crop.lower()):
                common_crops.append(irr_crop)
                crop_mapping[irr_crop] = yield_crop
                break
    
    print("\nCommon crops between irrigation and yield datasets:")
    print(crop_mapping)
    
    if common_crops:
        # Create a merged dataset for common crops
        efficiency_data = []
        
        for year in irrigation_df['Year'].unique():
            for state in irrigation_df['State Name'].unique():
                for district in irrigation_df[irrigation_df['State Name'] == state]['Dist Name'].unique():
                    district_data = {}
                    district_data['Year'] = year
                    district_data['State Name'] = state
                    district_data['Dist Name'] = district
                    
                    # Get irrigation data
                    irr_mask = ((irrigation_df['Year'] == year) & 
                                (irrigation_df['State Name'] == state) & 
                                (irrigation_df['Dist Name'] == district))
                    
                    # Get yield data
                    yield_mask = ((crop_yield_cleaned['Year'] == year) & 
                                  (crop_yield_cleaned['State Name'] == state) & 
                                  (crop_yield_cleaned['Dist Name'] == district))
                    
                    if irr_mask.any() and yield_mask.any():
                        # Calculate efficiency for each common crop
                        for irr_crop, yield_crop in crop_mapping.items():
                            irrigation_value = irrigation_df.loc[irr_mask, irr_crop].values[0] if irrigation_df.loc[irr_mask, irr_crop].any() else np.nan
                            yield_value = crop_yield_cleaned.loc[yield_mask, yield_crop].values[0] if crop_yield_cleaned.loc[yield_mask, yield_crop].any() else np.nan
                            
                            # Calculate efficiency (yield per irrigation unit)
                            # Convert to metric tons per 1000 hectares
                            if not np.isnan(irrigation_value) and not np.isnan(yield_value) and irrigation_value > 0:
                                # Yield is in kg/hectare, irrigation in thousand hectares
                                # Convert yield to tons (÷ 1000) and multiply by irrigation area (× 1000)
                                district_data[f"{irr_crop}_Irrigation"] = irrigation_value
                                district_data[f"{irr_crop}_Yield"] = yield_value
                                district_data[f"{irr_crop}_Efficiency"] = yield_value / 1000  # Tons per hectare
                            
                        efficiency_data.append(district_data)
        
        if efficiency_data:
            # Create efficiency dataframe
            efficiency_df = pd.DataFrame(efficiency_data)
            
            # Save the irrigation efficiency dataset
            print("Saving irrigation efficiency dataset...")
            efficiency_df.to_csv('data/cleaned/irrigation_efficiency.csv', index=False)
            
            # Plot irrigation efficiency by crop
            plt.figure(figsize=(14, 8))
            
            # Extract efficiency columns
            efficiency_cols = [col for col in efficiency_df.columns if '_Efficiency' in col]
            
            # Calculate average efficiency by crop
            avg_efficiency = {}
            for col in efficiency_cols:
                crop_name = col.split('_')[0]
                avg_efficiency[crop_name] = efficiency_df[col].mean()
            
            # Create and sort dataframe
            avg_efficiency_df = pd.DataFrame.from_dict(avg_efficiency, orient='index', columns=['Efficiency'])
            avg_efficiency_df = avg_efficiency_df.sort_values('Efficiency', ascending=False)
            
            sns.barplot(x=avg_efficiency_df.index, y=avg_efficiency_df['Efficiency'])
            plt.title('Average Irrigation Efficiency by Crop (Tons per Hectare)')
            plt.xlabel('Crop')
            plt.ylabel('Efficiency (Tons per Hectare)')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig('images/irrigation_efficiency_by_crop.png')
            
            # Plot 9: Efficiency trends over time for top crops
            plt.figure(figsize=(15, 10))
            
            # Calculate yearly average for top efficiency crops
            top_efficiency_crops = avg_efficiency_df.head(5).index.tolist()
            yearly_efficiency = {}
            
            for year in efficiency_df['Year'].unique():
                year_data = efficiency_df[efficiency_df['Year'] == year]
                for crop in top_efficiency_crops:
                    col_name = f"{crop}_Efficiency"
                    if col_name in year_data.columns:
                        if crop not in yearly_efficiency:
                            yearly_efficiency[crop] = {}
                        yearly_efficiency[crop][year] = year_data[col_name].mean()
            
            # Plot efficiency trends
            for crop in yearly_efficiency:
                years = sorted(yearly_efficiency[crop].keys())
                values = [yearly_efficiency[crop][year] for year in years]
                plt.plot(years, values, marker='o', label=crop)
            
            plt.title('Irrigation Efficiency Trends Over Time (Top 5 Crops)')
            plt.xlabel('Year')
            plt.ylabel('Efficiency (Tons per Hectare)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('images/irrigation_efficiency_trends.png')

print("\nIrrigation analysis completed successfully!")
print("Cleaned datasets saved to data/cleaned/ directory")
print("Visualization plots saved to images/ directory")

# Print conclusions
print("\nKey Findings:")
print(f"1. Top 3 crops by irrigated area: {', '.join(crop_total_irrigation.head(3).index.tolist())}")
print(f"2. Bottom 3 crops by irrigated area: {', '.join(crop_total_irrigation.tail(3).index.tolist())}")

# Calculate year-over-year growth for each crop's irrigated area
yearly_total = irrigation_df.groupby('Year')[crop_cols].sum()
irrigation_growth_rates = {}
for crop in crop_cols:
    if len(yearly_total) > 1:
        first_year = yearly_total[crop].iloc[0]
        last_year = yearly_total[crop].iloc[-1]
        if first_year > 0:
            growth = ((last_year - first_year) / first_year) * 100
            irrigation_growth_rates[crop] = growth

# Print growth rates
if irrigation_growth_rates:
    sorted_growth = sorted(irrigation_growth_rates.items(), key=lambda x: x[1], reverse=True)
    print("3. Irrigation area growth rates over the period:")
    for crop, growth in sorted_growth[:5]:
        print(f"   - {crop}: {growth:.2f}%")
    
    # Find crops with negative growth
    negative_growth = [crop for crop, growth in irrigation_growth_rates.items() if growth < 0]
    if negative_growth:
        print(f"4. Crops with decreasing irrigated area: {', '.join(negative_growth)}")

# Print states with highest irrigation
high_irrigation_states = state_total_irrigation.sort_values(ascending=False).head(5).index.tolist()
print(f"5. States with highest total irrigation: {', '.join(high_irrigation_states)}")

# Print rainfall-irrigation correlation if available
if have_rainfall and 'r_value' in locals():
    correlation_type = "positive" if r_value > 0 else "negative"
    correlation_strength = "strong" if abs(r_value) > 0.5 else "weak"
    print(f"6. Rainfall has a {correlation_strength} {correlation_type} correlation (r={r_value:.2f}) with irrigation")

# Print irrigation efficiency information if available
if have_yield and 'avg_efficiency_df' in locals():
    high_efficiency_crops = avg_efficiency_df.head(3).index.tolist()
    low_efficiency_crops = avg_efficiency_df.tail(3).index.tolist()
    print(f"7. Crops with highest irrigation efficiency: {', '.join(high_efficiency_crops)}")
    print(f"8. Crops with lowest irrigation efficiency: {', '.join(low_efficiency_crops)}") 