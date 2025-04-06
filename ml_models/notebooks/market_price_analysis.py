import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create the cleaned directory if it doesn't exist
os.makedirs('data/cleaned', exist_ok=True)
os.makedirs('images', exist_ok=True)

# Load crop price dataset
print("Loading CropPrice.csv...")
crop_price_df = pd.read_csv('data/CropPrice.csv', quotechar='"')

# Data preprocessing for crop price dataset
print("\nPreprocessing crop price data...")

# Basic information about the dataset
print("\nCrop Price Dataset Info:")
print(f"Shape: {crop_price_df.shape}")
print(f"Columns: {crop_price_df.columns.tolist()}")

# Convert -1.0 values to NaN
crop_price_df = crop_price_df.replace(-1.0, np.nan)

# Summary statistics for crop price
print("\nCrop Price Summary Statistics:")
price_summary = crop_price_df.describe()
print(price_summary)

# Check missing values
missing_values = crop_price_df.isnull().sum()
print("\nMissing Values in Crop Price Dataset:")
print(missing_values)

# Get list of crops
crop_cols = [col for col in crop_price_df.columns if col not in ['State Name', 'Dist Name', 'Year']]
print("\nList of crops:", crop_cols)

# Clean the crop price dataset
crop_price_cleaned = crop_price_df.copy()

# Fill missing values with median for each crop and year
for crop in crop_cols:
    if missing_values[crop] > 0:
        # Group by year to maintain time trends
        for year in crop_price_cleaned['Year'].unique():
            year_mask = crop_price_cleaned['Year'] == year
            year_median = crop_price_cleaned.loc[year_mask, crop].median()
            if not pd.isna(year_median):
                # Replace NaN with year median
                crop_price_cleaned.loc[year_mask & crop_price_cleaned[crop].isna(), crop] = year_median
        
        # If there are still NaNs (in case entire year is NaN), use overall median
        if crop_price_cleaned[crop].isna().sum() > 0:
            overall_median = crop_price_cleaned[crop].median()
            crop_price_cleaned[crop] = crop_price_cleaned[crop].fillna(overall_median)

# Save the cleaned crop price dataset
print("Saving cleaned crop price dataset...")
crop_price_cleaned.to_csv('data/cleaned/crop_price_cleaned.csv', index=False)

# Load crop yield data for correlation analysis
print("Loading cleaned crop yield data for correlation analysis...")
try:
    crop_yield_cleaned = pd.read_csv('data/cleaned/crop_yield_cleaned.csv')
    have_yield = True
except:
    print("Crop yield data not found, skipping price-yield correlation analysis")
    have_yield = False

# Data visualization
print("\nGenerating visualizations...")

# Set the style for plots
sns.set(style="whitegrid")

# Plot 1: Average price by crop
plt.figure(figsize=(14, 8))
crop_avg_price = crop_price_df[crop_cols].mean().sort_values(ascending=False)
sns.barplot(x=crop_avg_price.index, y=crop_avg_price.values)
plt.title('Average Price by Crop')
plt.xlabel('Crop')
plt.ylabel('Average Price (Rs per Quintal)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('images/average_price_by_crop.png')

# Plot 2: Price trends over time
plt.figure(figsize=(15, 10))

# Select top 5 crops by average price
top_crops = crop_avg_price.head(5).index.tolist()

# Calculate yearly average for top crops
yearly_avg = crop_price_df.groupby('Year')[top_crops].mean()

# Plot trends
for crop in top_crops:
    plt.plot(yearly_avg.index, yearly_avg[crop], marker='o', label=crop)

plt.title('Price Trends Over Time (Top 5 Crops)')
plt.xlabel('Year')
plt.ylabel('Average Price (Rs per Quintal)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('images/price_trends_over_time.png')

# Plot 3: State-wise comparison for top crop
plt.figure(figsize=(14, 8))
top_crop = crop_avg_price.index[0]
state_crop_price = crop_price_df.groupby('State Name')[top_crop].mean().sort_values(ascending=False)
sns.barplot(x=state_crop_price.index, y=state_crop_price.values)
plt.title(f'Average {top_crop} Price by State')
plt.xlabel('State')
plt.ylabel(f'{top_crop} Price (Rs per Quintal)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('images/top_crop_price_by_state.png')

# Plot 4: Price distribution for top crops
plt.figure(figsize=(14, 8))
price_data = pd.melt(crop_price_df, id_vars=['State Name', 'Dist Name', 'Year'], 
                     value_vars=top_crops, var_name='Crop', value_name='Price')
sns.boxplot(x='Crop', y='Price', data=price_data)
plt.title('Price Distribution for Top 5 Crops')
plt.xlabel('Crop')
plt.ylabel('Price (Rs per Quintal)')
plt.tight_layout()
plt.savefig('images/price_distribution_top_crops.png')

# Plot 5: Price volatility (coefficient of variation) by crop
plt.figure(figsize=(14, 8))
# Calculate coefficient of variation (standard deviation / mean)
cv_data = []
for crop in crop_cols:
    crop_data = crop_price_df[crop].dropna()
    if len(crop_data) > 0:
        cv = crop_data.std() / crop_data.mean() * 100  # As percentage
        cv_data.append({'Crop': crop, 'CV': cv})

cv_df = pd.DataFrame(cv_data)
cv_df = cv_df.sort_values('CV', ascending=False)

sns.barplot(x='Crop', y='CV', data=cv_df)
plt.title('Price Volatility by Crop (Coefficient of Variation)')
plt.xlabel('Crop')
plt.ylabel('Coefficient of Variation (%)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('images/price_volatility_by_crop.png')

# Plot 6: Year-on-year price change for major crops
plt.figure(figsize=(15, 10))
# Calculate year-on-year percentage change
yearly_pct_change = yearly_avg.pct_change() * 100

# Plot for top crops
for crop in top_crops:
    plt.plot(yearly_pct_change.index[1:], yearly_pct_change[crop][1:], marker='o', label=crop)

plt.title('Year-on-Year Price Change (Top 5 Crops)')
plt.xlabel('Year')
plt.ylabel('Percentage Change (%)')
plt.legend()
plt.grid(True)
plt.axhline(y=0, color='r', linestyle='--')
plt.tight_layout()
plt.savefig('images/price_year_on_year_change.png')

# Plot 7: Correlation between price and yield (if yield data available)
if have_yield:
    # Find common crops between price and yield datasets
    yield_crop_cols = [col for col in crop_yield_cleaned.columns 
                      if col not in ['State Name', 'Dist Name', 'Year']]
    
    common_crops = []
    crop_mapping = {}
    
    for price_crop in crop_cols:
        for yield_crop in yield_crop_cols:
            # Check if the crop names match or are similar
            if (price_crop.lower() in yield_crop.lower() or 
                yield_crop.lower() in price_crop.lower()):
                common_crops.append(price_crop)
                crop_mapping[price_crop] = yield_crop
                break
    
    print("\nCommon crops between price and yield datasets:")
    print(crop_mapping)
    
    if common_crops:
        # Create a merged dataset for common crops
        price_yield_data = []
        
        for price_crop, yield_crop in crop_mapping.items():
            # Group data by year
            price_by_year = crop_price_df.groupby('Year')[price_crop].mean()
            yield_by_year = crop_yield_cleaned.groupby('Year')[yield_crop].mean()
            
            # Merge on Year
            merged = pd.DataFrame({
                'Year': price_by_year.index,
                'Price': price_by_year.values,
                'Yield': yield_by_year.reindex(price_by_year.index).values,
                'Crop': price_crop
            })
            
            # Add to the list
            price_yield_data.append(merged)
        
        if price_yield_data:
            # Combine all crop data
            price_yield_df = pd.concat(price_yield_data, ignore_index=True)
            
            # Remove rows with NaN
            price_yield_df = price_yield_df.dropna()
            
            # Save the merged price-yield dataset
            print("Saving price-yield correlation dataset...")
            price_yield_df.to_csv('data/cleaned/price_yield_correlation.csv', index=False)
            
            # Plot price vs yield for each crop
            plt.figure(figsize=(15, 10))
            for i, crop in enumerate(crop_mapping.keys()):
                if i < 4:  # Only plot first 4 crops to avoid overcrowding
                    plt.subplot(2, 2, i+1)
                    crop_data = price_yield_df[price_yield_df['Crop'] == crop]
                    
                    # Ensure we have enough data
                    if len(crop_data) >= 3:
                        sns.scatterplot(x='Yield', y='Price', data=crop_data)
                        
                        # Add regression line
                        from scipy import stats
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            crop_data['Yield'], crop_data['Price']
                        )
                        
                        # Add trendline and correlation info
                        x = crop_data['Yield']
                        plt.plot(x, intercept + slope * x, 'r', 
                                label=f'r = {r_value:.2f}, p = {p_value:.3f}')
                        
                        plt.title(f'{crop} Price vs Yield')
                        plt.xlabel(f'Yield ({yield_crop})')
                        plt.ylabel('Price (Rs per Quintal)')
                        plt.legend()
            
            plt.tight_layout()
            plt.savefig('images/price_vs_yield.png')
            
            # Calculate and plot elasticity (% change in price / % change in yield)
            price_yield_df = price_yield_df.sort_values(['Crop', 'Year'])
            
            elasticity_data = []
            for crop in crop_mapping.keys():
                crop_data = price_yield_df[price_yield_df['Crop'] == crop].copy()
                
                if len(crop_data) > 1:
                    # Calculate percentage changes
                    crop_data['Price_Pct_Change'] = crop_data.groupby('Crop')['Price'].pct_change() * 100
                    crop_data['Yield_Pct_Change'] = crop_data.groupby('Crop')['Yield'].pct_change() * 100
                    
                    # Calculate elasticity (avoiding division by zero)
                    crop_data['Elasticity'] = np.where(
                        crop_data['Yield_Pct_Change'] != 0,
                        crop_data['Price_Pct_Change'] / crop_data['Yield_Pct_Change'],
                        np.nan
                    )
                    
                    mean_elasticity = crop_data['Elasticity'].mean()
                    if not pd.isna(mean_elasticity):
                        elasticity_data.append({
                            'Crop': crop,
                            'Elasticity': mean_elasticity
                        })
            
            if elasticity_data:
                elasticity_df = pd.DataFrame(elasticity_data)
                
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Crop', y='Elasticity', data=elasticity_df)
                plt.title('Price Elasticity by Crop (% Change in Price / % Change in Yield)')
                plt.xlabel('Crop')
                plt.ylabel('Elasticity')
                plt.xticks(rotation=90)
                plt.axhline(y=0, color='r', linestyle='--')
                plt.tight_layout()
                plt.savefig('images/price_elasticity_by_crop.png')

print("\nMarket Price analysis completed successfully!")
print("Cleaned datasets saved to data/cleaned/ directory")
print("Visualization plots saved to images/ directory")

# Print conclusions
print("\nKey Findings:")
print(f"1. Top 3 crops by price: {', '.join(crop_avg_price.head(3).index.tolist())}")
print(f"2. Bottom 3 crops by price: {', '.join(crop_avg_price.tail(3).index.tolist())}")

# Calculate year-over-year growth for each crop price
yearly_avg = crop_price_df.groupby('Year')[crop_cols].mean()
price_growth_rates = {}
for crop in crop_cols:
    if len(yearly_avg) > 1:
        first_year = yearly_avg[crop].iloc[0]
        last_year = yearly_avg[crop].iloc[-1]
        if first_year > 0:
            growth = ((last_year - first_year) / first_year) * 100
            price_growth_rates[crop] = growth

# Print price growth rates
if price_growth_rates:
    sorted_growth = sorted(price_growth_rates.items(), key=lambda x: x[1], reverse=True)
    print("3. Crop price growth rates over the period:")
    for crop, growth in sorted_growth[:5]:
        print(f"   - {crop}: {growth:.2f}%")
    
    # Find crops with negative growth
    negative_growth = [crop for crop, growth in price_growth_rates.items() if growth < 0]
    if negative_growth:
        print(f"4. Crops with decreasing prices: {', '.join(negative_growth)}")

# Print volatility information
if cv_data:
    high_volatility = cv_df.head(3)['Crop'].tolist()
    low_volatility = cv_df.tail(3)['Crop'].tolist()
    print(f"5. Crops with highest price volatility: {', '.join(high_volatility)}")
    print(f"6. Crops with lowest price volatility: {', '.join(low_volatility)}")

# Print elasticity information if available
if have_yield and 'elasticity_df' in locals():
    elastic_crops = elasticity_df[elasticity_df['Elasticity'] < -0.5]['Crop'].tolist()
    if elastic_crops:
        print(f"7. Price-sensitive crops (elastic, elasticity < -0.5): {', '.join(elastic_crops)}")
    
    inelastic_crops = elasticity_df[elasticity_df['Elasticity'] > -0.5]['Crop'].tolist()
    if inelastic_crops:
        print(f"8. Price-insensitive crops (inelastic, elasticity > -0.5): {', '.join(inelastic_crops)}") 