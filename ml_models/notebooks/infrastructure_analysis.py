import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create the cleaned directory if it doesn't exist
os.makedirs('data/cleaned', exist_ok=True)
os.makedirs('images', exist_ok=True)

# Load infrastructure dataset
print("Loading Infrastructure.csv...")
infrastructure_df = pd.read_csv('data/Infrastructure.csv', quotechar='"')

# Data preprocessing for infrastructure dataset
print("\nPreprocessing infrastructure data...")

# Basic information about the dataset
print("\nInfrastructure Dataset Info:")
print(f"Shape: {infrastructure_df.shape}")
print(f"Columns: {infrastructure_df.columns.tolist()}")

# Convert -1.0 values to NaN
infrastructure_df = infrastructure_df.replace(-1.0, np.nan)

# Summary statistics for infrastructure
print("\nInfrastructure Summary Statistics:")
infrastructure_summary = infrastructure_df.describe()
print(infrastructure_summary)

# Check missing values
missing_values = infrastructure_df.isnull().sum()
print("\nMissing Values in Infrastructure Dataset:")
print(missing_values)

# Clean the infrastructure dataset
infrastructure_cleaned = infrastructure_df.copy()

# Fill missing values with median for each infrastructure type
for col in infrastructure_df.columns:
    if col not in ['State Name', 'Dist Name', 'Year'] and missing_values[col] > 0:
        infrastructure_cleaned[col] = infrastructure_cleaned[col].fillna(infrastructure_cleaned[col].median())

# Save the cleaned infrastructure dataset
print("Saving cleaned infrastructure dataset...")
infrastructure_cleaned.to_csv('data/cleaned/infrastructure_cleaned.csv', index=False)

# Data visualization
print("\nGenerating visualizations...")

# Set the style for plots
sns.set(style="whitegrid")

# Plot 1: Infrastructure facilities by state
plt.figure(figsize=(14, 8))
state_infrastructure = infrastructure_df.groupby('State Name')[['Number of Banks', 'Number of Post Offices']].sum()
state_infrastructure = state_infrastructure.sort_values(by='Number of Banks', ascending=False)

# Create a bar plot
state_infrastructure.plot(kind='bar', figsize=(14, 8))
plt.title('Infrastructure Facilities by State')
plt.xlabel('State')
plt.ylabel('Number of Facilities')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('images/infrastructure_by_state.png')

# Plot 2: Correlation between banks and post offices
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='Number of Banks', 
    y='Number of Post Offices', 
    data=infrastructure_df,
    hue='State Name',
    alpha=0.7
)
plt.title('Correlation between Banks and Post Offices')
plt.xlabel('Number of Banks')
plt.ylabel('Number of Post Offices')
plt.legend(title='State', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('images/banks_vs_postoffices.png')

# Plot 3: Distribution of infrastructure by district
plt.figure(figsize=(14, 10))

# Create two subplots
plt.subplot(2, 1, 1)
sns.histplot(infrastructure_df['Number of Banks'], bins=20, kde=True)
plt.title('Distribution of Banks across Districts')
plt.xlabel('Number of Banks')
plt.ylabel('Number of Districts')

plt.subplot(2, 1, 2)
sns.histplot(infrastructure_df['Number of Post Offices'], bins=20, kde=True)
plt.title('Distribution of Post Offices across Districts')
plt.xlabel('Number of Post Offices')
plt.ylabel('Number of Districts')

plt.tight_layout()
plt.savefig('images/infrastructure_distribution.png')

# Plot 4: Calculate infrastructure density (per area or population)
# Since we don't have direct area/population data, let's compare relative infrastructure
# using a combined metric of banks and post offices
plt.figure(figsize=(14, 8))

# Calculate combined infrastructure metric
infrastructure_df['Total Facilities'] = infrastructure_df['Number of Banks'] + infrastructure_df['Number of Post Offices']
infrastructure_df['Bank to Post Office Ratio'] = infrastructure_df['Number of Banks'] / infrastructure_df['Number of Post Offices'].replace(0, np.nan)

# Group by state
state_metrics = infrastructure_df.groupby('State Name')[['Total Facilities', 'Bank to Post Office Ratio']].mean()
state_metrics = state_metrics.sort_values(by='Total Facilities', ascending=False)

# Create a bar plot for total facilities
plt.subplot(1, 2, 1)
sns.barplot(x=state_metrics.index, y=state_metrics['Total Facilities'])
plt.title('Average Total Facilities by State')
plt.xlabel('State')
plt.ylabel('Average Total Facilities')
plt.xticks(rotation=90)

# Create a bar plot for Bank to Post Office Ratio
plt.subplot(1, 2, 2)
sns.barplot(x=state_metrics.index, y=state_metrics['Bank to Post Office Ratio'])
plt.title('Average Bank to Post Office Ratio by State')
plt.xlabel('State')
plt.ylabel('Ratio')
plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig('images/infrastructure_metrics_by_state.png')

# Load crop yield data for correlation with infrastructure
print("Loading cleaned crop yield data for infrastructure correlation analysis...")
try:
    crop_yield_cleaned = pd.read_csv('data/cleaned/crop_yield_cleaned.csv')
    have_yield = True
    
    # Merge infrastructure and yield data
    # Group infrastructure by state-district
    infrastructure_avg = infrastructure_df.groupby(['State Name', 'Dist Name'])[['Number of Banks', 'Number of Post Offices', 'Total Facilities']].mean()
    infrastructure_avg = infrastructure_avg.reset_index()
    
    # Group yield by state-district (average across all years and crops)
    yield_cols = [col for col in crop_yield_cleaned.columns if col not in ['State Name', 'Dist Name', 'Year']]
    crop_yield_avg = crop_yield_cleaned.groupby(['State Name', 'Dist Name'])[yield_cols].mean()
    
    # Calculate average yield across all crops
    crop_yield_avg['Average Yield'] = crop_yield_avg.mean(axis=1)
    crop_yield_avg = crop_yield_avg.reset_index()
    
    # Merge the datasets
    infrastructure_yield = pd.merge(
        infrastructure_avg,
        crop_yield_avg[['State Name', 'Dist Name', 'Average Yield']],
        on=['State Name', 'Dist Name'],
        how='inner'
    )
    
    if len(infrastructure_yield) > 0:
        # Save the merged dataset
        print("Saving infrastructure-yield correlation dataset...")
        infrastructure_yield.to_csv('data/cleaned/infrastructure_yield_correlation.csv', index=False)
        
        # Plot correlation
        plt.figure(figsize=(15, 10))
        
        # Plot Banks vs Yield
        plt.subplot(2, 2, 1)
        sns.scatterplot(
            x='Number of Banks', 
            y='Average Yield', 
            data=infrastructure_yield,
            alpha=0.6
        )
        
        # Add regression line
        from scipy import stats
        if len(infrastructure_yield) >= 3:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                infrastructure_yield['Number of Banks'], 
                infrastructure_yield['Average Yield']
            )
            
            # Add trendline and correlation info
            x = infrastructure_yield['Number of Banks']
            plt.plot(x, intercept + slope * x, 'r', 
                    label=f'r = {r_value:.2f}, p = {p_value:.3f}')
        
        plt.title('Banks vs Average Crop Yield')
        plt.xlabel('Number of Banks')
        plt.ylabel('Average Yield (Kg/Hectare)')
        plt.legend()
        
        # Plot Post Offices vs Yield
        plt.subplot(2, 2, 2)
        sns.scatterplot(
            x='Number of Post Offices', 
            y='Average Yield', 
            data=infrastructure_yield,
            alpha=0.6
        )
        
        # Add regression line
        if len(infrastructure_yield) >= 3:
            slope, intercept, r_value2, p_value, std_err = stats.linregress(
                infrastructure_yield['Number of Post Offices'], 
                infrastructure_yield['Average Yield']
            )
            
            # Add trendline and correlation info
            x = infrastructure_yield['Number of Post Offices']
            plt.plot(x, intercept + slope * x, 'r', 
                    label=f'r = {r_value2:.2f}, p = {p_value:.3f}')
        
        plt.title('Post Offices vs Average Crop Yield')
        plt.xlabel('Number of Post Offices')
        plt.ylabel('Average Yield (Kg/Hectare)')
        plt.legend()
        
        # Plot Total Facilities vs Yield
        plt.subplot(2, 2, 3)
        sns.scatterplot(
            x='Total Facilities', 
            y='Average Yield', 
            data=infrastructure_yield,
            alpha=0.6
        )
        
        # Add regression line
        if len(infrastructure_yield) >= 3:
            slope, intercept, r_value3, p_value, std_err = stats.linregress(
                infrastructure_yield['Total Facilities'], 
                infrastructure_yield['Average Yield']
            )
            
            # Add trendline and correlation info
            x = infrastructure_yield['Total Facilities']
            plt.plot(x, intercept + slope * x, 'r', 
                    label=f'r = {r_value3:.2f}, p = {p_value:.3f}')
        
        plt.title('Total Facilities vs Average Crop Yield')
        plt.xlabel('Total Facilities')
        plt.ylabel('Average Yield (Kg/Hectare)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('images/infrastructure_vs_yield.png')
except:
    print("Crop yield data not found or error in correlation analysis")
    have_yield = False

# Load market price data for correlation with infrastructure
print("Loading cleaned market price data for infrastructure correlation analysis...")
try:
    crop_price_cleaned = pd.read_csv('data/cleaned/crop_price_cleaned.csv')
    have_price = True
    
    # Merge infrastructure and price data
    # Group infrastructure by state-district
    if 'infrastructure_avg' not in locals():
        infrastructure_avg = infrastructure_df.groupby(['State Name', 'Dist Name'])[['Number of Banks', 'Number of Post Offices', 'Total Facilities']].mean()
        infrastructure_avg = infrastructure_avg.reset_index()
    
    # Group price by state-district (average across all years and crops)
    price_cols = [col for col in crop_price_cleaned.columns if col not in ['State Name', 'Dist Name', 'Year']]
    crop_price_avg = crop_price_cleaned.groupby(['State Name', 'Dist Name'])[price_cols].mean()
    
    # Calculate average price across all crops
    crop_price_avg['Average Price'] = crop_price_avg.mean(axis=1)
    crop_price_avg = crop_price_avg.reset_index()
    
    # Merge the datasets
    infrastructure_price = pd.merge(
        infrastructure_avg,
        crop_price_avg[['State Name', 'Dist Name', 'Average Price']],
        on=['State Name', 'Dist Name'],
        how='inner'
    )
    
    if len(infrastructure_price) > 0:
        # Save the merged dataset
        print("Saving infrastructure-price correlation dataset...")
        infrastructure_price.to_csv('data/cleaned/infrastructure_price_correlation.csv', index=False)
        
        # Plot correlation
        plt.figure(figsize=(15, 10))
        
        # Plot Banks vs Price
        plt.subplot(2, 2, 1)
        sns.scatterplot(
            x='Number of Banks', 
            y='Average Price', 
            data=infrastructure_price,
            alpha=0.6
        )
        
        # Add regression line
        if len(infrastructure_price) >= 3:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                infrastructure_price['Number of Banks'], 
                infrastructure_price['Average Price']
            )
            
            # Add trendline and correlation info
            x = infrastructure_price['Number of Banks']
            plt.plot(x, intercept + slope * x, 'r', 
                    label=f'r = {r_value:.2f}, p = {p_value:.3f}')
        
        plt.title('Banks vs Average Crop Price')
        plt.xlabel('Number of Banks')
        plt.ylabel('Average Price (Rs per Quintal)')
        plt.legend()
        
        # Plot Post Offices vs Price
        plt.subplot(2, 2, 2)
        sns.scatterplot(
            x='Number of Post Offices', 
            y='Average Price', 
            data=infrastructure_price,
            alpha=0.6
        )
        
        # Add regression line
        if len(infrastructure_price) >= 3:
            slope, intercept, r_value2, p_value, std_err = stats.linregress(
                infrastructure_price['Number of Post Offices'], 
                infrastructure_price['Average Price']
            )
            
            # Add trendline and correlation info
            x = infrastructure_price['Number of Post Offices']
            plt.plot(x, intercept + slope * x, 'r', 
                    label=f'r = {r_value2:.2f}, p = {p_value:.3f}')
        
        plt.title('Post Offices vs Average Crop Price')
        plt.xlabel('Number of Post Offices')
        plt.ylabel('Average Price (Rs per Quintal)')
        plt.legend()
        
        # Plot Total Facilities vs Price
        plt.subplot(2, 2, 3)
        sns.scatterplot(
            x='Total Facilities', 
            y='Average Price', 
            data=infrastructure_price,
            alpha=0.6
        )
        
        # Add regression line
        if len(infrastructure_price) >= 3:
            slope, intercept, r_value3, p_value, std_err = stats.linregress(
                infrastructure_price['Total Facilities'], 
                infrastructure_price['Average Price']
            )
            
            # Add trendline and correlation info
            x = infrastructure_price['Total Facilities']
            plt.plot(x, intercept + slope * x, 'r', 
                    label=f'r = {r_value3:.2f}, p = {p_value:.3f}')
        
        plt.title('Total Facilities vs Average Crop Price')
        plt.xlabel('Total Facilities')
        plt.ylabel('Average Price (Rs per Quintal)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('images/infrastructure_vs_price.png')
except:
    print("Market price data not found or error in correlation analysis")
    have_price = False

print("\nInfrastructure analysis completed successfully!")
print("Cleaned datasets saved to data/cleaned/ directory")
print("Visualization plots saved to images/ directory")

# Print conclusions
print("\nKey Findings:")

# Top states by infrastructure
high_bank_states = state_infrastructure.sort_values(by='Number of Banks', ascending=False).head(3).index.tolist()
high_postoffice_states = state_infrastructure.sort_values(by='Number of Post Offices', ascending=False).head(3).index.tolist()

print(f"1. States with highest number of banks: {', '.join(high_bank_states)}")
print(f"2. States with highest number of post offices: {', '.join(high_postoffice_states)}")

# Calculate the correlation between banks and post offices
bank_postoffice_corr = infrastructure_df[['Number of Banks', 'Number of Post Offices']].corr().iloc[0, 1]
correlation_strength = "strong" if abs(bank_postoffice_corr) > 0.5 else "weak"
correlation_type = "positive" if bank_postoffice_corr > 0 else "negative"

print(f"3. Banks and post offices have a {correlation_strength} {correlation_type} correlation (r={bank_postoffice_corr:.2f})")

# Print states with highest and lowest bank to post office ratio
high_ratio_states = state_metrics.sort_values(by='Bank to Post Office Ratio', ascending=False).head(3).index.tolist()
low_ratio_states = state_metrics.sort_values(by='Bank to Post Office Ratio', ascending=True).head(3).index.tolist()

print(f"4. States with highest bank to post office ratio: {', '.join(high_ratio_states)}")
print(f"5. States with lowest bank to post office ratio: {', '.join(low_ratio_states)}")

# Print correlation between infrastructure and yield/price if available
if have_yield and 'r_value' in locals() and 'r_value2' in locals() and 'r_value3' in locals():
    print("6. Correlation between infrastructure and crop yield:")
    print(f"   - Banks vs. Yield: r={r_value:.2f}")
    print(f"   - Post Offices vs. Yield: r={r_value2:.2f}")
    print(f"   - Total Facilities vs. Yield: r={r_value3:.2f}")

if have_price and 'r_value' in locals() and 'r_value2' in locals() and 'r_value3' in locals():
    print("7. Correlation between infrastructure and crop price:")
    print(f"   - Banks vs. Price: r={r_value:.2f}")
    print(f"   - Post Offices vs. Price: r={r_value2:.2f}")
    print(f"   - Total Facilities vs. Price: r={r_value3:.2f}") 