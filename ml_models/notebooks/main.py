import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import subprocess
import importlib.util

def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'scipy']
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + ' '.join(missing_packages))
        return False
    return True

def run_module(module_name):
    """Run a specific analysis module."""
    print(f"\n{'='*80}")
    print(f"Running {module_name}...")
    print(f"{'='*80}\n")
    
    try:
        # Import the module
        spec = importlib.util.spec_from_file_location(module_name, f"models/{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"\n{module_name} completed successfully.\n")
        return True
    except Exception as e:
        print(f"Error running {module_name}: {str(e)}")
        return False

def check_data_files():
    """Check if all required data files exist."""
    required_files = [
        'data/RainfallDataset.csv',
        'data/LengthOfGrowingPeriod.csv',
        'data/SoilDataset.csv',
        'data/soil.csv',
        'data/Crop_recommendation.csv',
        'data/CropYield.csv',
        'data/CropPrice.csv',
        'data/Irrigation.csv',
        'data/Infrastructure.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"Missing required data files: {', '.join(missing_files)}")
        return False
    return True

def ensure_directories():
    """Ensure that all required directories exist."""
    os.makedirs('data/cleaned', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('images', exist_ok=True)

def generate_report():
    """Generate a comprehensive report from all analysis results."""
    cleaned_files = [f for f in os.listdir('data/cleaned') if f.endswith('.csv')]
    visualization_files = [f for f in os.listdir('images') if f.endswith('.png')]
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_file = f"reports/farming_agent_data_analysis_report_{timestamp}.html"
    
    # Create HTML report
    with open(report_file, 'w') as f:
        # Write header
        f.write(f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Farming Agent Data Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .section {{ margin-bottom: 30px; }}
                .visualization {{ margin: 20px 0; text-align: center; }}
                .visualization img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; padding: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                table, th, td {{ border: 1px solid #ddd; }}
                th, td {{ padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <h1>Farming Agent Data Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>
                    This report provides a comprehensive analysis of various agricultural datasets 
                    for the Farming Agent system. The analysis includes weather and climate data, 
                    soil and fertility data, crop growth and yield data, market price data, 
                    irrigation and water resources data, and infrastructure data.
                </p>
            </div>
        """)
        
        # Weather and Climate Section
        f.write("""
            <div class="section">
                <h2>1. Weather and Climate Analysis</h2>
                <p>
                    Analysis of rainfall patterns and growing periods across different regions,
                    which are crucial factors affecting agricultural productivity and planning.
                </p>
        """)
        
        # Add weather and climate visualizations
        weather_visualizations = [v for v in visualization_files if 
                                any(keyword in v for keyword in ['rainfall', 'growing_period'])]
        for viz in weather_visualizations:
            f.write(f"""
                <div class="visualization">
                    <h3>{viz.replace('.png', '').replace('_', ' ').title()}</h3>
                    <img src="../images/{viz}" alt="{viz}">
                </div>
            """)
        f.write("</div>")
        
        # Soil and Fertility Section
        f.write("""
            <div class="section">
                <h2>2. Soil and Fertility Analysis</h2>
                <p>
                    Analysis of soil types, nutrient levels, and their relation to crop requirements,
                    essential for optimizing fertilization and crop selection.
                </p>
        """)
        
        # Add soil visualizations
        soil_visualizations = [v for v in visualization_files if 
                            any(keyword in v for keyword in ['soil', 'nutrient', 'crop_soil'])]
        for viz in soil_visualizations:
            f.write(f"""
                <div class="visualization">
                    <h3>{viz.replace('.png', '').replace('_', ' ').title()}</h3>
                    <img src="../images/{viz}" alt="{viz}">
                </div>
            """)
        f.write("</div>")
        
        # Crop Yield Section
        f.write("""
            <div class="section">
                <h2>3. Crop Growth and Yield Analysis</h2>
                <p>
                    Analysis of crop yields across different regions and over time,
                    identifying the most productive crops and regions.
                </p>
        """)
        
        # Add yield visualizations
        yield_visualizations = [v for v in visualization_files if 
                                'yield' in v and 'price' not in v and 'infrastructure' not in v]
        for viz in yield_visualizations:
            f.write(f"""
                <div class="visualization">
                    <h3>{viz.replace('.png', '').replace('_', ' ').title()}</h3>
                    <img src="../images/{viz}" alt="{viz}">
                </div>
            """)
        f.write("</div>")
        
        # Market Price Section
        f.write("""
            <div class="section">
                <h2>4. Market Price Analysis</h2>
                <p>
                    Analysis of crop prices, their trends over time, and factors affecting price volatility,
                    which is essential for market decision-making.
                </p>
        """)
        
        # Add price visualizations
        price_visualizations = [v for v in visualization_files if 
                                'price' in v and 'infrastructure' not in v]
        for viz in price_visualizations:
            f.write(f"""
                <div class="visualization">
                    <h3>{viz.replace('.png', '').replace('_', ' ').title()}</h3>
                    <img src="../images/{viz}" alt="{viz}">
                </div>
            """)
        f.write("</div>")
        
        # Irrigation Section
        f.write("""
            <div class="section">
                <h2>5. Irrigation and Water Resources Analysis</h2>
                <p>
                    Analysis of irrigation patterns, water resource utilization, and irrigation efficiency,
                    critical for water resource management in agriculture.
                </p>
        """)
        
        # Add irrigation visualizations
        irrigation_visualizations = [v for v in visualization_files if 'irrigation' in v]
        for viz in irrigation_visualizations:
            f.write(f"""
                <div class="visualization">
                    <h3>{viz.replace('.png', '').replace('_', ' ').title()}</h3>
                    <img src="../images/{viz}" alt="{viz}">
                </div>
            """)
        f.write("</div>")
        
        # Infrastructure Section
        f.write("""
            <div class="section">
                <h2>6. Infrastructure Analysis</h2>
                <p>
                    Analysis of agricultural infrastructure like banks and post offices,
                    and their relation to agricultural productivity and market access.
                </p>
        """)
        
        # Add infrastructure visualizations
        infrastructure_visualizations = [v for v in visualization_files if 'infrastructure' in v]
        for viz in infrastructure_visualizations:
            f.write(f"""
                <div class="visualization">
                    <h3>{viz.replace('.png', '').replace('_', ' ').title()}</h3>
                    <img src="../images/{viz}" alt="{viz}">
                </div>
            """)
        f.write("</div>")
        
        # Cleaned Datasets Section
        f.write("""
            <div class="section">
                <h2>7. Cleaned Datasets</h2>
                <p>
                    The following datasets have been cleaned and processed for use in the Farming Agent system.
                </p>
                <table>
                    <tr>
                        <th>Dataset Name</th>
                        <th>Description</th>
                    </tr>
        """)
        
        for cleaned_file in cleaned_files:
            dataset_name = cleaned_file.replace('.csv', '').replace('_', ' ').title()
            f.write(f"""
                <tr>
                    <td>{cleaned_file}</td>
                    <td>{dataset_name} Dataset</td>
                </tr>
            """)
        
        f.write("""
                </table>
            </div>
        """)
        
        # Conclusion Section
        f.write("""
            <div class="section">
                <h2>8. Conclusion and Recommendations</h2>
                <p>
                    Based on the comprehensive analysis of the agricultural datasets, the following 
                    insights and recommendations emerge for the Farming Agent system:
                </p>
                <ul>
                    <li>The weather and climate analysis shows significant regional variation in rainfall patterns,
                        which should be factored into crop selection and planning.</li>
                    <li>Soil analysis reveals diverse soil compositions across regions, with implications for
                        crop suitability and fertilizer requirements.</li>
                    <li>Crop yield analysis identifies the most productive crops and regions, which should guide
                        resource allocation and farming strategies.</li>
                    <li>Market price analysis exposes price volatility and trends, which should inform market entry
                        and exit strategies.</li>
                    <li>Irrigation analysis uncovers water resource usage patterns and efficiency, which should 
                        guide water management strategies.</li>
                    <li>Infrastructure analysis shows the distribution of agricultural support facilities, which
                        should be considered in market access strategies.</li>
                </ul>
                
                <p>
                    These insights, combined with the cleaned datasets, provide a solid foundation for the 
                    development of an effective Farming Agent system.
                </p>
                
                <p>
                    For further improvements, additional data on pest and disease occurrences would enhance
                    the system's predictive capabilities.
                </p>
            </div>
        """)
        
        # Close HTML tags
        f.write("""
        </body>
        </html>
        """)
    
    print(f"\nReport generated: {report_file}")
    return report_file

def main():
    """Main function to run all analyses and generate the report."""
    print("\nFarming Agent Data Analysis\n")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check data files
    if not check_data_files():
        sys.exit(1)
    
    # Ensure directories exist
    ensure_directories()
    
    # List of analysis modules to run
    modules = [
        'weather_climate_analysis',
        'soil_fertility_analysis',
        'crop_yield_analysis',
        'market_price_analysis',
        'irrigation_analysis',
        'infrastructure_analysis'
    ]
    
    # Run each module
    successful_modules = []
    for module in modules:
        if run_module(module):
            successful_modules.append(module)
    
    # Generate report if at least one module ran successfully
    if successful_modules:
        report_file = generate_report()
        print(f"\n{len(successful_modules)}/{len(modules)} modules completed successfully.")
        
        # Try to open the report in the default browser
        try:
            if sys.platform == 'darwin':  # macOS
                subprocess.run(['open', report_file], check=False)
            elif sys.platform == 'win32':  # Windows
                os.startfile(report_file)
            elif sys.platform == 'linux':  # Linux
                subprocess.run(['xdg-open', report_file], check=False)
        except:
            print(f"Report generated but could not open automatically. Please open manually: {report_file}")
    else:
        print("\nNo modules completed successfully. Report generation skipped.")

if __name__ == "__main__":
    main() 