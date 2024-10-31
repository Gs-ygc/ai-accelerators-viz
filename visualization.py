import pandas as pd
import matplotlib.pyplot as plt
import math

# Load the CSV file
file_paths = ['peak_accelerators_ieee_hpec_2019.csv', 'peak_accelerators_ieee_hpec_2020.csv',
              'peak_accelerators_ieee_hpec_2021.csv','peak_accelerators_ieee_hpec_2022.csv',
              'peak_accelerators_ieee_hpec_2023.csv']

dataframes = [pd.read_csv(file) for file in file_paths]
data = pd.concat(dataframes)
# Define a function to categorize technologies into CPU, GPU, and Others
def categorize_technology(tech):
    if pd.isna(tech):  # Check if the value is NaN
        return tech  # Skip processing, return NaN as is
    tech = tech.lower()  # Convert to lowercase for case-insensitivity

    if 'cpu' in tech or 'multicore' in tech or 'manycore' in tech:
        return 'CPU'
    elif 'gpu' in tech:
        return 'GPU'
    elif 'dataflow' in tech:
        return 'dataflow'
    else:
        return 'Others'


# Apply the function to the Technology column to create a new category column
data['TechnologyCategory'] = data['Technology'].apply(categorize_technology)

# Filter out rows where PeakPerformance is missing or zero
data_filtered = data[data['PeakPerformance'] > 1000]

# Mapping colors for the new categories
category_colors = {'CPU': 'blue', 'GPU': 'green','dataflow':'orange' ,'Others': 'red'}

# Create figure after filtering
plt.figure(figsize=(12, 8))
plt.xscale('log')
plt.yscale('log')

# Plot data points based on the new categories (CPU, GPU, Others)
for category in category_colors.keys():
    subset = data_filtered[data_filtered['TechnologyCategory'] == category]
    if not subset.empty:
        plt.scatter(
            subset['Power'],
            subset['PeakPerformance'],
            label=f'{category}',
            c=category_colors[category],
            marker='o',  # Use circle markers for all
            alpha=0.7
        )
        for i, row in subset.iterrows():
            label = f"{row['Product']}"
            plt.text(
                row['Power'], row['PeakPerformance'], label,
                fontsize=8, alpha=0.7,
                bbox=dict(facecolor='none', alpha=0.05, edgecolor='none'),  # Add a white background to text
                ha='right', va='bottom',  # Horizontal and vertical alignment
                rotation=0,  # Optional: slight rotation for better spacing
                clip_on=True  # Ensure the text stays within the plot area when zooming
            )

# Customize axes
plt.xlabel('Peak Power (W)',fontsize='12')
plt.ylabel('Peak Performance (Op/s)',fontsize='12')
plt.title('Peak Performance vs. Power of Hardware for AI by CPU, GPU, and Others',fontsize='12')

# Add grid, legend, and display
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(loc='upper left', title='Technology Category', bbox_to_anchor=(0, 1), borderaxespad=0)
plt.tight_layout()
plt.show()
