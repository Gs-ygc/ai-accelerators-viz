import pandas as pd
import matplotlib.pyplot as plt
import math

# Load the CSV file
# file_paths = ['peak_accelerators_ieee_hpec_2019.csv', 'peak_accelerators_ieee_hpec_2020.csv',
#               'peak_accelerators_ieee_hpec_2021.csv','peak_accelerators_ieee_hpec_2022.csv',
#               'peak_accelerators_ieee_hpec_2023.csv']
file_paths = ['peak_accelerators_ieee_hpec_2023.csv', 'peak_accelerators_ieee_hpec_2024.csv',
              'peak_accelerators_ieee_hpec_2025.csv']

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

# Function to calculate performance multiplier based on precision
def get_precision_multiplier(precision):
    if pd.isna(precision):
        return 1
    precision = str(precision).lower()
    if 'fp16' in precision or 'fp16.32' in precision or 'bf16' in precision or 'bf16.32' in precision:
        return 2
    elif 'fp32' in precision:
        return 4
    elif 'int16' in precision:
        return 2
    elif 'int4' in precision or 'int4.8' in precision:
        return 0.5
    return 1  # Default for int8 and others

# Filter for inference only, where PeakPerformance is missing or zero, and exclude Netcast
data_filtered = data[
    (data['PeakPerformance'] > 1000) & 
    (data['IorT'] == 'inference') & 
    (data['Product'] != 'Netcast')
]

# Remove duplicates based on Product name, keeping the latest entry
data_filtered = data_filtered.sort_values('Updated', ascending=True).drop_duplicates(subset=['Product'], keep='last')

# Apply precision normalization to get Int8 equivalent performance
data_filtered['PrecisionMultiplier'] = data_filtered['Precision'].apply(get_precision_multiplier)
data_filtered['PeakPerformance_Normalized'] = data_filtered['PeakPerformance'] * data_filtered['PrecisionMultiplier']

# Mapping colors for the new categories
category_colors = {'CPU': 'blue', 'GPU': 'green','dataflow':'orange' ,'Others': 'red'}

# Create figure after filtering
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xscale('log')
ax.set_yscale('log')

# Create annotation (initially hidden)
annot = ax.annotate("", xy=(0,0), xytext=(10,10),textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind, scatter, df):
    pos = scatter.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = ""
    for idx in ind["ind"]:
        row = df.iloc[idx]
        text += f"Product: {row['Product']}\n"
        text += f"Company: {row['Company']}\n"
        text += f"Technology: {row['Technology']}\n"
        text += f"Precision: {row['Precision']}\n"
        text += f"Power: {row['Power']:.2f}W\n"
        text += f"Peak Performance: {row['PeakPerformance_TOPS']:.2f} TOPS\n"
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.9)

# Convert normalized PeakPerformance to TOPS (Tera Operations Per Second)
data_filtered['PeakPerformance_TOPS'] = data_filtered['PeakPerformance_Normalized'] / 1e12

# Plot data points based on the new categories (CPU, GPU, Others)
scatter_plots = []
for category in category_colors.keys():
    subset = data_filtered[data_filtered['TechnologyCategory'] == category]
    if not subset.empty:
        scatter = ax.scatter(
            subset['Power'],
            subset['PeakPerformance_TOPS'],
            label=f'{category}',
            c=category_colors[category],
            marker='o',  # Use circle markers for all
            alpha=0.7
        )
        scatter_plots.append((scatter, subset))
        
        # Add product names as labels with offset
        for idx, row in subset.iterrows():
            # Calculate offset direction based on position in plot
            # Create abbreviated label (first and last letter)
            product_name = row['Product']
            if len(product_name) > 2:
                label = f"{product_name[0]}..{product_name[-1]}"
            else:
                label = product_name

            dx = 5  # smaller horizontal offset in points
            ax.annotate(label,
                       (row['Power'], row['PeakPerformance_TOPS']),
                       xytext=(dx, 0), textcoords='offset points',
                       fontsize=6, alpha=0.7,  # smaller font size
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=0.2))

# Customize axes
ax.set_xlabel('Peak Power (W)', fontsize='12')
ax.set_ylabel('Peak Performance (INT8 TOPS)', fontsize='12')
ax.set_title('Peak Performance (INT8 Equivalent) vs. Power of Hardware for AI by CPU, GPU, and Others', fontsize='12')

# Add hover function
def hover(event):
    if event.inaxes == ax:
        cont, ind = False, dict()
        for scatter, df in scatter_plots:
            cont, ind = scatter.contains(event)
            if cont:
                annot.set_visible(True)
                update_annot(ind, scatter, df)
                fig.canvas.draw_idle()
                break
        if not cont:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()

# Connect the hover function
fig.canvas.mpl_connect("motion_notify_event", hover)

# Add grid, legend, and display
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend(loc='upper left', title='Technology Category', bbox_to_anchor=(0, 1), borderaxespad=0)
plt.tight_layout()

# Save as static image (for basic display)
plt.savefig('ai_accelerators_viz.png', dpi=300, bbox_inches='tight')

# Save as interactive HTML using plotly
import plotly.express as px
import plotly.graph_objects as go

# Create a new interactive figure
fig_plotly = go.Figure()

# Add scatter plots for each category
for category in category_colors.keys():
    subset = data_filtered[data_filtered['TechnologyCategory'] == category]
    if not subset.empty:
        fig_plotly.add_trace(go.Scatter(
            x=subset['Power'],
            y=subset['PeakPerformance_TOPS'],
            mode='markers+text',
            name=category,
            text=subset['Product'],
            textposition="top center",
            hovertemplate="<b>Product:</b> %{text}<br>" +
                         "<b>Company:</b> %{customdata[0]}<br>" +
                         "<b>Technology:</b> %{customdata[1]}<br>" +
                         "<b>Precision:</b> %{customdata[2]}<br>" +
                         "<b>Power:</b> %{x:.2f}W<br>" +
                         "<b>Performance:</b> %{y:.2f} TOPS<br>",
            customdata=subset[['Company', 'Technology', 'Precision']].values,
            marker=dict(color=category_colors[category])
        ))

# Update layout
fig_plotly.update_layout(
    title='Peak Performance (INT8 Equivalent) vs. Power of Hardware for AI by CPU, GPU, and Others',
    xaxis_title='Peak Power (W)',
    yaxis_title='Peak Performance (INT8 TOPS)',
    xaxis_type="log",
    yaxis_type="log",
    hovermode='closest',
    showlegend=True
)

# Save as HTML
fig_plotly.write_html('ai_accelerators_viz.html')

# Display the plot
plt.show()
