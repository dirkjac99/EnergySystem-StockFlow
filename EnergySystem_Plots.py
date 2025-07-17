# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 15:48:54 2025

@author: 31629
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import matplotlib.lines as mlines



# ensure that the right path leads to the input excel file 
inputfile = r'EnergySystem_Results_overview.xlsx'

# loading data for further plotting
system = pd.read_excel(inputfile, sheet_name='Baseline')
wind_turbine = pd.read_excel(inputfile, sheet_name='BB wind turbines')
solar_pv = pd.read_excel(inputfile, sheet_name='BB c-si')
Nuclear_offshore_wind = pd.read_excel(inputfile, sheet_name='BB nuclear energy')
reduced_battery = pd.read_excel(inputfile, sheet_name='BB battery capacity')
low_crm_redox = pd.read_excel(inputfile, sheet_name='BB low crm redox')
max_caes = pd.read_excel(inputfile, sheet_name='BB max caes')
electrolyser_ms = pd.read_excel(inputfile, sheet_name='BB electrolyser ms')
electrolyser_iridium = pd.read_excel(inputfile, sheet_name='BB electrolyser iridium')
electrolyser_grouped = pd.read_excel(inputfile, sheet_name='BB electrolyser grouped')


baseline_per_tech = pd.read_excel(inputfile, sheet_name='baseline per tech').set_index("Material")
RMF_per_tech = pd.read_excel(inputfile, sheet_name='RMF per tech').set_index("Material")
peak_years = pd.read_excel(inputfile, sheet_name='peak years')
waterfall = pd.read_excel(inputfile, sheet_name='Waterfall').set_index('Material')

#%% mean share of global supply: baseline vs RMF

selected_materials = ['Vanadium', 'Germanium', 'Terbium', 'Iridium', 'Dysprosium', 'Lithium', 'Neodymium', 'Silicon', 'Praseodymium', 'Tellurium']
filtered_df = peak_years.set_index('Material').loc[selected_materials].reset_index()

# Calculate percentage change if not present
if 'Percentage Change' not in filtered_df.columns:
    filtered_df['Percentage Change'] = 100 * (filtered_df['RMF Mean Share'] - filtered_df['Baseline Mean Share']) / filtered_df['Baseline Mean Share']


# Define thresholds and colours
gdp_threshold = 1
pop_threshold = 0.22
colours = {'high': '#EC6842', 'mid': '#00B8C8', 'low': '#6CC24A'}

# Function to determine bar colour
def classify_share(value):
    if value > gdp_threshold:
        return colours['high']
    elif value > pop_threshold:
        return colours['mid']
    else:
        return colours['low']

# Plotting with updated threshold line colours and legend descriptions
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.35
x = range(len(filtered_df))
arrow_offset = 0.12
min_vertical_step = 0.02

# Colours for threshold lines
gdp_line_color = '#8B0000'  #  red
pop_line_color = '#00008B'  #  blue

# Classify bars
baseline_colours = [classify_share(v) for v in filtered_df['Baseline Mean Share']]
rmf_colours = [classify_share(v) for v in filtered_df['RMF Mean Share']]

# Bars
for i in x:
    ax.bar(i - bar_width/2, filtered_df['Baseline Mean Share'][i], width=bar_width, color=baseline_colours[i], zorder=3)
    ax.bar(i + bar_width/2, filtered_df['RMF Mean Share'][i], width=bar_width, color=rmf_colours[i], hatch='\\\\\\', edgecolor='white', zorder=3)

# Threshold lines
ax.axhline(gdp_threshold, color=gdp_line_color, linestyle='--', linewidth=2, alpha=0.5)
ax.axhline(pop_threshold, color=pop_line_color, linestyle='--', linewidth=2, alpha = 0.5)

# Set y-axis limit to start from 0
ax.set_ylim(top=15)

ax.axhspan(gdp_threshold, ax.get_ylim()[1], facecolor='#EC6842', alpha=0.1, label='Severely Disproportionate (>1.0%)')
ax.axhspan(pop_threshold, gdp_threshold, facecolor='#00B8C8', alpha=0.1, label='Disproportionate (0.22% - 1.0%)')
ax.axhspan(0, pop_threshold, facecolor='#6CC24A', alpha=0.1, label='Proportionate (<0.22%)')

# Arrows
for i, (b, r) in enumerate(zip(filtered_df['Baseline Mean Share'], filtered_df['RMF Mean Share'])):
    y_start = max(b, r) + arrow_offset + min_vertical_step
    x_start = i - bar_width/2
    x_end = i + bar_width/2
    ax.plot([x_start, x_start], [b, y_start], color='black', lw=1)
    ax.plot([x_start, x_end], [y_start, y_start], color='black', lw=1)
    ax.annotate('', xy=(x_end, r), xytext=(x_end, y_start),
                arrowprops=dict(arrowstyle="->", color='black', lw=2))

# Annotations
for i, pct in enumerate(filtered_df['Percentage Change']):
    ax.text(i, max(filtered_df['Baseline Mean Share'][i], filtered_df['RMF Mean Share'][i]) + 0.14,
            f"{pct:.1f}%", ha='center', va='bottom', fontsize=15)
    

# Legend elements
legend_elements = [
    mpatches.Patch(facecolor='white', hatch='', edgecolor='black', label='Middle of the Road'),
    mpatches.Patch(facecolor='white', hatch='\\\\\\', edgecolor='black', label='Reduced Material Future'),
    mpatches.Patch(facecolor=colours['high'], label='Severely Disproportionate (> 1.0%)'),
    mpatches.Patch(facecolor=colours['mid'], label='Disproportionate (0.22% - 1.0%)'),
    mpatches.Patch(facecolor=colours['low'], label='Proportionate (< 0.22%)'),
    mlines.Line2D([], [], color=gdp_line_color, linestyle='--', label=' Dutch share of global GDP (1.0%)'),
    mlines.Line2D([], [], color=pop_line_color, linestyle='--', label='Dutch share of global population (0.22%)')
]

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(filtered_df['Material'],fontsize=13)
ax.set_ylabel('Percentage of global supply [%]', fontsize= 14)
# ax.set_title('Baseline vs RMF Max Share with Disproportion Indicators and Thresholds')
ax.legend(handles=legend_elements, prop={'size': 12})
ax.grid(axis='y', zorder=0)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()

#%% Waterfall graph per material

# Select material to plot
material = 'Steel'
values = waterfall.loc[material]
labels = values.index.tolist()
steps = values.values

labels = [label.replace("Decreased Battery Capacity", "Decreased\nBattery\nCapacity") for label in labels]
labels = [label.replace("Geared Turbines", "Geared\nTurbines") for label in labels]
labels = [label.replace("Low CRM RF", "Low CRM\nRedox Flow") for label in labels]
labels = [label.replace("Hydrogen MS", "Hydrogen\nMarket Share") for label in labels]
labels = [label.replace("Iridium Efficiency", "Iridium\nEfficiency") for label in labels]
labels = [label.replace("RMF", "Reduced\nMaterial\nFuture") for label in labels]

# Compute deltas and starting positions
intervention_deltas = [steps[i] - steps[i - 1] for i in range(1, len(steps) - 1)]
starts = [steps[0]]
for delta in intervention_deltas[:-1]:
    starts.append(starts[-1] + delta)

bar_values = [steps[0]] + intervention_deltas + [steps[-1]]
bar_starts = [0] + starts + [0]

# Define colors
color_baseline = '#00B8C8'
color_increase = '#EC6842'
color_decrease = '#6CC24A'
colors = [color_baseline]
colors += [color_decrease if d < 0 else color_increase for d in intervention_deltas]
colors += [color_baseline]



fig, ax = plt.subplots(figsize=(13, 2.5))
bars = []
hatches = ['']  # no hatch for baseline
hatches += ['\\\\\\\\' if d > 0 else '\\\\\\\\' for d in intervention_deltas]
hatches += ['']  # no hatch for RMF

edgecolors = ['white' if i > 0 and i < len(labels) - 1 else 'none' for i in range(len(labels))]

for i, (label, value, bottom, color, hatch, edge) in enumerate(zip(labels, bar_values, bar_starts, colors, hatches, edgecolors)):
    bar = ax.bar(label, value, bottom=bottom, color=color, hatch=hatch, edgecolor=edge,alpha=1, linewidth=1.0)
    bars.append(bar)

    # Annotate Baseline and RMF
    if i == 0 or i == len(labels) - 1:
        ax.text(i, bottom + value + 0.03 * max(bar_values), f'{value:.2e}',
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

    # Annotate percentage for interventions
    elif bar_starts[i] != 0:
        pct = (value / bar_starts[i]) * 100
        sign = "+" if pct >= 0 else ""
        ax.text(i, bottom + value / 2, f"{sign}{pct:.1f}%",
                ha='center', va='center', fontsize=12, color='black')


# Updated legend to include hatch explanation

legend_elements = [
    Patch(facecolor=color_baseline, label='Scenario', edgecolor='none'),
    Patch(facecolor=color_increase, hatch='\\\\\\\\', label='Increase by\nintervention', edgecolor='white'),
    Patch(facecolor=color_decrease, hatch='\\\\\\\\', label='Decrease by\nintervention', edgecolor='white'),
]
ax.legend(handles=legend_elements, loc='lower center', ncol=3)

# Extend y-axis
max_label_y = max(bar_starts[i] + bar_values[i] for i in [0, len(bar_values) - 1]) * 1.2
ax.set_ylim(top=max_label_y)

# Aesthetics
# ax.set_title(f"Waterfall Plot for {material} (Final RMF as Reference Bar)", fontsize=14)
ax.set_ylabel("Cumulative\nInflow in tonnes", fontsize=15)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=13)
plt.yticks(fontsize=13)
# ax.grid(True, axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()


#%% change in cumulative demand for technologies affected by individual interventions

df = solar_pv # change to desired Intervention

# Filter out rows where both Baseline and Intervention are zero
df = df[(df['Baseline'] != 0) | (df['Intervention'] != 0)]

df['Change %'] *= 100  # Convert to percentage

baseline_color = '#EC6842'
intervention_color = '#00B8C8'

# Cleaned-up version of the top-materials log-scale bar chart with embedded engineering-format labels
df['Total'] = df['Baseline'] + df['Intervention']
df_top = df.sort_values(by='Total', ascending=False).copy()
df_top.reset_index(drop=True, inplace=True)

fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
x = range(len(df_top))

# Clip y-axis to handle log scale limits
y_max = 1e8

# Bars with engineering-format labels inside
for i in x:
    b = df_top['Baseline'].iloc[i]
    r = df_top['Intervention'].iloc[i]
    ax.bar(i - bar_width/2, b, width=bar_width, color=baseline_color, zorder=3)
    ax.bar(i + bar_width/2, r, width=bar_width, color=intervention_color, zorder=3)
    if b > 0:
        ax.text(i - bar_width/2, b / 2, f"{b:.2E}", ha='center', va='top', fontsize=12, rotation=90, color='white')
    if r > 0:
        ax.text(i + bar_width/2, r / 2, f"{r:.2E}", ha='center', va='top', fontsize=12, rotation=90, color='white')


# Percentage annotations above bars
for i, pct in enumerate(df_top['Change %']):
    if pd.notna(pct):
        top_val = max(df_top['Baseline'].iloc[i], df_top['Intervention'].iloc[i])
        if top_val > 0:
            ax.text(i, top_val * 1.2, f"{pct:.1f}%", ha='center', va='bottom', fontsize=12)

# Formatting
ax.set_yscale('log')
ax.set_ylim(1, y_max)
ax.set_xticks(x)
ax.set_xticklabels(df_top['Material'], fontsize=12, rotation=45, ha='right')
ax.set_ylabel('Cumulative inflow in tonnes', fontsize=13)
legend_elements = [
    mpatches.Patch(color=baseline_color, label='Baseline'),
    mpatches.Patch(color=intervention_color, label='Intervention')
]
ax.legend(handles=legend_elements, fontsize=11)
# ax.grid(axis='y', which='both', zorder=0, )
plt.tight_layout()
plt.show()




