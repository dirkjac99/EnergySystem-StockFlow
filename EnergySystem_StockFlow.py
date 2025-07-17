# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 11:36:47 2025

@author: Dirk Jacobs
"""
#%% importing packages

import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines

#%% Modelling Options

# adjust to match path of inputfile
InputFile = r'EnergySystem_Inputs.xlsx' # input file can be found at https://github.com/dirkjac99/EnergySystem-StockFlow.git 

# select one the grid operators scenarios. middle of road (KM) is used in the msc thesis 
Scenario = 'KM' #choices: KM, EV, GB and HA

# interpolation method. linear is used in thesis. other options include: ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘barycentric’, ‘polynomial’. see documentation for further information: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html
interpolation_method = 'linear' 

#interventions aka building blocks (BB), select true for enabling individual Interventions. Middle of the road = All False, Reduced Material Future = All True
BB_Geared_Turbines = False # 100% marketshare for geared windturbines both for on and offshore

BB_CSi_solar = False #change marketshare form 95% cristalline solar PV to 100%

BB_Nuclear, n_nuclear_plants = False, 8 # select how many nuclear plants to install, each nuclear plant reduces the amount of offshore wind

BB_Decreased_Battery_Capacity = False #reducing amount of battery storage needed
Battery_storage_assumption = 0.025 # required battery capacity in terms of percentage of annual electricity generation, baseline assumes 0.05%

BB_lowcrmRF = False #switch to Zinc-Iodine redox flow

BB_MaxCAES = False # maximum Compressed air energy storage reduces amount of batteries installed

BB_hydrogen_MS = False
MS_PEM, MS_AE = 0.2, 0.8 #marketshares electrolyzers proton exchange membrane (PEM) and Alkaline Electrolysers (AE)

BB_Iridium_Efficiency = False # tonnes per gigawatt, enhanced intensity iridium (0.05 t/GW) in PEM electrolysers


#%% Scenario and Material Intensity inputs

# loading relevant data from input excel sheet
InstalledCapacities = pd.read_excel(InputFile, sheet_name=Scenario)
MaterialIntensities = pd.read_excel(InputFile, sheet_name='MI Master')
Lifetimes = pd.read_excel(InputFile, sheet_name='Lifetime')
GlobalSupply = pd.read_excel(InputFile, sheet_name='Global Supply')

#setting indices for dataframes and transposing to desired format
MaterialIntensities = MaterialIntensities.set_index(['Technology'])
Lifetimes = Lifetimes.set_index(['Technology'])
GlobalSupply = GlobalSupply.set_index(['Material'])
InstalledCapacities = InstalledCapacities.set_index('Year')
InstalledCapacities = InstalledCapacities.transpose()

# dropping irrelevant data in Global Supplies
GlobalSupply = GlobalSupply.loc[:, ~GlobalSupply.columns.str.contains('^Unnamed')]
GlobalSupply = GlobalSupply.drop(columns=['Symbol'])


#changing NaN values to zero material intensity table
MaterialIntensities = MaterialIntensities.fillna(0)

#%% defining stock flow function for single technology. function altered fromthe Material Flow Analysis II course, which part of Industrial Ecology master programme. head instructer: Tomer Fishman

def StockFlow(tech):
    #input for survival curve
    TimeMax = len(InstalledCapacitiesInterpolated)
    TimeSteps = np.arange(len(InstalledCapacitiesInterpolated))
    
    # Standard deviations outside of scope, all STD are assumed to be 1
    StdDeviation = 1 
    
    # lifetimes from the input excel sheet
    LT = Lifetimes['Lifetime'].loc[tech]

    # creating survival curve based on lifetime and standard deviation
    SurvivalCurve = scipy.stats.norm.sf(TimeSteps, loc=LT, scale=StdDeviation)
    # create empty survival curve matrix
    SurvivalCurveMatrix = pd.DataFrame(0.0, index=TimeSteps, columns=TimeSteps)

    # fill in values for survival curve matrix
    for time in TimeSteps:
        SurvivalCurveMatrix.loc[time:, time] = SurvivalCurve[0:TimeMax - time]

    CohortSurvivalMatrix = pd.DataFrame(0, index=TimeSteps, columns=TimeSteps)

    # create cohort survival matrix with placeholder zeros
    StocksAndFlows = pd.DataFrame({'Stock':InstalledCapacitiesInterpolated[tech],
                                      'Inflow':np.zeros(len(InstalledCapacitiesInterpolated.index)),
                                      'Outflow':np.zeros(len(InstalledCapacitiesInterpolated.index)),
                                      'NAS':np.zeros(len(InstalledCapacitiesInterpolated.index))})

    for time in TimeSteps:
        StocksAndFlows['Inflow'].iloc[time] = (StocksAndFlows['Stock'].iloc[time] - CohortSurvivalMatrix.loc[time,:].sum()) / SurvivalCurveMatrix.loc[time, time]
        CohortSurvivalMatrix.loc[:,time] = SurvivalCurveMatrix.loc[:, time] * StocksAndFlows['Inflow'].iloc[time]

    CohortSurvivalMatrix.index = StocksAndFlows.index

    # calculate outflows and NAS using the cohort_surv_matrix
    StocksAndFlows['NAS'] = np.diff(StocksAndFlows['Stock'], prepend=0) # prepending 0 assumes no initial stock
    StocksAndFlows['Outflow'] = StocksAndFlows['Inflow'] - StocksAndFlows['NAS']
    
    return StocksAndFlows

#%% function to calculate inflows for all materials for individual technologies

def MaterialInflowfunc(Technology):
    TechnologyFlows = pd.DataFrame(InflowCapacities.index)

    for material in MaterialIntensities:
        column = pd.DataFrame({material:np.zeros(len(InflowCapacities.index))})
        TechnologyFlows = pd.concat([TechnologyFlows, column], axis=1)

    TechnologyFlows = TechnologyFlows.set_index(TechnologyFlows.columns[0])

    for year in InflowCapacities.index:
        Material_inflow_technology = InflowCapacities[Technology].loc[year]*MaterialIntensities.loc[Technology]
        TechnologyFlows.loc[year] = Material_inflow_technology
    return TechnologyFlows
#%% increased nuclear energy building block

#  half of the nuclear plants are installed in 2040, half in 2050
if BB_Nuclear == True:
    capacity_per_plant = 1.6 # GigaWatt
    capacity_borssele = 0.486 # Gigawatt for existing nuclear reactor in Borssele
    nuclear_cap_to_wind = 3.23 # GigaWatt, amount of offshore wind capacity needed to replace one nuclear plant
    change_in_nuclear_plants = 4 - n_nuclear_plants
    InstalledCapacities['Nuclear plant'] = [capacity_borssele,
                                            capacity_borssele,
                                            capacity_borssele,
                                            capacity_borssele,
                                            capacity_borssele,
                                            capacity_borssele+capacity_per_plant*0.5*n_nuclear_plants, 
                                            capacity_borssele+capacity_per_plant*n_nuclear_plants]
    
    # reducing offshore wind capacity based on the amount of nuclear plants installed 
    InstalledCapacities['Wind offshore'].loc[2040] = InstalledCapacities['Wind offshore'].loc[2040]+0.5*change_in_nuclear_plants*nuclear_cap_to_wind
    InstalledCapacities['Wind offshore'].loc[2050] = InstalledCapacities['Wind offshore'].loc[2050]+change_in_nuclear_plants*nuclear_cap_to_wind

#%% hydrogen market share building block and Iridium efficient electrolysers, both building blocks can be implemented independently 

if BB_hydrogen_MS == True:
    # changing material intensities to reflect new market shares. values are taken from 'MI P2G' sheet in the input excel file    
    MaterialIntensities.loc['Power-to-gas', 'Platinum'] = 0.17*MS_PEM + 0*MS_AE #tonnes per GigaWatt
    MaterialIntensities.loc['Power-to-gas', 'Titanium'] = 151.32*MS_PEM + 0*MS_AE #tonnes per GigaWatt
    MaterialIntensities.loc['Power-to-gas', 'Gold'] = 0.33*MS_PEM + 0*MS_AE #tonnes per GigaWatt
    MaterialIntensities.loc['Power-to-gas', 'Steel'] = 250.75*MS_PEM + 23746.39*MS_AE #tonnes per GigaWatt
    MaterialIntensities.loc['Power-to-gas', 'Aluminium'] = 19.05*MS_PEM + 0*MS_AE #tonnes per GigaWatt
    MaterialIntensities.loc['Power-to-gas', 'Zirconium'] = 0*MS_PEM + 70.57*MS_AE #tonnes per GigaWatt
    MaterialIntensities.loc['Power-to-gas', 'Nickel'] = 0*MS_PEM + 1797*MS_AE #tonnes per GigaWatt
    if BB_Iridium_Efficiency == True:
        MaterialIntensities.loc['Power-to-gas', 'Iridium'] = 0.05 * MS_PEM + 0*MS_AE # if both hydrogen ms and increased iridium efficiency building blocks are enabled
    else:
        MaterialIntensities.loc['Power-to-gas', 'Iridium'] = 0.45*MS_PEM + 0*MS_AE #tonnes per GigaWatt
    
if BB_hydrogen_MS == False and BB_Iridium_Efficiency == True:
    MI_P2G_eff_iridium = pd.read_excel(InputFile, sheet_name='BB Iridium Efficiency', index_col=0)
    MaterialIntensities.loc['Power-to-gas', 'Iridium'] = MI_P2G_eff_iridium['Intensity'].loc['Iridium']

#%% Geared turbines building block

#new Material intensity based on altered marketshares
if BB_Geared_Turbines == True:
    MI_Geared_Turbines = pd.read_excel(InputFile, sheet_name='BB geared turbines', index_col=0) 
    for material in MI_Geared_Turbines.index:
        MaterialIntensities[material].loc['Wind onshore'] = MI_Geared_Turbines['Wind onshore'].loc[material]
        MaterialIntensities[material].loc['Wind offshore'] = MI_Geared_Turbines['Wind offshore'].loc[material]

    
#%% Cristalline silicon building block

#new Material intensity based on altered marketshares
if BB_CSi_solar == True:
    MI_CSi_solar = pd.read_excel(InputFile, sheet_name='BB C-Si solar', index_col=0)
    MI_CSi_solar = MI_CSi_solar.iloc[:, :-2]
    for material in MI_CSi_solar:
        MaterialIntensities[material].loc['Solar PV'] = MI_CSi_solar[material].loc['C-Si']
    

#%% decreased Battery capacity building block

if BB_Decreased_Battery_Capacity == True:
    InstalledCapacities['Batteries'] = InstalledCapacities['Batteries']/(0.05/Battery_storage_assumption) #alter the amount of battery storage needed
    
#%% low Critical material redox flow batteries: Zinc-Iodine

if BB_lowcrmRF == True:
    MaterialIntensities['Vanadium'].loc['IDES (redox-flow)'] = 0
    MaterialIntensities['Zinc'].loc['IDES (redox-flow)'] = 134201.053 # t/GW, zinc intensity for zinc-iodine redox flow (see 'MI redox flow' sheet for calculation of zinc content)


#%% Interpolation of scenario

# Generate years for interpolation
years_interpolated = pd.Series(range(InstalledCapacities.index.min(), InstalledCapacities.index.max() + 1))
InstalledCapacitiesInterpolated = pd.DataFrame({"Year": years_interpolated})
InstalledCapacitiesInterpolated = InstalledCapacitiesInterpolated.set_index(['Year'])

# Create a new complete index from 2019 to 2050
full_index = pd.Index(range(2019, 2051), name=InstalledCapacities.index.name)

# Reindex the dataframe to include all years
InstalledCapacitiesInterpolated = InstalledCapacities.reindex(full_index)

# interpolate capacities, interpolation method can be adjusted to polynomials
InstalledCapacitiesInterpolated = InstalledCapacitiesInterpolated.interpolate(method= interpolation_method)

#%% maximum compressed air energy storage building block

if BB_MaxCAES == True:
    Annual_Additional_CAES = 3.7/20 # Gigawatt, 3.7 GW increase spread out over 20 years
    Annual_Decreased_Batteries = 2.39/20 # Gigawatt, 2.8 GW decrease spread out over 20 years
    for year in InstalledCapacitiesInterpolated.loc[2030:].index:
        InstalledCapacitiesInterpolated['MDES (CAES)'].loc[year] = InstalledCapacitiesInterpolated['MDES (CAES)'].loc[year] + Annual_Additional_CAES*(year-2030)
        InstalledCapacitiesInterpolated['Batteries'].loc[year] = InstalledCapacitiesInterpolated['Batteries'].loc[year] - Annual_Decreased_Batteries*(year-2030)

#%% creation of emppty capacity stock and flows dataframes
InflowCapacities = pd.DataFrame(InstalledCapacitiesInterpolated.index)
OutflowCapacities = pd.DataFrame(InstalledCapacitiesInterpolated.index)
NASCapacities = pd.DataFrame(InstalledCapacitiesInterpolated.index)

for tech in InstalledCapacitiesInterpolated:
    column = pd.DataFrame({tech:np.zeros(len(InstalledCapacitiesInterpolated.index))})
    InflowCapacities = pd.concat([InflowCapacities, column], axis=1)
    OutflowCapacities = pd.concat([OutflowCapacities, column], axis=1)
    NASCapacities = pd.concat([NASCapacities, column], axis=1)

InflowCapacities = InflowCapacities.set_index(InflowCapacities.columns[0])
OutflowCapacities = OutflowCapacities.set_index(OutflowCapacities.columns[0])
NASCapacities = NASCapacities.set_index(NASCapacities.columns[0])




#%% perform stockflow function for all installed capacities. 
for tech in InstalledCapacitiesInterpolated:
    column = StockFlow(tech)
    InflowCapacities[tech] = column['Inflow']
    OutflowCapacities[tech] = column['Outflow']
    NASCapacities[tech] = column['NAS']

# manually convert negative inflows to zero as inflows are by definition positive values
InflowCapacities[InflowCapacities < 0] = 0
OutflowCapacities[OutflowCapacities < 0] = 0


#%% translation of capacities into material flows for indivual technologies

Wind_onshore_mat_inflow = MaterialInflowfunc('Wind onshore')
Wind_offshore_mat_inflow = MaterialInflowfunc('Wind offshore')
Solar_PV_mat_inflow = MaterialInflowfunc('Solar PV')
Waste_plant_mat_inflow = MaterialInflowfunc('Waste plant')
Nuclear_plant_mat_inflow = MaterialInflowfunc('Nuclear plant')
Coal_plant_mat_inflow = MaterialInflowfunc('Coal plant (fossil + bio)')
Methane_plant_mat_inflow = MaterialInflowfunc('Methane plant')
Hydrogen_plant_mat_inflow = MaterialInflowfunc('Hydrogen plant')
RedoxFlow_mat_inflow = MaterialInflowfunc('IDES (redox-flow)')
CAES_mat_inflow = MaterialInflowfunc('MDES (CAES)')
Interconnection_mat_inflow =  MaterialInflowfunc('Interconnection')
Curtailment_mat_inflow = MaterialInflowfunc('Curtailment')
Batteries_mat_inflow = MaterialInflowfunc('Batteries')
Powertoheat_mat_inflow = MaterialInflowfunc('Power-to-heat')
Powertogas_mat_inflow = MaterialInflowfunc('Power-to-gas')
DSR_mat_inflow = MaterialInflowfunc('DSR')


#%% translation of capacities into material flows for entire system

# Matrix multiplication of capacities with material intensities to calculate material flows
MaterialInflow = InflowCapacities.dot(MaterialIntensities)
MaterialOutflow = OutflowCapacities.dot(MaterialIntensities)
MaterialNAS = NASCapacities.dot(MaterialIntensities)
MaterialStock = InstalledCapacitiesInterpolated.dot(MaterialIntensities)


#%% conversion of material flows to share of global supply

#create empty dataframe
PercentageInflow = pd.DataFrame()

# divide material flows by global supply, multiplication by 100 for percentages
for material in MaterialInflow:
    PercentageInflow[material] =100*(MaterialInflow[material]/GlobalSupply.loc[material, 'Global supply'])
    
#%% preparing results

# removing 2019 reference year for inflows as 2019 inflows are wrongly assumed to fill the initial stock
MaterialInflow = MaterialInflow.iloc[1:]
MaterialNAS = MaterialNAS.iloc[1:]
MaterialOutflow = MaterialOutflow.iloc[1:]
PercentageInflow = PercentageInflow.iloc[1:]

# dataframe to extract results for later comparison between scenarios. dataframe has no further use in the model
# inflow_results = pd.DataFrame({'Peak Year' : MaterialInflow.idxmax(), 'Max Inflow' : MaterialInflow.max(), 'Mean Inflow' : MaterialInflow.mean(), 'Max Share' :PercentageInflow.max(), 'Mean share' : PercentageInflow.mean()})

# cumulative material inflows for the interventions used to create the figures found in the supplementary information
cum_wind = Wind_onshore_mat_inflow.sum().add(Wind_offshore_mat_inflow.sum(), fill_value=0)
cum_solar = Solar_PV_mat_inflow.sum()
cum_nuclear_offshorewind = Nuclear_plant_mat_inflow.sum().add(Wind_offshore_mat_inflow.sum(), fill_value=0)
cum_battery = Batteries_mat_inflow.sum()
cum_redoxflow = RedoxFlow_mat_inflow.sum()
cum_caes_batteries = CAES_mat_inflow.sum().add(Batteries_mat_inflow.sum(), fill_value=0)
cum_electrolyser_ms = Powertogas_mat_inflow.sum()
cum_electrolyser_iridium = Powertogas_mat_inflow.sum()
cum_system = MaterialInflow.sum()

#%% cumulative material inflow graph

cum_system_filtered = cum_system[cum_system > 0].sort_values(ascending=False)

# Create plot
fig, ax = plt.subplots(figsize=(5, 7))
bars = ax.barh(cum_system_filtered.index, cum_system_filtered.values, color='#00B8C8')

# Set log scale
ax.set_xscale('log')
ax.invert_yaxis()
ax.set_xlabel('Cumulative inflow in tonnes (log scale)', fontsize=12)

# Annotate bars
n_smallest = 5 # remove smallest values outside of bar for better plotting
smallest_indices = cum_system_filtered.nsmallest(n_smallest).index
for bar, value, label in zip(bars, cum_system_filtered.values, cum_system_filtered.index):
    x = bar.get_width()
    text = f"{value:.2e}"
    if label in smallest_indices:
        ax.text(x + x * 0.05, bar.get_y() + bar.get_height()/2, text,
                va='center', ha='left', fontsize=10)
    else:
        ax.text(x - x * 0.05, bar.get_y() + bar.get_height()/2, text,
                va='center', ha='right', fontsize=10, color='white')

plt.yticks(fontsize=12)
plt.xticks(fontsize=12)

# Remove top and bottom whitespace
y_min, y_max = ax.get_ylim()
bar_height = bars[0].get_height()
ax.set_ylim(y_min - bar_height * 1.8, y_max + bar_height * 1.8)

plt.tight_layout()
plt.show()


#%% stacked area chart for all material inflows, outflows and NAS

# this can be changed to MaterialInflow, MaterialOutflow and MaterialNAS
MaterialInflow.plot.area(stacked=True, figsize=(12, 6))

plt.title('Total Material Inflows Over Time') #change title accordingly
plt.xlabel('Year')
plt.ylabel('Tonnes per year')
plt.legend(title='Material', loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=7)
plt.tight_layout()
plt.show()

#%% Stacked area chart for selected materials
# adjust to desired materials
selected_materials = ['Neodymium','Praseodymium','Dysprosium', 'Terbium', 'Yttrium'] #rare earth elements are selected in the thesis

# Filter the cumulative inflow for the selected materials
selected_data = MaterialNAS[selected_materials]

# Define the custom colours
custom_colors = ['#00B8C8', '#6CC24A', '#0076C2', '#5C5C5C', '#E03C31']

# Plotting the area chart
plt.figure(figsize=(12, 3))
plt.stackplot(
    selected_data.index,
    selected_data.T,
    labels=selected_materials,
    colors=custom_colors[:len(selected_materials)],
    zorder=3,
    alpha = 1
)
# plt.title('Annual Inflow per Year for Selected Materials')
plt.xlabel('Year', fontsize= 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize=14)
plt.ylim((0,600))
plt.ylabel('Annual NAS in tonnes', fontsize = 14)
plt.xlim(selected_data.index.min(), selected_data.index.max())
plt.legend(loc='upper left')
# plt.grid(True, axis='y', zorder=0, alpha = 0.7)
plt.tight_layout()
plt.show()


#%% plot Material inflow, outflow and NAS over time for all inidivual materials

# creating individual material flow plots for all materials
# figure_n=10
# for material in MaterialInflow:
#     plt.figure(figure_n)
#     MaterialInflow[material].plot(title=material, label='inflow')
#     MaterialOutflow[material].plot(label = 'outflow')
#     MaterialNAS[material].plot(label = 'NAS')
#     # MaterialStock[material].plot(label = 'stock')
#     plt.legend()
#     figure_n = figure_n+1


#%% plot for shares of global supply over time or selected material

selected_materials = ['Vanadium', 'Germanium', 'Terbium', 'Iridium', 'Dysprosium', 'Lithium', 'Neodymium', 'Silicon', 'Praseodymium', 'Tellurium']

# Ensure selected materials exist in the dataframe
valid_materials = [mat for mat in selected_materials if mat in PercentageInflow.columns]

# Define thresholds
gdp_threshold = 1.0
pop_threshold = 0.22

# Redefine the plot with the legend inside the graph at the upper left
fig, ax = plt.subplots(figsize=(12, 6))

# Plot material lines and collect handles
material_lines = []
for material in valid_materials:
    line, = ax.plot(PercentageInflow.index, PercentageInflow[material],
                    label=material, linewidth=2.5, alpha=0.9)
    material_lines.append(line)

# Define threshold line legend handles
gdp_line = mlines.Line2D([], [], color='#8B0000', linestyle='--', linewidth=1, label='Gross Domestic Product (1.0%)')
pop_line = mlines.Line2D([], [], color='#00008B', linestyle='--', linewidth=1, label='Population (0.22%)')


# Plot material lines and collect handles
material_lines = []
for material in valid_materials:
    line, = ax.plot(PercentageInflow.index, PercentageInflow[material],
                    label=material, linewidth=2.5, alpha=0.9)
    material_lines.append(line)

ax.set_xlim([2020, 2050])

# Set y-axis limit to start from 0
ax.set_ylim(bottom=0)

# Get updated y-axis top limit after setting bottom
y_top = ax.get_ylim()[1]
# Add shaded areas for thresholds
ax.axhspan(gdp_threshold, ax.get_ylim()[1], facecolor='#EC6842', alpha=0.1, label='Severely Disproportionate (>1.0%)')
ax.axhspan(pop_threshold, gdp_threshold, facecolor='#00B8C8', alpha=0.1, label='Disproportionate (0.22% - 1.0%)')
ax.axhspan(0, pop_threshold, facecolor='#6CC24A', alpha=0.1, label='Proportionate (<0.22%)')

# Threshold line legend handles
gdp_line = mlines.Line2D([], [], color='#8B0000', linestyle='--', linewidth=1, label='Gross Domestic Product (1.0%)')
pop_line = mlines.Line2D([], [], color='#00008B', linestyle='--', linewidth=1, label='Population (0.22%)')

# Combine all legend handles
all_legend_handles = [
    mlines.Line2D([], [], color='#EC6842', alpha=0.6, linewidth=10, label='Severely Disproportionate (>1.0%)'),
    mlines.Line2D([], [], color='#00B8C8', alpha=0.6, linewidth=10, label='Disproportionate (0.22% - 1.0%)'),
    mlines.Line2D([], [], color='#6CC24A', alpha=0.6, linewidth=10, label='Proportionate (<0.22%)'),
    gdp_line, pop_line,
] + material_lines

# Add combined legend inside the plot at the upper left
ax.legend(handles=all_legend_handles, title='Proportionality and Materials', title_fontsize=12, fontsize=11,
          loc='upper left', frameon=True, ncol=3)

# Plot threshold lines
ax.axhline(gdp_threshold, color='#8B0000', linestyle='--', linewidth=2)
ax.axhline(pop_threshold, color='#00008B', linestyle='--', linewidth=2)

# Labels and title
# ax.set_title("Percentage of Global Supply Over Time", fontsize=16, weight='bold')
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Percentage of Global Supply (%)', fontsize=14)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
ax.tick_params(axis='both', labelsize=14)

# Adjust layout
plt.tight_layout()
plt.show()











