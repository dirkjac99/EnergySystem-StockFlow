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
import seaborn as sns
from scipy.interpolate import make_interp_spline, PchipInterpolator
import warnings


#%% Modelling Options
InputFile = r'EnergySystem_Inputs_vGreenlight.xlsx'

# choice KM, EV, GB and HA
Scenario = 'KM'

#building blocks, select true for enabling individual buildings. Middle of the road = All False, Reduced Material Future = All True

BB_MaxCAES = False

BB_Nuclear, n_nuclear_plants = False, 8

BB_hydrogen_MS = False
MS_PEM, MS_AE = 0.2, 0.8 #marketshares electrolyzers

BB_Iridium_Efficiency = False # tonnes per gigawatt, enhanced intensity iridium (0.05 t/GW) in PEM electrolysers

BB_Decreased_Battery_Capacity = False
Battery_storage_assumption = 0.025 # required battery capacity in terms of percentage of annual electricity generation

BB_Geared_Turbines = False

BB_CSi_solar = False

BB_lowcrmRF = False




#%% Scenario and Material Intensity inputs


InstalledCapacities = pd.read_excel(InputFile, sheet_name=Scenario)
MaterialIntensities = pd.read_excel(InputFile, sheet_name='MI Master')
Lifetimes = pd.read_excel(InputFile, sheet_name='Lifetime')
GlobalSupply = pd.read_excel(InputFile, sheet_name='Global Supply')

#setting indices for dataframes and transposing to desired format
# InstalledCapacities = InstalledCapacities.set_index(['Year'])
MaterialIntensities = MaterialIntensities.set_index(['Technology'])
Lifetimes = Lifetimes.set_index(['Technology'])
GlobalSupply = GlobalSupply.set_index(['Material'])
InstalledCapacities = InstalledCapacities.set_index('Year')
InstalledCapacities = InstalledCapacities.transpose()

# dropping irrelevant data in Global Supplies
GlobalSupply = GlobalSupply.loc[:, ~GlobalSupply.columns.str.contains('^Unnamed')]
GlobalSupply = GlobalSupply.drop(columns=['Symbol'])


#changing NaN values to zero
MaterialIntensities = MaterialIntensities.fillna(0)

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
    
    InstalledCapacities['Wind offshore'].loc[2040] = InstalledCapacities['Wind offshore'].loc[2040]+0.5*change_in_nuclear_plants*nuclear_cap_to_wind
    InstalledCapacities['Wind offshore'].loc[2050] = InstalledCapacities['Wind offshore'].loc[2050]+change_in_nuclear_plants*nuclear_cap_to_wind

#%% hydrogen market share building block

if BB_hydrogen_MS == True:    
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

if BB_Geared_Turbines == True:
    MI_Geared_Turbines = pd.read_excel(InputFile, sheet_name='BB geared turbines', index_col=0)
    for material in MI_Geared_Turbines.index:
        MaterialIntensities[material].loc['Wind onshore'] = MI_Geared_Turbines['Wind onshore'].loc[material]
        MaterialIntensities[material].loc['Wind offshore'] = MI_Geared_Turbines['Wind offshore'].loc[material]

    
#%%

if BB_CSi_solar == True:
    MI_CSi_solar = pd.read_excel(InputFile, sheet_name='BB C-Si solar', index_col=0)
    MI_CSi_solar = MI_CSi_solar.iloc[:, :-2]
    for material in MI_CSi_solar:
        MaterialIntensities[material].loc['Solar PV'] = MI_CSi_solar[material].loc['C-Si']
    

#%% decreased Battery capacity building block

if BB_Decreased_Battery_Capacity == True:
    InstalledCapacities['Batteries'] = InstalledCapacities['Batteries']/(0.05/Battery_storage_assumption)
    
#%% low Critical material redox flow batteries: Zinc-Iodine

if BB_lowcrmRF == True:
    MaterialIntensities['Vanadium'].loc['IDES (redox-flow)'] = 0
    MaterialIntensities['Zinc'].loc['IDES (redox-flow)'] = 134201.053 # t/GW, zinc intensity for zinc-iodine redox flow


#%% Interpolation of scenario

# Generate years for interpolation
years_interpolated = pd.Series(range(InstalledCapacities.index.min(), InstalledCapacities.index.max() + 1))
InstalledCapacitiesInterpolated = pd.DataFrame({"Year": years_interpolated})
InstalledCapacitiesInterpolated = InstalledCapacitiesInterpolated.set_index(['Year'])

# for year in InstalledCapacities.index:
#     InstalledCapacitiesInterpolated.loc[year] = InstalledCapacities.loc[year]

# Create a new complete index from 2019 to 2050
full_index = pd.Index(range(2019, 2051), name=InstalledCapacities.index.name)

# Reindex the dataframe to include all years
InstalledCapacitiesInterpolated = InstalledCapacities.reindex(full_index)

# interpolate capacities, interpolation method can be adjusted to polynomials
InstalledCapacitiesInterpolated = InstalledCapacitiesInterpolated.interpolate(method= 'linear')

#%% maximum compressed air energy storage building block

if BB_MaxCAES == True:
    Annual_Additional_CAES = 3.7/20 # Gigawatt, 3.7 GW increase spread out over 20 years
    Annual_Decreased_Batteries = 2.39/20 # Gigawatt, 2.8 GW decrease spread out over 20 years
    for year in InstalledCapacitiesInterpolated.loc[2030:].index:
        InstalledCapacitiesInterpolated['MDES (CAES)'].loc[year] = InstalledCapacitiesInterpolated['MDES (CAES)'].loc[year] + Annual_Additional_CAES*(year-2030)
        InstalledCapacitiesInterpolated['Batteries'].loc[year] = InstalledCapacitiesInterpolated['Batteries'].loc[year] - Annual_Decreased_Batteries*(year-2030)

#%% creation Stock and flows dataframes
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


#%% defining stock flow function for single technology

def StockFlow(tech):
    #input for survival curve
    TimeMax = len(InstalledCapacitiesInterpolated)
    TimeSteps = np.arange(len(InstalledCapacitiesInterpolated))
    
    # Standard deviations outside of scope, all STD are assumed to be 1
    StdDeviation = 1 
    
    #temporary LT to find error
    LT = Lifetimes['Lifetime'].loc[tech]

    # creating survival curve
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

#%% perform stockflow function for all individual technologies
for tech in InstalledCapacitiesInterpolated:
    column = StockFlow(tech)
    InflowCapacities[tech] = column['Inflow']
    OutflowCapacities[tech] = column['Outflow']
    NASCapacities[tech] = column['NAS']

# manually set convert negative inflows to zero
InflowCapacities[InflowCapacities < 0] = 0
OutflowCapacities[OutflowCapacities < 0] = 0


#%% calculating material inflows for each individual technologies

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

#dropping material for which the technology does not have an intensity
Wind_onshore_mat_inflow = Wind_onshore_mat_inflow.loc[:, (Wind_onshore_mat_inflow != 0).any(axis=0)]
Wind_offshore_mat_inflow = Wind_offshore_mat_inflow.loc[:, (Wind_offshore_mat_inflow != 0).any(axis=0)]
Solar_PV_mat_inflow = Solar_PV_mat_inflow.loc[:, (Solar_PV_mat_inflow != 0).any(axis=0)]
Waste_plant_mat_inflow = Waste_plant_mat_inflow.loc[:, (Waste_plant_mat_inflow != 0).any(axis=0)]
Nuclear_plant_mat_inflow = Nuclear_plant_mat_inflow.loc[:, (Nuclear_plant_mat_inflow != 0).any(axis=0)]
Coal_plant_mat_inflow = Coal_plant_mat_inflow.loc[:, (Coal_plant_mat_inflow != 0).any(axis=0)]
Methane_plant_mat_inflow = Methane_plant_mat_inflow.loc[:, (Methane_plant_mat_inflow != 0).any(axis=0)]
Hydrogen_plant_mat_inflow = Hydrogen_plant_mat_inflow.loc[:, (Hydrogen_plant_mat_inflow != 0).any(axis=0)]
RedoxFlow_mat_inflow = RedoxFlow_mat_inflow.loc[:, (RedoxFlow_mat_inflow != 0).any(axis=0)]
CAES_mat_inflow = CAES_mat_inflow.loc[:, (CAES_mat_inflow != 0).any(axis=0)]
Interconnection_mat_inflow = Interconnection_mat_inflow.loc[:, (Interconnection_mat_inflow != 0).any(axis=0)]
Curtailment_mat_inflow = Curtailment_mat_inflow.loc[:, (Curtailment_mat_inflow != 0).any(axis=0)]
Batteries_mat_inflow = Batteries_mat_inflow.loc[:, (Batteries_mat_inflow != 0).any(axis=0)]
Powertoheat_mat_inflow = Powertoheat_mat_inflow.loc[:, (Powertoheat_mat_inflow != 0).any(axis=0)]
Powertogas_mat_inflow = Powertogas_mat_inflow.loc[:, (Powertogas_mat_inflow != 0).any(axis=0)]
DSR_mat_inflow = DSR_mat_inflow.loc[:, (DSR_mat_inflow != 0).any(axis=0)]


#%% Material Flows full system

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

#dropping columns that contain no values
PercentageInflow = PercentageInflow.loc[:, (PercentageInflow != 0).any(axis=0)]
MaterialInflow = MaterialInflow.loc[:, (MaterialInflow!= 0).any(axis=0)]

# removing 2019 reference year for inflows as 2019 inflows are wrongly assumed to fill the initial stock
MaterialInflow = MaterialInflow.iloc[1:]
MaterialNAS = MaterialNAS.iloc[1:]
MaterialOutflow = MaterialOutflow.iloc[1:]
PercentageInflow = PercentageInflow.iloc[1:]






#%% cumulative inlfows per material

# Calculate cumulative inflows
cumulative_inflows = MaterialInflow.sum()

# Sort by descending inflow
cumulative_inflows_sorted = cumulative_inflows.sort_values(ascending=False)

# Plot using Axes for more control
fig, ax = plt.subplots(figsize=(14, 6))

bars = ax.bar(cumulative_inflows_sorted.index, cumulative_inflows_sorted.values, zorder=2)

# Add rotated value labels on bars
for bar in bars:
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.0f}',
                ha='left', va='bottom', fontsize=10, rotation=45)

# Set log scale and labels
ax.set_yscale('log')
ax.set_ylabel('Cumulative Inflows in tonnes')
ax.set_title('Cumulative Material Inflows in tonnes')

# Grid behind bars
# ax.grid(True, axis='y', which='both', linestyle='--', zorder=1)

# X-axis formatting
plt.title('Cumulative Inflows per material for entire system')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#%% plot stacked area chart for inflows, outflows and NAS

# this can be changed to MaterialInflow, MaterialOutflow and MaterialNAS
MaterialInflow.plot.area(stacked=True, figsize=(12, 6))

plt.title('Total Material Inflows Over Time')
plt.xlabel('Year')
plt.ylabel('Tonnes per year')
plt.legend(title='Material', loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=7)
plt.tight_layout()
plt.show()

#%%
#Compute max percentage inflow per material
max_per_material = PercentageInflow.max() #take highest percentages, skip inflow ref year
bars = max_per_material.sort_values(ascending=False)

#Create the plot
fig, ax = plt.subplots(figsize=(16, 6))

# Define colours based on thresholds
colors = [
    '#A50034' if val > 1 else      # red
    '#EC6842' if val >= 0.22 else   # orange
    '#00A6D6'                      # blue
    for val in bars.values
]

bar_container = ax.bar(bars.index, bars.values, color=colors)

#Add value labels to each bar
for bar in bar_container:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.00002, f'{height:.2f}%', 
            ha='center', va='bottom', fontsize=9)

#Add reference lines with labels
thresholds = {
    'Population (0.22%)': 0.22,
    'Energy (0.52%)': 0.52,
    'GDP (1.0%)': 1.0
}

for label, y in thresholds.items():
    ax.axhline(y=y, color='#5C5C5C', linestyle='--', linewidth=2)
    ax.text(len(bars)-0.5, y + 0.02, label, color='#5C5C5C', va='bottom', ha='right', fontsize=14)

#formatting
# plt.yscale('log') #logarithmic scale
ax.set_ylabel('Percentage (%)')
ax.set_title('Maximum share of global material supply')
ax.set_xticks(range(len(bars)))
ax.set_xticklabels(bars.index, rotation=45, ha='right')
ax.grid(False, axis='x') #suppres vertical gridlines
ax = plt.gca()  # Get current axis
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:g}')) #change 10^-1 notation to 0.1 for legend
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

#%% cumulative material inflow individual or groups of technologies

# # Calculate cumulative inflows for each material (sum over rows)
# # select technologies here
# cumulative_inflows = Wind_offshore_mat_inflow.sum() 

# # # use this line to combine two technologies
# # cumulative_inflows = Wind_onshore_mat_inflow.sum().add(Wind_offshore_mat_inflow.sum(), fill_value=0) 

# # Sort the materials by cumulative inflow in descending order
# cumulative_inflows_sorted = cumulative_inflows.sort_values(ascending=False)

# # Plot as a bar chart
# plt.figure(figsize=(12, 6))
# bars = cumulative_inflows_sorted.plot(kind='bar')

# # Add values on top of bars
# for i, val in enumerate(cumulative_inflows_sorted):
#     plt.text(i, val, f'{val:.1e}', ha='left', va='bottom', fontsize=11, rotation=45)

# plt.grid(True, axis='y')
# plt.ylabel('Cumulative Inflows in tonnes')
# plt.yscale('log')
# plt.rc('xtick', labelsize=15) 
# plt.rc('ytick', labelsize=15)
# plt.rc('axes', labelsize=15) 
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()

#%% plots for stock and flows with all technologies combined

# # plotting all Installed Capacities
# plt.figure(3)
# plt.plot(InstalledCapacitiesInterpolated, label = InstalledCapacitiesInterpolated.columns)
# plt.title("Installed Capacity per Technology")
# # plt.yscale("log") #using logarithmic scale, comment for normal scale
# plt.xlabel('Year')
# plt.ylabel('Capacity (GW)')
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
#           fancybox=True, shadow=True, ncol=4)
# plt.savefig('plot_material_InstalledCapacities')

#%% plot for all shares of global supply over time

# # # plotting all shares of global supply
# plt.figure(5)
# plt.plot(PercentageInflow, label = PercentageInflow.columns)
# plt.title("Percentage of global supply")
# # plt.yscale("log") #using logarithmic scale, comment for normal scale
# plt.xlabel('Year')
# plt.ylabel('Percentage')
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
#           fancybox=True, shadow=True, ncol=4)






#%% Material inflow over time for individual technologies

# # select technology here
# tech_to_plot = Powertogas_mat_inflow
# plt.figure(0)
# plt.plot(tech_to_plot, label = tech_to_plot.columns)
# plt.title("Material Inflows")
# # plt.yscale("log") #using logarithmic scale, comment for normal scale
# plt.xlabel('Year')
# plt.ylabel('Mass (tonnes)')
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
#           fancybox=True, shadow=True, ncol=4)
# plt.savefig('plot_material_inflows')