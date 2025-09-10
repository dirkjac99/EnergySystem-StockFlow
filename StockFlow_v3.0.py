# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 10:08:06 2025

@author: Dirk Jacobs
"""

import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt



#%% Modelling Options



# adjust to match path of inputfile
InputFile = r'StockFlow_Inputs_v3.1.xlsx' # input file can be found at https://github.com/dirkjac99/EnergySystem-StockFlow.git 

# select one the grid operators scenarios. middle of road (KM) is used in the msc thesis 
Scenario = 'KM' #choices: 'KM', 'EV', 'GB', 'HA', "Scenario X", "Scenario Y", "Scenario Z"

# interpolation method. linear is used in thesis. Other options include: ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘barycentric’, ‘polynomial’. see documentation for further information: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html
interpolation_method = 'linear' 

# enable or disable recycling
Recycling_enabled = True

# enable or disable the export to Excel
Excel_export_enabled = True

# name model run for export to Excel
ModelRun = 'Baseline_with_recycling4'

#%% Scenario and Material Intensity inputs

# loading relevant data from input excel sheet
InstalledCapacities = pd.read_excel(InputFile, sheet_name=Scenario)
DynamicLifetimes = pd.read_excel(InputFile, sheet_name='dLifetime')
StandardDeviation = pd.read_excel(InputFile, sheet_name='Standard Deviation')
dGlobalSupply = pd.read_excel(InputFile, sheet_name='dGlobal Supply').set_index('Year')

# loading dynamic material intensities and recycling rates into a multi-index dataframe
dMI = pd.read_excel(InputFile, sheet_name='dMaterial Intensity').set_index(['Technology', 'Year'])
dRecyclingRates = pd.read_excel(InputFile, sheet_name='dRecycling Rates').set_index(['Technology', 'Year'])

# get dataframe into desired format
InstalledCapacities = InstalledCapacities.set_index('Year')
InstalledCapacities = InstalledCapacities.drop(['Unit'], axis=1)
InstalledCapacities = InstalledCapacities.transpose()
DynamicLifetimes = DynamicLifetimes.set_index('Year')
StandardDeviation = StandardDeviation.set_index('Technology')
StandardDeviation = StandardDeviation['Standard Deviation']
HistoricalStocks = pd.read_excel(InputFile, sheet_name='Historical Stocks')
HistoricalStocks = HistoricalStocks.drop(['Unit'], axis=1)
HistoricalStocks = HistoricalStocks.set_index('Year').transpose()
HistoricalStocks = HistoricalStocks.loc[:2018].astype(float)


# dropping 2025 values to fix interpolation error that results in no inflow between 2023 and 2025
InstalledCapacities = InstalledCapacities.drop([2025])

# copy of installed capacity to avoid errors in merging historical stocks with installed capacities 
InstalledCapacities_original = InstalledCapacities.copy()


#%% interpolation of scenario with historical stocks from CBS


    
# merging the historical stocks for from CBS and the scenarios from the grid operators
InstalledCapacities = pd.concat([HistoricalStocks, InstalledCapacities_original])
    
# Generate years for interpolation
years_interpolated = pd.Series(range(InstalledCapacities.index.min(), InstalledCapacities.index.max() + 1))
InstalledCapacitiesInterpolated = pd.DataFrame({"Year": years_interpolated})
InstalledCapacitiesInterpolated = InstalledCapacitiesInterpolated.set_index(['Year'])

# Create a new complete index from 2000 to 2050
full_index = pd.Index(range(2000, 2051), name=InstalledCapacities.index.name)

# Reindex the dataframe to include all years
InstalledCapacitiesInterpolated = InstalledCapacities.reindex(full_index)

# interpolate capacities, interpolation method can be adjusted to polynomials
InstalledCapacitiesInterpolated = InstalledCapacitiesInterpolated.interpolate(method= interpolation_method)
   
InstalledCapacitiesInterpolated = InstalledCapacitiesInterpolated.fillna(0.0)

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

#%% defining stock flow function for single technology. function altered fromthe Material Flow Analysis II course, which part of Industrial Ecology master programme. head instructer: Tomer Fishman

def StockFlow(tech):
    #input for survival curve
    #input for survival curve
    TimeMax = len(InstalledCapacitiesInterpolated)
    TimeSteps = np.arange(len(InstalledCapacitiesInterpolated))

    # Standard deviations of lifetimes outside of scope, all STD are assumed to be 1
    STD = StandardDeviation.loc[tech]
    SurvivalCurveMatrix = pd.DataFrame(0.0, index=TimeSteps, columns=TimeSteps)

    # fill in values for survival curve matrix
    for time in TimeSteps:
        LT = DynamicLifetimes[tech].iloc[time]
        SurvivalCurve = scipy.stats.norm.sf(TimeSteps, loc=LT, scale=STD)    
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

#%% perform stockflow function for all installed capacities. 
for tech in InstalledCapacitiesInterpolated:
    column = StockFlow(tech)
    InflowCapacities[tech] = column['Inflow']
    OutflowCapacities[tech] = column['Outflow']
    NASCapacities[tech] = column['NAS']

# adding negative capacity inflows to the outflows
OutflowCapacities = OutflowCapacities - InflowCapacities.where(InflowCapacities < 0, 0)

# manually convert negative inflows to zero as inflows are by definition positive values
InflowCapacities[InflowCapacities < 0] = 0
OutflowCapacities[OutflowCapacities < 0] = 0

# convert inflow values for year 2000 to zero. Otherwise model assumes that the stock from 2000 must be met by the inflow for the year 2000
InflowCapacities.loc[2000] = 0

#%%

# creation of empty dataframes for material flows
InflowMaterials = dMI * 0.0
OutflowMaterials = dMI * 0.0
NASMaterials = dMI * 0.0

# still gives some future warning, fix this
# multiplication of each capacity flow with ith material intensity
for (tech, year), row in dMI.iterrows():
    InflowMaterials.loc[(tech, year)] = InflowCapacities.at[year, tech] * dMI.loc[(tech, year)]
    OutflowMaterials.loc[(tech, year)] = OutflowCapacities.at[year, tech] * dMI.loc[(tech, year)]
    NASMaterials.loc[(tech, year)] = NASCapacities.at[year, tech] * dMI.loc[(tech, year)]


#%% implementation of dynamic recycling rates

# create empty dataframe
RecycledMaterials = InflowMaterials * 0.0

# only implement if recycling is enabled, toggle on or off at top of the code
if Recycling_enabled == True:
    
    for (tech, year), row in dRecyclingRates.iterrows():
        # calculation of recycled materials
        RecycledMaterials.loc[(tech, year)] = OutflowMaterials.loc[(tech, year)]*dRecyclingRates.loc[(tech, year)]
        
        # subtracting the recycled materials from the in and outflows
        InflowMaterials.loc[(tech, year)]  = InflowMaterials.loc[(tech, year)] - RecycledMaterials.loc[(tech, year)]
        OutflowMaterials.loc[(tech, year)] = OutflowMaterials.loc[(tech, year)] - RecycledMaterials.loc[(tech, year)]


#%% summing all Technologies per Year for each material column
InflowMaterialsSystem = (
    InflowMaterials
      .groupby(level="Year")
      .sum()
      .sort_index()
)

OutflowMaterialsSystem = (
    OutflowMaterials
      .groupby(level="Year")
      .sum()
      .sort_index()
)

NASMaterialsSystem = (
    NASMaterials
      .groupby(level="Year")
      .sum()
      .sort_index()
)

RecycledMaterialsSystem = (
    RecycledMaterials
      .groupby(level="Year")
      .sum()
      .sort_index()
)

#%% expressing material flows in percentages of global supply

# Ensure same column order
dGlobalSupply = dGlobalSupply[InflowMaterials.columns]

# Reindex dGlobalsupply to match MultiIndex (Technology, Year)
dGlobalSupply_Aligned = dGlobalSupply.reindex(
    InflowMaterials.index.get_level_values("Year")
).set_index(InflowMaterials.index, inplace=False)

# Compute percentage of global supply
InflowPercentage = InflowMaterials.div(dGlobalSupply_Aligned).mul(100)
OutflowPercentage = OutflowMaterials.div(dGlobalSupply_Aligned).mul(100)
NASPercentage = NASMaterials.div(dGlobalSupply_Aligned).mul(100)

InflowPercentageSystem = (
    InflowPercentage
      .groupby(level="Year")
      .sum()
      .sort_index()
)

OutflowPercentageSystem = (
    OutflowPercentage
      .groupby(level="Year")
      .sum()
      .sort_index()
)

NASPercentageSystem = (
    NASPercentage
      .groupby(level="Year")
      .sum()
      .sort_index()
)
#%% # export modelling results to a structured Excel document
if Excel_export_enabled == True:
    paths = f"stock_flow_export_{ModelRun}.xlsx"

    # export_stock_flow_excel is a file that comes along in the model folder
    from export_stock_flow_excel import export_stock_flow_to_excel

    export_stock_flow_to_excel(
        InflowCapacities=InflowCapacities,   
        NASCapacities=NASCapacities,        
        OutflowCapacities=OutflowCapacities, 
        InflowMaterials=InflowMaterials,
        NASMaterials=NASMaterials,
        OutflowMaterials=OutflowMaterials,
        InflowMaterialsSystem=InflowMaterialsSystem,
        NASMaterialsSystem=NASMaterialsSystem,
        OutflowMaterialsSystem=OutflowMaterialsSystem,
        InflowPercentage=InflowPercentage,
        NASPercentage=NASPercentage,
        OutflowPercentage=OutflowPercentage,
        InflowPercentageSystem=InflowPercentageSystem,
        NASPercentageSystem=NASPercentageSystem,
        OutflowPercentageSystem=OutflowPercentageSystem,
        RecycledMaterials=RecycledMaterials,
        RecycledMaterialsSystem=RecycledMaterialsSystem,
        path=f"stock_flow_export_{ModelRun}.xlsx",
        system_label="System"
    )



    from export_stock_flow_excel import build_pivot_chart_with_slicers
    #from export_stock_flow_excel import build_pivot_pie_chart
    # build_pivot_chart_with_slicers(paths)
    build_pivot_chart_with_slicers(paths, keep_excel_open=False)

    #build_pivot_pie_chart(paths)

 