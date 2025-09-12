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
InputFile = r'StockFlow_Inputs_Generic_v1.xlsx' # input file can be found at https://github.com/dirkjac99/EnergySystem-StockFlow.git 

#select scenario
Scenario = 'Scenario X' #choices: , "Scenario X", "Scenario Y"

# interpolation method. linear is used in thesis. Other options include: ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘barycentric’, ‘polynomial’. see documentation for further information: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html
interpolation_method = 'linear' 

# enable or disable recycling
Recycling_enabled = True

# enable or disable the export of results to Excel
Excel_export_enabled = True

# name model run for export to Excel
ModelRun = 'Generic1_recycling'

#%% Scenario and Material Intensity inputs

# loading relevant data from input excel sheet
InstalledCapacities = pd.read_excel(InputFile, sheet_name=Scenario)
DynamicLifetimes = pd.read_excel(InputFile, sheet_name='dLifetime')
StandardDeviation = pd.read_excel(InputFile, sheet_name='Standard Deviation')

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

# copy of installed capacity to avoid errors in merging historical stocks with installed capacities 
InstalledCapacities_original = InstalledCapacities.copy()


#%% interpolation of scenario with historical stocks from CBS

InstalledCapacitiesInterpolated = InstalledCapacities * 0.0

    
# # Generate years for interpolation
years_interpolated = pd.Series(range(InstalledCapacities.index.min(), InstalledCapacities.index.max() + 1))
# InstalledCapacitiesInterpolated = pd.DataFrame({"Year": years_interpolated})
# InstalledCapacitiesInterpolated = InstalledCapacitiesInterpolated.set_index(['Year'])

# # interpolate capacities, interpolation method can be adjusted to polynomials
InstalledCapacitiesInterpolated = InstalledCapacities.interpolate(method= interpolation_method)
   
InstalledCapacitiesInterpolated = InstalledCapacitiesInterpolated.fillna(0.0)

#storing capacities in similar name for flow format
StockCapacities = InstalledCapacitiesInterpolated

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
StockMaterials = dMI * 0.0

# still gives some future warning, fix this
# multiplication of each capacity flow with ith material intensity
for (tech, year), row in dMI.iterrows():
    InflowMaterials.loc[(tech, year)] = InflowCapacities.at[year, tech] * dMI.loc[(tech, year)]
    OutflowMaterials.loc[(tech, year)] = OutflowCapacities.at[year, tech] * dMI.loc[(tech, year)]
    NASMaterials.loc[(tech, year)] = NASCapacities.at[year, tech] * dMI.loc[(tech, year)]

StockMaterials = NASMaterials.groupby(level=0).cumsum()

#%% creation of material stock

# for (tech, year), row in dMI.iterrows():
#     StockMaterials.loc[(tech, year)] = NASMaterials


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

StockMaterialsSystem = (
    StockMaterials
      .groupby(level="Year")
      .sum()
      .sort_index()
)


#%% # export modelling results to a structured Excel document
if Excel_export_enabled == True:
    paths = f"stock_flow_export_{ModelRun}.xlsx"

    from export_stock_flow_excel_generic import export_stock_flow_to_excel

    export_stock_flow_to_excel(
        StockCapacities=StockCapacities,
        InflowCapacities=InflowCapacities,
        NASCapacities=NASCapacities,
        OutflowCapacities=OutflowCapacities,
        StockMaterials=StockMaterials,
        InflowMaterials=InflowMaterials,
        NASMaterials=NASMaterials,
        OutflowMaterials=OutflowMaterials,
        StockMaterialsSystem=StockMaterialsSystem,            # optional
        InflowMaterialsSystem=InflowMaterialsSystem,          # optional
        NASMaterialsSystem=NASMaterialsSystem,                # optional
        OutflowMaterialsSystem=OutflowMaterialsSystem,        # optional
        RecycledMaterials=RecycledMaterials,
        RecycledMaterialsSystem=RecycledMaterialsSystem,      # optional
        path=f"stock_flow_export_{ModelRun}.xlsx",
        system_label="System"
    )



    from export_stock_flow_excel_generic import build_pivot_chart_with_slicers
    #from export_stock_flow_excel import build_pivot_pie_chart
    build_pivot_chart_with_slicers(paths)
    

    #build_pivot_pie_chart(paths)

 