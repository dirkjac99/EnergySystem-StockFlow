# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 10:08:06 2025

@author: Dirk Jacobs
"""

import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl




#%% Modelling Options



# adjust to match path of inputfile
InputFile = r'StockFlow_Inputs_v4.0.xlsx' # input file can be found at https://github.com/dirkjac99/EnergySystem-StockFlow.git 

# select one the grid operators scenarios. middle of road (KM) is used in the msc thesis 
Scenario = 'KM' #choices: 'KM', 'EV', 'GB', 'HA', "Scenario X", "Scenario Y", "Scenario Z"

# interpolation method. linear is used in thesis. Other options include: ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘barycentric’, ‘polynomial’. see documentation for further information: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html
interpolation_method = 'linear' 

# enable or disable recycling. Enabling will subtract the recycled materials from both the inflow and outflow from its corresponds technology. 
Recycling_enabled = False

# enable or disable the export of results to Excel
Excel_export_enabled = True

# name model run for export to Excel
ModelRun = 'Baseline_no_recycling'

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



#%% implementation of dynamic recycling rates

# create empty dataframe
RecycledMaterials = InflowMaterials * 0.0

# only implement if recycling is enabled, toggle on or off at top of the code  
for (tech, year), row in dRecyclingRates.iterrows():
    # calculation of recycled materials
    RecycledMaterials.loc[(tech, year)] = OutflowMaterials.loc[(tech, year)]*dRecyclingRates.loc[(tech, year)]
    
    if Recycling_enabled == True:
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
StockPercentage = StockMaterials.div(dGlobalSupply_Aligned).mul(100)

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

StockPercentageSystem = (
    StockPercentage
      .groupby(level="Year")
      .sum()
      .sort_index()
)


#%% 

totalmaterialinflow = InflowMaterialsSystem.cumsum()
meansharesinflow = InflowPercentageSystem.mean()

# plt.plot(StockMaterialsSystem)
# plt.legend()

#%% # export modelling results to a structured Excel document
if Excel_export_enabled == True:
    paths = f"stock_flow_export_{ModelRun}.xlsx"

    # export_stock_flow_excel is a file that comes along in the model folder
    from export_stock_flow_excel import export_stock_flow_to_excel

    export_stock_flow_to_excel(
        StockCapacities=StockCapacities,
        InflowCapacities=InflowCapacities,   
        NASCapacities=NASCapacities,        
        OutflowCapacities=OutflowCapacities, 
        StockMaterials=StockMaterials,
        InflowMaterials=InflowMaterials,
        NASMaterials=NASMaterials,
        OutflowMaterials=OutflowMaterials,
        StockMaterialsSystem=StockMaterialsSystem,
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
    build_pivot_chart_with_slicers(paths)
    

    #build_pivot_pie_chart(paths)



#%% Helper: robustly prepare a Year x Materials dataframe from various shapes
# def _ensure_year_by_materials(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Accepts either:
#       - Year (index) x Materials (columns), or
#       - MultiIndex rows (e.g., Technology, Year) x Materials
#     Returns: Year (index) x Materials (columns)
#     """
#     # If the index is a MultiIndex and includes 'Year', aggregate by Year
#     if isinstance(df.index, pd.MultiIndex):
#         names = list(df.index.names)
#         if 'Year' in names:
#             # group only by Year level; sum over any other row levels (e.g., Technology)
#             df_prepared = df.groupby(level='Year').sum(numeric_only=True)
#         else:
#             # If there is no explicit 'Year', try the last level as a year-like index
#             last_level = names[-1]
#             df_prepared = df.groupby(level=last_level).sum(numeric_only=True)
#     else:
#         # Already Year-indexed
#         df_prepared = df.copy()

#     # Ensure the index is sorted by Year if numeric/integer-like
#     try:
#         df_prepared = df_prepared.sort_index()
#     except Exception:
#         pass

#     # Keep only numeric columns
#     df_prepared = df_prepared.select_dtypes(include=[np.number])

#     return df_prepared


#%% all code below is for making plots and is not essential for the modelling results
#-------------------------------------------------------------------------------------------

#%% Bar chart function
def plot_cumulative_materials(
    df: pd.DataFrame,
    start_year: int | None = None,
    end_year: int | None = None,
    title: str | None = None,
    top_n: int | None = None,
    legend_columns: int = 3,
    save_path: str | None = None,
    unit_label: str = "",
    log_scale: bool = False  # <<< NEW
):
    """
    Cumulative (sum over years) bar chart by material.
    - Accepts Year×Materials OR MultiIndex (Technology, Year)×Materials.
    - Drops all-zero materials.
    - Legend outside, palette suited for 30+ materials.
    - Optional year range and top_n to group the rest into 'Other'.
    - Optionally use log scale on x-axis.
    """

    # ---- helpers ----
    def ensure_year_by_materials(df_):
        if isinstance(df_.index, pd.MultiIndex):
            names = list(df_.index.names)
            if 'Year' in names:
                df_prep = df_.groupby(level='Year').sum(numeric_only=True)
            else:
                df_prep = df_.groupby(level=names[-1]).sum(numeric_only=True)
        else:
            df_prep = df_.copy()
        try:
            df_prep = df_prep.sort_index()
        except Exception:
            pass
        return df_prep.select_dtypes(include=[np.number])

    from matplotlib import cm
    def long_qual_palette(n):
        bases = [cm.get_cmap('tab20'), cm.get_cmap('tab20b'), cm.get_cmap('tab20c')]
        cols = []
        for m in bases:
            cols.extend([m(i) for i in range(m.N)])
        if n <= len(cols):
            return cols[:n]
        reps = int(np.ceil(n / len(cols)))
        return (cols * reps)[:n]
    # -----------------

    df_year_mat = ensure_year_by_materials(df)

    if (start_year is not None) or (end_year is not None):
        s = df_year_mat.index.min() if start_year is None else start_year
        e = df_year_mat.index.max() if end_year   is None else end_year
        df_year_mat = df_year_mat.loc[(df_year_mat.index >= s) & (df_year_mat.index <= e)]

    cumulative = df_year_mat.sum(axis=0, numeric_only=True)
    cumulative = cumulative[cumulative != 0]
    if cumulative.empty:
        print("Nothing to plot.")
        return

    cumulative = cumulative.sort_values(ascending=True)

    if (top_n is not None) and (top_n > 0) and (len(cumulative) > top_n):
        largest = cumulative.iloc[-top_n:]
        other_sum = cumulative.iloc[:-top_n].sum()
        cumulative = largest.copy()
        if other_sum != 0:
            cumulative.loc['Other'] = other_sum
        cumulative = cumulative.sort_values(ascending=True)

    n_bars = len(cumulative)
    fig_h = max(6.0, 0.38 * n_bars)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    colours = long_qual_palette(n_bars)
    y = np.arange(n_bars)
    ax.barh(y, cumulative.values, color=colours, edgecolor='none')
    ax.set_yticks(y, labels=cumulative.index)

    # Optional log scale
    if log_scale:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: f"{x:,.0f}"))

    period = ""
    if (start_year is not None) or (end_year is not None):
        s = df_year_mat.index.min() if start_year is None else start_year
        e = df_year_mat.index.max() if end_year   is None else end_year
        period = f" ({s}–{e})"
    ax.set_title(title or f"Cumulative by Material{period}", pad=12)

    xlab = "Cumulative Total"
    if unit_label:
        xlab += f" [{unit_label}]"
    ax.set_xlabel(xlab)

    if not log_scale:
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: f"{x:,.0f}"))

    ax.grid(axis='x', linestyle=':', alpha=0.5)
    ax.grid(axis='y', visible=False)

    handles = [mpl.patches.Patch(facecolor=colours[i], label=str(name)) for i, name in enumerate(cumulative.index)]
    plt.tight_layout(rect=[0, 0, 0.78, 1])
    ax.legend(
        handles=handles,
        title="Materials",
        loc='upper left',
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        ncol=legend_columns,
        fontsize='small',
        title_fontsize='small'
    )
    ax.invert_yaxis()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

#%% line chart
def plot_selected_lines(
    df: pd.DataFrame,
    technologies: list[str] | None = None,
    materials: list[str] | None = None,
    title: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    log_scale: bool = False,
    legend_outside: bool = True,
    legend_cols: int = 2,
    linewidth: float = 2.0,
    # NEW (optional; defaults keep old behaviour):
    y_label: str | None = None,
    y_is_percent: bool = False,        # set True for NASPercentage / StockPercentage / etc.
    auto_expand_years_if_zero: bool = False
):
    """
    Flexible line plot for:
      1) Year × Technology (e.g., InflowCapacities), or
      2) (Technology, Year) × Materials (e.g., NASPercentage, OutflowMaterials, StockPercentage).

    - Legend labels show 'Technology — Material' when materials exist.
    - Optional log-scale.
    - y_label and y_is_percent control axis text/formatting.
    - NEW: auto_expand_years_if_zero widens year range only if the selected slice is all zero.
    """

    # ---- detect shape ----
    is_multi = isinstance(df.index, pd.MultiIndex)
    has_tech_year = is_multi and set(df.index.names) >= {"Technology", "Year"}

    # ---- gather series ----
    series = []

    if has_tech_year:
        df_num = df.select_dtypes(include=[np.number])

        # materials filter
        cols_full = list(df_num.columns)
        if materials is not None:
            cols_sel = [c for c in cols_full if c in set(materials)]
            if not cols_sel:
                print("No matching materials found in dataframe columns.")
                return
        else:
            cols_sel = cols_full

        # technologies filter
        all_techs = df_num.index.get_level_values("Technology").unique().tolist()
        techs = all_techs if technologies is None else [t for t in technologies if t in set(all_techs)]
        if not techs:
            print("No matching technologies found in dataframe index.")
            return

        for tech in techs:
            df_t = df_num.xs(tech, level="Technology")  # Year × Materials

            # robust numeric year index for slicing
            years = pd.to_numeric(df_t.index, errors='coerce')
            df_t = df_t.set_index(years).sort_index()
            df_t = df_t[cols_sel]

            # initial slice
            if (start_year is not None) or (end_year is not None):
                s = df_t.index.min() if start_year is None else start_year
                e = df_t.index.max() if end_year   is None else end_year
                df_slice = df_t.loc[(df_t.index >= s) & (df_t.index <= e)]
            else:
                df_slice = df_t

            # if the slice is all zero and auto-expand is enabled, widen to non-zero span
            if auto_expand_years_if_zero and (not df_slice.empty) and (df_slice.to_numpy() == 0).all():
                nz_any = (df_t != 0).any(axis=1)
                if nz_any.any():
                    s2, e2 = df_t.index[nz_any].min(), df_t.index[nz_any].max()
                    df_slice = df_t.loc[s2:e2]

            for mat in df_slice.columns:
                y = df_slice[mat].astype(float)
                if (y == 0).all():
                    continue
                series.append((y.index.values, y.values, f"{tech} — {mat}"))
    else:
        # Year × Technology
        df_num = df.select_dtypes(include=[np.number]).copy()
        # ensure numeric year index for slicing
        years = pd.to_numeric(df_num.index, errors='coerce')
        df_num = df_num.set_index(years).sort_index()

        if (start_year is not None) or (end_year is not None):
            s = df_num.index.min() if start_year is None else start_year
            e = df_num.index.max() if end_year   is None else end_year
            df_slice = df_num.loc[(df_num.index >= s) & (df_num.index <= e)]
        else:
            df_slice = df_num

        cols = df_slice.columns.tolist()
        use_cols = cols if technologies is None else [c for c in technologies if c in set(cols)]
        if not use_cols:
            print("No matching technologies found in dataframe columns.")
            return

        # if the slice is all zero and auto-expand is enabled, widen to non-zero span
        if auto_expand_years_if_zero and (not df_slice.empty) and (df_slice[use_cols].to_numpy() == 0).all():
            nz_any = (df_num[use_cols] != 0).any(axis=1)
            if nz_any.any():
                s2, e2 = df_num.index[nz_any].min(), df_num.index[nz_any].max()
                df_slice = df_num.loc[s2:e2]

        for tech in use_cols:
            y = df_slice[tech].astype(float)
            if (y == 0).all():
                continue
            series.append((y.index.values, y.values, f"{tech}"))

    if not series:
        print("Nothing to plot with the current filters.")
        return

    # ---- styling ----
    def long_palette(n: int):
        bases = [cm.get_cmap('tab20'), cm.get_cmap('tab20b'), cm.get_cmap('tab20c')]
        colours = []
        for m in bases:
            colours.extend([m(i) for i in range(m.N)])
        if n <= len(colours):
            return colours[:n]
        reps = int(np.ceil(n / len(colours)))
        return (colours * reps)[:n]

    colours = long_palette(len(series))
    fig, ax = plt.subplots(figsize=(12, 6))

    for (x, y, lbl), c in zip(series, colours):
        ax.plot(x, y, label=lbl, linewidth=linewidth, color=c)

    # ---- y-axis formatting ----
    def smart_number_fmt(v):
        if abs(v) >= 1_000_000:
            return f"{v/1_000_000:.1f}M"
        if abs(v) >= 1_000:
            return f"{v/1_000:.1f}k"
        if abs(v) < 1 and not y_is_percent:
            return f"{v:.3g}"
        return f"{v:,.0f}"

    def smart_percent_fmt(v):
        av = abs(v)
        if av == 0:
            return "0%"
        if av < 0.01:
            return f"{v:.3f}%"
        if av < 1:
            return f"{v:.2f}%"
        if av < 10:
            return f"{v:.1f}%"
        return f"{v:.0f}%"

    if log_scale:
        ax.set_yscale("log")
        formatter = mpl.ticker.FuncFormatter(lambda v, pos: smart_percent_fmt(v) if y_is_percent else smart_number_fmt(v))
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    else:
        formatter = mpl.ticker.FuncFormatter(lambda v, pos: smart_percent_fmt(v) if y_is_percent else smart_number_fmt(v))
        ax.yaxis.set_major_formatter(formatter)

    ax.set_xlabel("Year")
    ax.set_ylabel(y_label if y_label is not None else ("%" if y_is_percent else "Value"))

    if title is None:
        title = "Selected Technologies & Materials" if (is_multi and has_tech_year) else "Selected Technologies"
    ax.set_title(title, pad=10)

    ax.grid(True, axis='both', linestyle=':', alpha=0.5)

    # Legend
    if legend_outside:
        plt.tight_layout(rect=[0, 0, 0.78, 1])
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=False,
            ncol=legend_cols,
            fontsize='small',
            title="Series"
        )
    else:
        ax.legend(frameon=False, fontsize='small')

    plt.show()


#%% Call Bar chart
plot_cumulative_materials(
    InflowMaterialsSystem,               # select desired system dataframe, such as OutflowMaterialsSystem / NASMaterialsSystem / StockMaterialsSystem
    start_year=2000,
    end_year=2050,
    log_scale=True,
    title="Cumulative Inflow by Material",
    unit_label="t",
    top_n=30,            # keep top 30, group rest as 'Other' (handy for 30+ materials)
    legend_columns=3
)


#%% Call linechart
plot_selected_lines(
    InflowMaterials,            # select select dataframes such as: InflowMaterials, NASMaterials, InflowPercentage, StockMaterials etc. 
    technologies=["Wind offshore", 'Wind onshore'], # select technologies
    materials=["Phosphorus"],      #select materials
    title="Inflow Materials",           #change title
    y_label="Inflow material [t]",    #change Y label
    y_is_percent=False,             #change to true for percentage dataframes
    log_scale=False,
    start_year=2000, end_year=2050,   # even if this is all-zero…
    auto_expand_years_if_zero=True  # change when errors occur with displaying recycling values
)
