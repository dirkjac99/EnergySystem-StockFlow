
import os
import pandas as pd
from typing import Optional, Iterable

# Optional: early-bind for speed + reliable constants (only needed for build_pivot_chart_with_slicers)
try:
    import win32com.client as win32  # type: ignore
except Exception:  # pragma: no cover
    win32 = None

# Numeric fallbacks (only used if constants aren’t resolved for some reason)
XL = {
    "xlDatabase": 1,
    "xlRowField": 1,
    "xlColumnField": 2,
    "xlPageField": 3,
    "xlSum": -4157,
    "xlAreaStacked": 76,        # <-- Stacked Area
    # "xlAreaStacked100": 77,   # For 100%% stacked area
}

# =============================
# Helpers for tidy Pivot tables
# =============================

def _build_pivot_tidy_from_capacities(
    stock_df: pd.DataFrame,
    inflow_df: pd.DataFrame,
    nas_df: pd.DataFrame,
    outflow_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a long-form table for Pivot use from wide capacity tables:
        Year | Technology | Flow | Value

    Flow order enforced as: Stock, NAS, Inflow, Outflow
    Designed for shape:
        - index = Year (numeric), columns = technologies, values = floats
    Also tolerates slightly messy dtypes (e.g., numeric-as-strings).
    """

    def _melt_year_index(df: pd.DataFrame, flow_label: str) -> pd.DataFrame:
        d = df.copy()

        # 1) Treat the existing index as Year (coerce to numeric, drop non-year rows)
        year_series = pd.to_numeric(pd.Series(d.index), errors="coerce")
        mask = year_series.notna().to_numpy()
        if mask.sum() == 0:
            # Nothing looks like a year; return empty
            return pd.DataFrame(columns=["Year", "Technology", "Flow", "Value"])
        d = d.iloc[mask, :]
        d.index = year_series[mask].astype(int).values
        d.index.name = "Year"

        # 2) Coerce all data columns to numeric where possible
        for c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

        # 3) Melt to long
        long = (
            d.reset_index()
             .melt(id_vars=["Year"], var_name="Technology", value_name="Value")
             .dropna(subset=["Value"])
        )
        long["Technology"] = long["Technology"].astype(str)
        long["Flow"] = flow_label
        return long[["Year", "Technology", "Flow", "Value"]]

    tidy = pd.concat(
        [
            _melt_year_index(stock_df, "Stock"),
            _melt_year_index(nas_df, "NAS"),
            _melt_year_index(inflow_df, "Inflow"),
            _melt_year_index(outflow_df, "Outflow"),
        ],
        ignore_index=True
    )

    if not tidy.empty:
        # Enforce desired Flow order
        flow_order = pd.api.types.CategoricalDtype(categories=["Stock","NAS","Inflow","Outflow"], ordered=True)
        tidy["Flow"] = tidy["Flow"].astype(flow_order)
        tidy = tidy.sort_values(["Technology", "Year", "Flow"]).reset_index(drop=True)

    return tidy

def _build_pivot_material_from_material_flow(material_flow_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the wide 'Material flow' sheet (materials on top level; Stock/NAS/Inflow/Outflow under each)
    into a tidy long table suitable for PivotTables:

        Technology | Year | Material | Flow | Value

    Expects:
      - index: MultiIndex (Technology, Year)  [as produced in export_stock_flow_to_excel]
      - columns: MultiIndex (Material, Flow)  with Flow in {'Stock','NAS','Inflow','Outflow'}
    """
    df = material_flow_df.copy()

    # Be tolerant to index/column names
    if isinstance(df.index, pd.MultiIndex):
        try:
            df.index = df.index.set_names(['Technology', 'Year'])
        except Exception:
            pass

    # Stack Material, then Flow -> long series with index (Technology, Year, Material, Flow)
    long = (
        df.stack(level=0)   # stack Material
          .stack(level=0)   # stack Flow (Stock/NAS/Inflow/Outflow)
          .reset_index()
    )
    long.columns = ["Technology", "Year", "Material", "Flow", "Value"]

    # Clean types
    long["Technology"] = long["Technology"].astype(str)
    long["Year"] = pd.to_numeric(long["Year"], errors="coerce")
    long["Value"] = pd.to_numeric(long["Value"], errors="coerce")

    # Drop empty rows
    long = long.dropna(subset=["Year", "Value"]).reset_index(drop=True)

    # Enforce desired Flow order
    if not long.empty:
        flow_order = pd.api.types.CategoricalDtype(categories=["Stock","NAS","Inflow","Outflow"], ordered=True)
        long["Flow"] = long["Flow"].astype(flow_order)
        long = long.sort_values(["Technology", "Year", "Material", "Flow"]).reset_index(drop=True)

    return long

# =============================
# Index validation / coercion
# =============================

def _validate_multiindex(df: pd.DataFrame, name: str):
    if not isinstance(df.index, pd.MultiIndex) or df.index.names != ['Technology', 'Year']:
        raise ValueError(f"{name} must have a MultiIndex index with levels ['Technology', 'Year']. "
                         f"Got: {df.index.names} (type: {type(df.index)})")

def _validate_year_index(df: pd.DataFrame, name: str):
    if isinstance(df.index, pd.MultiIndex) or df.index.name != 'Year':
        raise ValueError(f"{name} must have a single index named 'Year'. "
                         f"Got: {df.index.name} (type: {type(df.index)})")

def _looks_like_single_technology_index(df: pd.DataFrame) -> bool:
    if isinstance(df.index, pd.MultiIndex):
        return False
    if df.index.name == 'Technology':
        return True
    if (df.index.name is None) or (str(df.index.name) in {'0', 'Index'}):
        return True
    return False

def _ensure_multiindex(df: pd.DataFrame, name: str, allow_single_technology: bool = False) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex) and df.index.names == ['Technology', 'Year']:
        return df

    if allow_single_technology and _looks_like_single_technology_index(df):
        df2 = df.copy()
        df2.index = df2.index.set_names('Technology')
        year_level = [''] * len(df2.index)
        new_index = pd.MultiIndex.from_arrays([df2.index, year_level], names=['Technology', 'Year'])
        df2.index = new_index
        return df2

    raise ValueError(
        f"{name} must have a MultiIndex ['Technology','Year']"
        + (", or a single Index named 'Technology' (unnamed also accepted; will be upcast)" if allow_single_technology else "")
        + f". Got: {getattr(df.index, 'names', df.index.name)} (type: {type(df.index)})"
    )

def _normalize_capacity_index(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Ensure capacity DF has MultiIndex (Technology, Year).
    Cases:
      - Already MultiIndex -> return as-is.
      - Single 'Technology' index + a 'Year' column -> move Year column into index.
      - Single 'Technology' index only -> upcast with empty Year level.
      - Unnamed single index (or name '0') -> treat as 'Technology' per above rules.
    """
    if isinstance(df.index, pd.MultiIndex):
        if df.index.names != ['Technology','Year']:
            # Attempt to rename if order matches
            try:
                df2 = df.copy()
                df2.index = df2.index.set_names(['Technology','Year'])
                return df2
            except Exception:
                pass
        return df

    # Single index path
    df2 = df.copy()
    df2.index = df2.index.set_names('Technology')

    # If a Year column exists, use it
    candidate_cols = [c for c in df2.columns if str(c).strip().lower() == 'year']
    if candidate_cols:
        year_col = candidate_cols[0]
        new_index = pd.MultiIndex.from_arrays([df2.index, df2[year_col]], names=['Technology','Year'])
        df2 = df2.drop(columns=[year_col])
        df2.index = new_index
        return df2

    # Otherwise upcast with empty Year
    year_level = [''] * len(df2.index)
    new_index = pd.MultiIndex.from_arrays([df2.index, year_level], names=['Technology','Year'])
    df2.index = new_index
    return df2

def _add_system_as_technology(
    df_multi: pd.DataFrame,
    system_df: Optional[pd.DataFrame],
    tech_label: str = 'System',
) -> pd.DataFrame:
    if system_df is None:
        if isinstance(df_multi.index, pd.MultiIndex):
            return df_multi
        # be tolerant: if single-index materials slip through, upcast
        df_multi = _ensure_multiindex(df_multi, "df_multi", allow_single_technology=True)
        return df_multi

    _validate_year_index(system_df, "system_df")
    if not isinstance(df_multi.index, pd.MultiIndex):
        df_multi = _ensure_multiindex(df_multi, "df_multi", allow_single_technology=True)

    union_cols = df_multi.columns.union(system_df.columns)
    df_multi_aligned = df_multi.reindex(columns=union_cols)
    system_df_aligned = system_df.reindex(columns=union_cols)

    sys_idx = pd.MultiIndex.from_product(
        [[tech_label], system_df_aligned.index],
        names=['Technology', 'Year']
    )
    sys_multi = system_df_aligned.copy()
    sys_multi.index = sys_idx

    combined = pd.concat([sys_multi, df_multi_aligned], axis=0)

    # Reorder Technology with system first, and then sort by Year
    idx_df = combined.index.to_frame(index=False)
    tech = idx_df['Technology'].astype(str)
    others = sorted(t for t in tech.unique() if t != tech_label)
    cat = pd.Categorical(tech, categories=[tech_label] + others, ordered=True)
    idx_df['Technology'] = cat

    idx_df_sorted = idx_df.sort_values(['Technology', 'Year'])
    combined_sorted = combined.iloc[idx_df_sorted.index.values].copy()
    combined_sorted.index = pd.MultiIndex.from_frame(idx_df_sorted.astype({'Technology':'string'}))
    return combined_sorted

# =============================
# Materials-first column order
# =============================

def _materials_first_columns(
    blocks: Iterable[pd.DataFrame],
    block_names=('Stock','NAS','Inflow','Outflow')
) -> pd.DataFrame:
    """
    Re-arrange columns so that MATERIAL is the top level and each material groups its
    (Stock,NAS,Inflow,Outflow) underneath. All blocks must have the same row index.
    Columns are aligned by union of materials.
    """
    blocks = list(blocks)
    if len(blocks) != len(block_names):
        raise ValueError("blocks and block_names length mismatch")

    # Align columns to union
    union_cols = blocks[0].columns
    for b in blocks[1:]:
        union_cols = union_cols.union(b.columns)

    aligned = [b.reindex(columns=union_cols) for b in blocks]

    # Build nested dict: {material: {block: series}}
    pieces = {}
    for mat in union_cols:
        sub_cols = {}
        for bname, bdf in zip(block_names, aligned):
            if mat in bdf.columns:
                sub_cols[bname] = bdf[mat]
            else:
                # fill NaN series if missing
                sub_cols[bname] = pd.Series(index=bdf.index, dtype='float64')
        pieces[mat] = pd.concat(sub_cols, axis=1)

    # Concatenate materials on top level
    out = pd.concat(pieces, axis=1)
    # Keep materials alphabetical; flows appear as given in block_names
    return out

def _triptych_side_by_side_materials_first(
    stock: pd.DataFrame,
    inflow: pd.DataFrame,
    nas: pd.DataFrame,
    outflow: pd.DataFrame,
    allow_single_technology=False,
) -> pd.DataFrame:
    """
    Columns grouped by MATERIAL first, then (Stock,NAS,Inflow,Outflow).
    """
    stock = _ensure_multiindex(stock, "stock", allow_single_technology=allow_single_technology)
    inflow = _ensure_multiindex(inflow, "inflow", allow_single_technology=allow_single_technology)
    nas = _ensure_multiindex(nas, "nas", allow_single_technology=allow_single_technology)
    outflow = _ensure_multiindex(outflow, "outflow", allow_single_technology=allow_single_technology)

    return _materials_first_columns([stock, nas, inflow, outflow], ('Stock','NAS','Inflow','Outflow'))

# =============================
# Export function
# =============================

def export_stock_flow_to_excel(
    # Installed Capacity (can be MultiIndex or single 'Technology' index, optionally with 'Year' column)
    StockCapacities: pd.DataFrame,
    InflowCapacities: pd.DataFrame,
    NASCapacities: pd.DataFrame,
    OutflowCapacities: pd.DataFrame,

    # Materials
    StockMaterials: pd.DataFrame,
    InflowMaterials: pd.DataFrame,
    NASMaterials: pd.DataFrame,
    OutflowMaterials: pd.DataFrame,
    StockMaterialsSystem: Optional[pd.DataFrame] = None,
    InflowMaterialsSystem: Optional[pd.DataFrame] = None,
    NASMaterialsSystem: Optional[pd.DataFrame] = None,
    OutflowMaterialsSystem: Optional[pd.DataFrame] = None,

    # Percentages
    InflowPercentage: pd.DataFrame = None,
    NASPercentage: pd.DataFrame = None,
    OutflowPercentage: pd.DataFrame = None,
    InflowPercentageSystem: Optional[pd.DataFrame] = None,
    NASPercentageSystem: Optional[pd.DataFrame] = None,
    OutflowPercentageSystem: Optional[pd.DataFrame] = None,

    # Recycling
    RecycledMaterials: pd.DataFrame = None,
    RecycledMaterialsSystem: Optional[pd.DataFrame] = None,

    # Output
    path: str = "stock_flow_export.xlsx",
    system_label: str = "System",
):
    """
    Create an Excel workbook with 4 sheets:
      1) Installed Capacity — MATERIALS on top; under each material: Stock, NAS, Inflow, Outflow.
         Accepts MultiIndex (Technology,Year) OR single 'Technology' index.
         If a 'Year' column exists, it will be moved into the index.
      2) Material flow — MATERIALS on top; under each material: Stock, NAS, Inflow, Outflow,
         with system values prepended as Technology==system_label if provided.
      3) Percentage — same layout as (2) but only NAS/Inflow/Outflow (no Stock).
      4) Recycled Materials — single block (materials as columns), system rows prepended if provided.
      + Pivot Technology, Pivot Material sheets to feed charts/dashboards.
    """
    # --- Installed Capacity: normalize indices and rearrange columns ---
    stk_cap = _normalize_capacity_index(StockCapacities, "StockCapacities")
    inf_cap = _normalize_capacity_index(InflowCapacities, "InflowCapacities")
    nas_cap = _normalize_capacity_index(NASCapacities, "NASCapacities")
    out_cap = _normalize_capacity_index(OutflowCapacities, "OutflowCapacities")

    cap_sheet = _triptych_side_by_side_materials_first(
        stk_cap, inf_cap, nas_cap, out_cap, allow_single_technology=False
    )

    pivot_tidy = _build_pivot_tidy_from_capacities(
        stock_df=StockCapacities,
        inflow_df=InflowCapacities,
        nas_df=NASCapacities,
        outflow_df=OutflowCapacities,
    )

    # Sanity check
    if pivot_tidy.empty:
        raise RuntimeError(
            "pivot_tidy is empty. Check that Stock/NAS/Inflow/Outflow capacity tables have Year as index "
            "and numeric values (or numeric-as-strings that can be coerced)."
        )

    # --- Materials: add system rows, then rearrange columns ---
    stock_mat_with_sys = _add_system_as_technology(StockMaterials, StockMaterialsSystem, tech_label=system_label)
    inflow_mat_with_sys = _add_system_as_technology(InflowMaterials, InflowMaterialsSystem, tech_label=system_label)
    nas_mat_with_sys = _add_system_as_technology(NASMaterials, NASMaterialsSystem, tech_label=system_label)
    outflow_mat_with_sys = _add_system_as_technology(OutflowMaterials, OutflowMaterialsSystem, tech_label=system_label)

    mat_sheet = _triptych_side_by_side_materials_first(
        stock_mat_with_sys, inflow_mat_with_sys, nas_mat_with_sys, outflow_mat_with_sys
    )

    # >>> Long-form pivot for materials (respects Stock, NAS, Inflow, Outflow order)
    pivot_material = _build_pivot_material_from_material_flow(mat_sheet)

    # --- Percentage: add system rows, then rearrange columns (no Stock here) ---
    if InflowPercentage is None or NASPercentage is None or OutflowPercentage is None:
        raise ValueError("InflowPercentage, NASPercentage, and OutflowPercentage must all be provided for the Percentage sheet.")

    inflow_pct_with_sys = _add_system_as_technology(InflowPercentage, InflowPercentageSystem, tech_label=system_label)
    nas_pct_with_sys = _add_system_as_technology(NASPercentage, NASPercentageSystem, tech_label=system_label)
    outflow_pct_with_sys = _add_system_as_technology(OutflowPercentage, OutflowPercentageSystem, tech_label=system_label)

    # Keep NAS, Inflow, Outflow order for the percentage sheet
    def _pct_columns_first(blocks: Iterable[pd.DataFrame], names=('NAS','Inflow','Outflow')) -> pd.DataFrame:
        blocks = list(blocks)
        if len(blocks) != len(names):
            raise ValueError("blocks and names length mismatch")
        union_cols = blocks[0].columns
        for b in blocks[1:]:
            union_cols = union_cols.union(b.columns)
        aligned = [b.reindex(columns=union_cols) for b in blocks]
        pieces = {}
        for mat in union_cols:
            sub_cols = {}
            for nm, bdf in zip(names, aligned):
                if mat in bdf.columns:
                    sub_cols[nm] = bdf[mat]
                else:
                    sub_cols[nm] = pd.Series(index=bdf.index, dtype='float64')
            pieces[mat] = pd.concat(sub_cols, axis=1)
        return pd.concat(pieces, axis=1)

    pct_sheet = _pct_columns_first([nas_pct_with_sys, inflow_pct_with_sys, outflow_pct_with_sys], ('NAS','Inflow','Outflow'))

    # --- Recycled Materials: add system rows (if provided); keep simple columns (materials) ---
    if RecycledMaterials is None:
        raise ValueError("RecycledMaterials must be provided for the final sheet.")
    recycled_with_sys = _add_system_as_technology(RecycledMaterials, RecycledMaterialsSystem, tech_label=system_label)

    # --- Write to Excel ---
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        cap_sheet.to_excel(writer, sheet_name="Installed Capacity")
        mat_sheet.to_excel(writer, sheet_name="Material flow")
        pct_sheet.to_excel(writer, sheet_name="Percentage")
        recycled_with_sys.to_excel(writer, sheet_name="Recycled Materials")
        pivot_tidy.to_excel(writer, sheet_name="Pivot Technology", index=False)
        pivot_material.to_excel(writer, sheet_name="Pivot Material", index=False)

        for sheet_name in ["Installed Capacity", "Material flow", "Percentage", "Recycled Materials", "Pivot Technology", "Pivot Material"]:
            worksheet = writer.sheets[sheet_name]
            # Freeze: for materials-first columns, there are 2 header rows and 2 index columns
            worksheet.freeze_panes(2, 2)

    return path

# ======================================
# Optional: Build Pivot Chart + Slicers
# ======================================

def build_pivot_chart_with_slicers(xlsx_path: str):
    """
    Builds a PivotTable + Stacked Area PivotChart on 'Charts' from tidy data on 'Pivot Material':
      Technology | Year | Material | Flow | Value

    Requires Windows + Excel + pywin32. If not available, this function will raise.
    """
    if win32 is None:
        raise RuntimeError("pywin32 / Excel not available in this environment.")

    excel = win32.gencache.EnsureDispatch("Excel.Application")
    excel.Visible = False

    try:
        # Constants helper
        try:
            from win32com.client import constants as C  # type: ignore
        except Exception:
            C = None

        def K(name):
            return getattr(C, name) if C is not None else XL[name]

        wb = excel.Workbooks.Open(os.path.abspath(xlsx_path))

        # Source sheet with tidy table
        try:
            src_ws = wb.Worksheets("Pivot Material")
        except Exception:
            raise RuntimeError("Sheet 'Pivot Material' not found in workbook")

        # Create/clear destination sheet
        try:
            view_ws = wb.Worksheets("Charts")
            view_ws.Cells.Clear()
        except Exception:
            view_ws = wb.Worksheets.Add()
            view_ws.Name = "Charts"

        src_range = src_ws.UsedRange

        # Pivot cache and table
        pc = wb.PivotCaches().Create(
            SourceType=K("xlDatabase"),
            SourceData=src_range
        )
        pt = pc.CreatePivotTable(
            TableDestination=view_ws.Range("A3"),
            TableName="PM_Pivot"
        )

        # Configure fields
        pf_year = pt.PivotFields("Year");       pf_year.Orientation = K("xlRowField")
        pf_flow = pt.PivotFields("Flow");       pf_flow.Orientation = K("xlColumnField")
        pf_val  = pt.AddDataField(pt.PivotFields("Value"), "Sum of Value", K("xlSum"))

        # Report filters (slicers will also be added)
        pf_tech = pt.PivotFields("Technology"); pf_tech.Orientation = K("xlPageField")
        pf_mat  = pt.PivotFields("Material");   pf_mat.Orientation = K("xlPageField")

        # --- Stacked Area PivotChart ---
        shp = view_ws.Shapes.AddChart2(240, K("xlAreaStacked"))
        chart = shp.Chart
        chart.SetSourceData(pt.TableRange2)  # link to the pivot

        # Position/size chart
        shp.Left   = view_ws.Range("H2").Left
        shp.Top    = view_ws.Range("H2").Top
        shp.Width  = 1080
        shp.Height = 300

        # Slicers (avoid line-continuation backslashes; use intermediate variables)
        sc = wb.SlicerCaches
        tech_cache = sc.Add2(pt, "Technology", "Technology")
        tech_cache.Slicers.Add(view_ws,
                               Top=view_ws.Range("A1").Top,
                               Left=view_ws.Range("A1").Left,
                               Height=150,
                               Width=200)
        mat_cache = sc.Add2(pt, "Material", "Material")
        mat_cache.Slicers.Add(view_ws,
                              Top=view_ws.Range("C1").Top,
                              Left=view_ws.Range("C1").Left,
                              Height=150,
                              Width=200)

        wb.Close(SaveChanges=True)
    finally:
        excel.Quit()
