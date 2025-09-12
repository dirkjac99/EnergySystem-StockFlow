
import os
import pandas as pd
from typing import Optional, Iterable

try:
    import win32com.client as win32  # type: ignore
except Exception:
    win32 = None

XL = {
    "xlDatabase": 1,
    "xlRowField": 1,
    "xlColumnField": 2,
    "xlPageField": 3,
    "xlSum": -4157,
    "xlAreaStacked": 76,
}

def _build_pivot_tidy_from_capacities(
    stock_df: pd.DataFrame,
    inflow_df: pd.DataFrame,
    nas_df: pd.DataFrame,
    outflow_df: pd.DataFrame,
) -> pd.DataFrame:
    def _melt_year_index(df: pd.DataFrame, flow_label: str) -> pd.DataFrame:
        d = df.copy()
        year_series = pd.to_numeric(pd.Series(d.index), errors="coerce")
        mask = year_series.notna().to_numpy()
        if mask.sum() == 0:
            return pd.DataFrame(columns=["Year", "Technology", "Flow", "Value"])
        d = d.iloc[mask, :]
        d.index = year_series[mask].astype(int).values
        d.index.name = "Year"
        for c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
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
        flow_order = pd.api.types.CategoricalDtype(categories=["Stock","NAS","Inflow","Outflow"], ordered=True)
        tidy["Flow"] = tidy["Flow"].astype(flow_order)
        tidy = tidy.sort_values(["Technology", "Year", "Flow"]).reset_index(drop=True)
    return tidy

def _build_pivot_material_from_material_flow(material_flow_df: pd.DataFrame) -> pd.DataFrame:
    df = material_flow_df.copy()
    if isinstance(df.index, pd.MultiIndex):
        try:
            df.index = df.index.set_names(['Technology', 'Year'])
        except Exception:
            pass
    long = (
        df.stack(level=0)
          .stack(level=0)
          .reset_index()
    )
    long.columns = ["Technology", "Year", "Material", "Flow", "Value"]
    long["Technology"] = long["Technology"].astype(str)
    long["Year"] = pd.to_numeric(long["Year"], errors="coerce")
    long["Value"] = pd.to_numeric(long["Value"], errors="coerce")
    long = long.dropna(subset=["Year", "Value"]).reset_index(drop=True)
    if not long.empty:
        flow_order = pd.api.types.CategoricalDtype(categories=["Stock","NAS","Inflow","Outflow"], ordered=True)
        long["Flow"] = long["Flow"].astype(flow_order)
        long = long.sort_values(["Technology", "Year", "Material", "Flow"]).reset_index(drop=True)
    return long

def _validate_multiindex(df: pd.DataFrame, name: str):
    if not isinstance(df.index, pd.MultiIndex) or df.index.names != ['Technology', 'Year']:
        raise ValueError(f"{name} must have a MultiIndex index with levels ['Technology', 'Year']. Got: {df.index.names} (type: {type(df.index)})")

def _validate_year_index(df: pd.DataFrame, name: str):
    if isinstance(df.index, pd.MultiIndex) or df.index.name != 'Year':
        raise ValueError(f"{name} must have a single index named 'Year'. Got: {df.index.name} (type: {type(df.index)})")

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
    if isinstance(df.index, pd.MultiIndex):
        if df.index.names != ['Technology','Year']:
            try:
                df2 = df.copy()
                df2.index = df2.index.set_names(['Technology','Year'])
                return df2
            except Exception:
                pass
        return df
    df2 = df.copy()
    df2.index = df2.index.set_names('Technology')
    candidate_cols = [c for c in df2.columns if str(c).strip().lower() == 'year']
    if candidate_cols:
        year_col = candidate_cols[0]
        new_index = pd.MultiIndex.from_arrays([df2.index, df2[year_col]], names=['Technology','Year'])
        df2 = df2.drop(columns=[year_col])
        df2.index = new_index
        return df2
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
    idx_df = combined.index.to_frame(index=False)
    tech = idx_df['Technology'].astype(str)
    others = sorted(t for t in tech.unique() if t != tech_label)
    cat = pd.Categorical(tech, categories=[tech_label] + others, ordered=True)
    idx_df['Technology'] = cat
    idx_df_sorted = idx_df.sort_values(['Technology', 'Year'])
    combined_sorted = combined.iloc[idx_df_sorted.index.values].copy()
    combined_sorted.index = pd.MultiIndex.from_frame(idx_df_sorted.astype({'Technology':'string'}))
    return combined_sorted

def _materials_first_columns(
    blocks: Iterable[pd.DataFrame],
    block_names=('Stock','NAS','Inflow','Outflow')
) -> pd.DataFrame:
    blocks = list(blocks)
    if len(blocks) != len(block_names):
        raise ValueError("blocks and block_names length mismatch")
    union_cols = blocks[0].columns
    for b in blocks[1:]:
        union_cols = union_cols.union(b.columns)
    aligned = [b.reindex(columns=union_cols) for b in blocks]
    pieces = {}
    for mat in union_cols:
        sub_cols = {}
        for bname, bdf in zip(block_names, aligned):
            if mat in bdf.columns:
                sub_cols[bname] = bdf[mat]
            else:
                sub_cols[bname] = pd.Series(index=bdf.index, dtype='float64')
        pieces[mat] = pd.concat(sub_cols, axis=1)
    out = pd.concat(pieces, axis=1)
    return out

def _triptych_side_by_side_materials_first(
    stock: pd.DataFrame,
    inflow: pd.DataFrame,
    nas: pd.DataFrame,
    outflow: pd.DataFrame,
    allow_single_technology=False,
) -> pd.DataFrame:
    stock = _ensure_multiindex(stock, "stock", allow_single_technology=allow_single_technology)
    inflow = _ensure_multiindex(inflow, "inflow", allow_single_technology=allow_single_technology)
    nas = _ensure_multiindex(nas, "nas", allow_single_technology=allow_single_technology)
    outflow = _ensure_multiindex(outflow, "outflow", allow_single_technology=allow_single_technology)
    return _materials_first_columns([stock, nas, inflow, outflow], ('Stock','NAS','Inflow','Outflow'))

def export_stock_flow_to_excel(
    StockCapacities: pd.DataFrame,
    InflowCapacities: pd.DataFrame,
    NASCapacities: pd.DataFrame,
    OutflowCapacities: pd.DataFrame,
    StockMaterials: pd.DataFrame,
    InflowMaterials: pd.DataFrame,
    NASMaterials: pd.DataFrame,
    OutflowMaterials: pd.DataFrame,
    StockMaterialsSystem: Optional[pd.DataFrame] = None,
    InflowMaterialsSystem: Optional[pd.DataFrame] = None,
    NASMaterialsSystem: Optional[pd.DataFrame] = None,
    OutflowMaterialsSystem: Optional[pd.DataFrame] = None,
    RecycledMaterials: pd.DataFrame = None,
    RecycledMaterialsSystem: Optional[pd.DataFrame] = None,
    path: str = "stock_flow_export.xlsx",
    system_label: str = "System",
):
    cap_sheet = _triptych_side_by_side_materials_first(
        _normalize_capacity_index(StockCapacities, "StockCapacities"),
        _normalize_capacity_index(InflowCapacities, "InflowCapacities"),
        _normalize_capacity_index(NASCapacities, "NASCapacities"),
        _normalize_capacity_index(OutflowCapacities, "OutflowCapacities"),
        allow_single_technology=False
    )

    pivot_tidy = _build_pivot_tidy_from_capacities(
        stock_df=StockCapacities,
        inflow_df=InflowCapacities,
        nas_df=NASCapacities,
        outflow_df=OutflowCapacities,
    )
    if pivot_tidy.empty:
        raise RuntimeError(
            "pivot_tidy is empty. Check that Stock/NAS/Inflow/Outflow capacity tables have Year as index "
            "and numeric values (or numeric-as-strings that can be coerced)."
        )

    stock_mat_with_sys = _add_system_as_technology(StockMaterials, StockMaterialsSystem, tech_label=system_label)
    inflow_mat_with_sys = _add_system_as_technology(InflowMaterials, InflowMaterialsSystem, tech_label=system_label)
    nas_mat_with_sys = _add_system_as_technology(NASMaterials, NASMaterialsSystem, tech_label=system_label)
    outflow_mat_with_sys = _add_system_as_technology(OutflowMaterials, OutflowMaterialsSystem, tech_label=system_label)

    mat_sheet = _triptych_side_by_side_materials_first(
        stock_mat_with_sys, inflow_mat_with_sys, nas_mat_with_sys, outflow_mat_with_sys
    )

    pivot_material = _build_pivot_material_from_material_flow(mat_sheet)

    if RecycledMaterials is None:
        raise ValueError("RecycledMaterials must be provided for the final sheet.")
    recycled_with_sys = _add_system_as_technology(RecycledMaterials, RecycledMaterialsSystem, tech_label=system_label)

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        cap_sheet.to_excel(writer, sheet_name="Installed Capacity")
        mat_sheet.to_excel(writer, sheet_name="Material flow")
        recycled_with_sys.to_excel(writer, sheet_name="Recycled Materials")
        pivot_tidy.to_excel(writer, sheet_name="Pivot Technology", index=False)
        pivot_material.to_excel(writer, sheet_name="Pivot Material", index=False)

        for sheet_name in ["Installed Capacity", "Material flow", "Recycled Materials", "Pivot Technology", "Pivot Material"]:
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes(2, 2)

    return path

def build_pivot_chart_with_slicers(xlsx_path: str):
    if win32 is None:
        raise RuntimeError("pywin32 / Excel not available in this environment.")
    excel = win32.gencache.EnsureDispatch("Excel.Application")
    excel.Visible = False
    try:
        try:
            from win32com.client import constants as C  # type: ignore
        except Exception:
            C = None
        def K(name):
            return getattr(C, name) if C is not None else XL[name]
        wb = excel.Workbooks.Open(os.path.abspath(xlsx_path))
        try:
            src_ws = wb.Worksheets("Pivot Material")
        except Exception:
            raise RuntimeError("Sheet 'Pivot Material' not found in workbook")
        try:
            view_ws = wb.Worksheets("Charts")
            view_ws.Cells.Clear()
        except Exception:
            view_ws = wb.Worksheets.Add()
            view_ws.Name = "Charts"
        src_range = src_ws.UsedRange
        pc = wb.PivotCaches().Create(SourceType=K("xlDatabase"), SourceData=src_range)
        pt = pc.CreatePivotTable(TableDestination=view_ws.Range("A3"), TableName="PM_Pivot")
        pf_year = pt.PivotFields("Year");       pf_year.Orientation = K("xlRowField")
        pf_flow = pt.PivotFields("Flow");       pf_flow.Orientation = K("xlColumnField")
        pf_val  = pt.AddDataField(pt.PivotFields("Value"), "Sum of Value", K("xlSum"))
        pf_tech = pt.PivotFields("Technology"); pf_tech.Orientation = K("xlPageField")
        pf_mat  = pt.PivotFields("Material");   pf_mat.Orientation = K("xlPageField")
        shp = view_ws.Shapes.AddChart2(240, K("xlAreaStacked"))
        chart = shp.Chart
        chart.SetSourceData(pt.TableRange2)
        shp.Left   = view_ws.Range("H2").Left
        shp.Top    = view_ws.Range("H2").Top
        shp.Width  = 1080
        shp.Height = 300
        sc = wb.SlicerCaches
        tech_cache = sc.Add2(pt, "Technology", "Technology")
        tech_cache.Slicers.Add(view_ws, Top=view_ws.Range("A1").Top, Left=view_ws.Range("A1").Left, Height=150, Width=200)
        mat_cache = sc.Add2(pt, "Material", "Material")
        mat_cache.Slicers.Add(view_ws, Top=view_ws.Range("C1").Top, Left=view_ws.Range("C1").Left, Height=150, Width=200)
        wb.Close(SaveChanges=True)
    finally:
        excel.Quit()
