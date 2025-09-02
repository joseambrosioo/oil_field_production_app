import io
import json
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative

from dash import Dash, html, dcc, Input, Output, State, callback_context, dash_table, no_update
import dash_bootstrap_components as dbc
import base64
from io import StringIO

# -----------------------------------
# Config
# -----------------------------------
APP_TITLE = "Volve Field — Well Production EDA Dashboard"
THEME = dbc.themes.FLATLY
DEFAULT_FILE: Optional[str] = "./volve_production_data.xlsx"

NUMERIC_COLS_CORE = [
    "ON_STREAM_HRS",
    "AVG_DOWNHOLE_PRESSURE",
    "AVG_DOWNHOLE_TEMPERATURE",
    "AVG_DP_TUBING",
    "AVG_ANNULUS_PRESS",
    "AVG_CHOKE_SIZE_P",
    "AVG_WHP_P",
    "AVG_WHT_P",
    "DP_CHOKE_SIZE",
    "BORE_OIL_VOL",
    "BORE_GAS_VOL",
    "BORE_WAT_VOL",
    "BORE_WI_VOL",
]
WELL_ID_COL = "NPD_WELL_BORE_CODE"
WELL_TYPE_COL = "WELL_TYPE"
DATE_COL = "DATEPRD"

LOW_PRODUCER_WELLS = {"7405", "7289", "5769"}

# -----------------------------------
# Helpers
# -----------------------------------
@dataclass
class DataBundle:
    df_raw: pd.DataFrame
    df_op: pd.DataFrame  # producer-only


def load_excel(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df


def prepare_bundle(df: pd.DataFrame) -> DataBundle:
    df = df.copy()
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df_op = df[df[WELL_TYPE_COL].astype(str).str.upper().eq("OP")].copy()
    if WELL_ID_COL in df_op.columns:
        df_op[WELL_ID_COL] = df_op[WELL_ID_COL].astype(str)
    return DataBundle(df_raw=df, df_op=df_op)

# Load data on app startup
if DEFAULT_FILE:
    initial_df = load_excel(DEFAULT_FILE)
    initial_bundle = prepare_bundle(initial_df)
    initial_data_json = initial_bundle.df_op.to_json(date_format="iso", orient="records")
else:
    initial_data_json = None


# -----------------------------------
# App
# -----------------------------------
app = Dash(__name__, external_stylesheets=[THEME], title=APP_TITLE, suppress_callback_exceptions=True)
server = app.server

header = dbc.Navbar(
    dbc.Container([
        html.Div([
            html.Span("⛽", className="me-2"),
            dbc.NavbarBrand(APP_TITLE, class_name="fw-bold", style={"color": "black"}),
        ], className="d-flex align-items-center"),
        dbc.Badge("Interactive EDA", color="primary", className="ms-auto")
    ]),
    color="light",
    class_name="shadow-sm mb-3"
)

# Remove file_controls entirely and replace with a simple message and dcc.Store
data_info = dbc.Card(
    dbc.CardBody([
        html.H6("Data Source", className="fw-semibold mb-2"),
        html.Div(f"Using default file: {DEFAULT_FILE}", className="small text-muted"),
    ]), class_name="mb-3"
)

# ---------------- Tabs -----------------
ask_tab = dcc.Markdown(
    """
### ASK — Business Task & Stakeholders
**Business Task**: Analyze production data of **producer wells** in the Volve field to compare wells, highlight high performers, and flag wells trending toward dryness.

**Stakeholders**: Primary — Production Manager. Secondary — Sales & Marketing.

**Deliverables**: Interactive report with summary, data prep notes, key findings, and recommendations.
    """,
    className="p-2"
)

prepare_tab = html.Div([
    html.H6("Data Source & Overview", className="fw-semibold"),
    dcc.Markdown(
        """
- Source: Equinor Volve Field Data Village (Well-Production Excel).  
- Columns of interest include: `DATEPRD`, `NPD_WELL_BORE_CODE`, `WELL_TYPE`, `BORE_OIL_VOL`, `BORE_GAS_VOL`, `BORE_WAT_VOL`, and operating/pressure/temperature metrics.
- We will focus on rows with `WELL_TYPE == 'OP'` (producer wells).
        """
    ),
    html.Div(id="overview-cards"),
    html.Hr(),
    html.H6("Missingness by Column (Producer subset)", className="fw-semibold"),
    dcc.Graph(id="missingness-bar"),
    html.Hr(),
    html.H6("Raw Preview (first 10 rows)", className="fw-semibold"),
    html.Div(id="raw-table"),
])

process_tab = html.Div([
    html.H6("Processing Notes", className="fw-semibold"),
    dcc.Markdown(
        """
- Filter to producer wells (`WELL_TYPE = 'OP'`).  
- Normalize types: cast `DATEPRD` to datetime and `NPD_WELL_BORE_CODE` to string.  
- Optional: drop descriptive facility/name columns not used in analysis.  
- Provide toggles to exclude historically low-producing wells (e.g., 7405, 7289, 5769).
        """
    ),
    dbc.Alert("Use the controls in the Analyze tab to filter wells and time windows.", color="info", class_name="mt-2")
])

analyze_tab = html.Div([
    dbc.Row([
        dbc.Col([
            html.H6("Filters", className="fw-semibold"),
            dbc.Checklist(
                id="exclude-low",
                options=[{"label": "Exclude historically low producers (7405, 7289, 5769)", "value": "exclude"}],
                value=["exclude"],
                switch=True,
                class_name="mb-2"
            ),
            html.Label("Select wells"),
            dcc.Dropdown(id="well-select", multi=True, placeholder="All producers"),
            html.Label("Date range", className="mt-2"),
            dcc.DatePickerRange(id="date-range"),
            html.Label("Metric", className="mt-2"),
            dcc.Dropdown(
                id="y-metric",
                options=[
                    {"label": "Oil volume", "value": "BORE_OIL_VOL"},
                    {"label": "Gas volume", "value": "BORE_GAS_VOL"},
                    {"label": "Water volume", "value": "BORE_WAT_VOL"},
                ], value="BORE_OIL_VOL", clearable=False
            ),
            html.Label("Rolling average (days)", className="mt-2"),
            dcc.Slider(id="roll-window", min=1, max=30, value=7, step=1,
                       marks={1: "1", 7: "7", 14: "14", 30: "30"}),
            html.Div(id="selection-summary", className="small text-muted mt-2"),
        ], md=3),
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(dcc.Graph(id="ts-line"), label="Time Series"),
                dbc.Tab(dcc.Graph(id="ecdf"), label="ECDF"),
                dbc.Tab(dcc.Graph(id="rank-bar"), label="Totals by Well"),
                dbc.Tab(dcc.Graph(id="corr-heat"), label="Correlation"),
            ])
        ], md=9)
    ], class_name="g-3")
])

share_tab = html.Div([
    html.H6("Key Findings (Auto-updated)", className="fw-semibold"),
    html.Div(id="findings"),
    html.Hr(),
    html.H6("Download Processed Data", className="fw-semibold"),
    html.Div([
        dbc.Button("Download Producer Subset (CSV)", id="btn-dl", color="primary"),
        dcc.Download(id="dl-csv")
    ])
])

act_tab = dcc.Markdown(
    """
### ACT — Recommendations
- **Monitor decline**: All significant wells show typical late-life decline; plan end-of-life optimization and lift strategies as needed.
- **Economics**: Historically low producers (e.g., 7405, 7289, 5769) exhibit minimal oil with rising water — candidates for decommissioning or workover only if justified by nearby infrastructure or enhanced recovery pilots.
- **Ops cadence**: Periodic zeros in production align with reduced on‑stream hours; ensure shutdowns are tracked to separate operational from reservoir effects.
- **Reporting**: Export producer-only dataset and share interactive dashboards with stakeholders for weekly reviews.
    """,
    className="p-2"
)

app.layout = dbc.Container([
    header,
    dcc.Store(id="store-bundle", data=initial_data_json),
    data_info,
    dbc.Tabs([
        dbc.Tab(ask_tab, label="Ask"),
        dbc.Tab(prepare_tab, label="Prepare"),
        dbc.Tab(process_tab, label="Process"),
        dbc.Tab(analyze_tab, label="Analyze"),
        dbc.Tab(share_tab, label="Share"),
        dbc.Tab(act_tab, label="Act"),
    ], id="tabs"),
], fluid=True)


# ---------------- Callbacks -----------------
@app.callback(
    Output("overview-cards", "children"),
    Output("missingness-bar", "figure"),
    Output("raw-table", "children"),
    Input("store-bundle", "data"),
)
def prep_overview(data_json):
    if not data_json:
        return html.Div("Default file not found."), go.Figure(), html.Div()
    df = pd.read_json(StringIO(data_json), convert_dates=[DATE_COL])

    # Overview cards
    wells = sorted(df[WELL_ID_COL].unique())
    cards = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("Producer rows", className="text-muted small"),
            html.H4(f"{len(df):,}")
        ]), class_name="shadow-sm"), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("Producer wells", className="text-muted small"),
            html.H4(f"{len(wells)}"),
            html.Div(", ".join([str(w) for w in wells]), className="small")
        ]), class_name="shadow-sm"), md=9),
    ], class_name="g-2")

    # Missingness bar
    miss_pct = df[NUMERIC_COLS_CORE].isna().mean().sort_values(ascending=False) * 100
    fig_miss = px.bar(miss_pct, labels={"value": "% Missing", "index": "Column"})
    fig_miss.update_layout(margin=dict(l=10, r=10, t=10, b=10))

    # Raw preview
    table = dash_table.DataTable(
        data=df.head(10).to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        style_table={"overflowX": "auto"},
        page_size=10,
    )

    return cards, fig_miss, table


@app.callback(
    Output("well-select", "options"),
    Output("date-range", "min_date_allowed"),
    Output("date-range", "max_date_allowed"),
    Output("date-range", "start_date"),
    Output("date-range", "end_date"),
    Input("store-bundle", "data"),
)
def populate_filters(data_json):
    if not data_json:
        return [], None, None, None, None
    df = pd.read_json(StringIO(data_json), convert_dates=[DATE_COL])
    wells = sorted(df[WELL_ID_COL].unique().tolist())
    options = [{"label": w, "value": w} for w in wells]
    dmin, dmax = df[DATE_COL].min(), df[DATE_COL].max()
    return options, dmin, dmax, dmin, dmax


@app.callback(
    Output("selection-summary", "children"),
    Output("ts-line", "figure"),
    Output("ecdf", "figure"),
    Output("rank-bar", "figure"),
    Output("corr-heat", "figure"),
    Input("store-bundle", "data"),
    Input("exclude-low", "value"),
    Input("well-select", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("y-metric", "value"),
    Input("roll-window", "value"),
)
def analytics(data_json, exclude_low, wells_choice, dstart, dend, ycol, win):
    if not data_json:
        empty = go.Figure()
        return "", empty, empty, empty, empty
    df = pd.read_json(StringIO(data_json), convert_dates=[DATE_COL])

    if exclude_low and "exclude" in exclude_low:
        df = df[~df[WELL_ID_COL].astype(str).isin(list(LOW_PRODUCER_WELLS))]

    if wells_choice:
        df = df[df[WELL_ID_COL].isin(wells_choice)]

    if dstart:
        df = df[df[DATE_COL] >= pd.to_datetime(dstart)]
    if dend:
        df = df[df[DATE_COL] <= pd.to_datetime(dend)]

    wells = ", ".join([str(well) for well in sorted(df[WELL_ID_COL].unique())]) if not df.empty else "—"
    summary = f"Rows: {len(df):,} | Wells: {wells}"

    ts = df.sort_values(DATE_COL).copy()
    ts["value"] = ts[ycol]
    ts["roll"] = ts.groupby(WELL_ID_COL)["value"].transform(lambda s: s.rolling(win, min_periods=1).mean())

    fig_ts = go.Figure()
    for w in sorted(ts[WELL_ID_COL].unique()):
        sub = ts[ts[WELL_ID_COL] == w]
        fig_ts.add_trace(go.Scatter(x=sub[DATE_COL], y=sub["value"], name=f"{w} (daily)", mode="lines", opacity=0.35))
        fig_ts.add_trace(go.Scatter(x=sub[DATE_COL], y=sub["roll"], name=f"{w} (rolling)", mode="lines"))
    fig_ts.update_layout(yaxis_title=ycol, xaxis_title="Date", legend_title="Wells", margin=dict(l=10, r=10, t=10, b=10))

    fig_ecdf = go.Figure()
    for w in sorted(df[WELL_ID_COL].unique()):
        x = df.loc[df[WELL_ID_COL] == w, ycol].dropna().sort_values().values
        if x.size == 0:
            continue
        y = np.arange(1, x.size + 1) / x.size
        fig_ecdf.add_trace(go.Scatter(x=x, y=y, mode="lines", name=str(w)))
    fig_ecdf.update_layout(xaxis_title=ycol, yaxis_title="Cumulative probability", margin=dict(l=10, r=10, t=10, b=10))

    totals = df.groupby(WELL_ID_COL)[["BORE_OIL_VOL", "BORE_GAS_VOL", "BORE_WAT_VOL"]].sum().sort_values("BORE_OIL_VOL", ascending=False)
    fig_rank = px.bar(totals.reset_index(), x=WELL_ID_COL, y="BORE_OIL_VOL", hover_data=["BORE_GAS_VOL", "BORE_WAT_VOL"], labels={"BORE_OIL_VOL": "Total Oil"})
    fig_rank.update_layout(margin=dict(l=10, r=10, t=10, b=10))

    corr_df = df[[c for c in NUMERIC_COLS_CORE if c in df.columns]].corr()
    fig_corr = px.imshow(corr_df, text_auto=False, aspect="auto", color_continuous_scale="RdYlGn")
    fig_corr.update_layout(margin=dict(l=10, r=10, t=10, b=10))

    return summary, fig_ts, fig_ecdf, fig_rank, fig_corr


@app.callback(
    Output("findings", "children"),
    Input("store-bundle", "data"),
    Input("exclude-low", "value"),
    Input("y-metric", "value"),
)
def findings_text(data_json, exclude_low, ycol):
    if not data_json:
        return dcc.Markdown("No data loaded. Please check the default file path.")
    df = pd.read_json(StringIO(data_json), convert_dates=[DATE_COL])
    if exclude_low and "exclude" in exclude_low:
        df = df[~df[WELL_ID_COL].astype(str).isin(list(LOW_PRODUCER_WELLS))]
    
    totals = df.groupby(WELL_ID_COL)[["BORE_OIL_VOL", "BORE_GAS_VOL", "BORE_WAT_VOL"]].sum().sort_values("BORE_OIL_VOL", ascending=False)
    
    if totals.empty:
        return dcc.Markdown("No data in current filter.")
    
    top_wells = totals.head(2).index.tolist()
    top = [str(well) for well in top_wells]
    
    all_wells = set(df[WELL_ID_COL].unique())
    low_wells = sorted(list(all_wells - set(top_wells)))
    low = [str(well) for well in low_wells]

    md = f"""
- **Top oil producers**: `{', '.join(top)}` by cumulative oil volume.
- **Low producers**: `{', '.join(low)}` under current filters.
- **Oil vs Gas**: Correlation often positive; confirm on **Correlation** tab.
- **Operational downtime**: Zero-production days apparent in Time Series — aligns with reduced `ON_STREAM_HRS`.
    """
    return dcc.Markdown(md)


@app.callback(
    Output("dl-csv", "data"),
    Input("btn-dl", "n_clicks"),
    State("store-bundle", "data"),
    prevent_initial_call=True,
)
def download_csv(n, data_json):
    if not data_json:
        return no_update
    df = pd.read_json(StringIO(data_json), convert_dates=[DATE_COL])
    return dcc.send_data_frame(df.to_csv, "volve_producers_subset.csv", index=False)


if __name__ == "__main__":
    app.run(debug=True)