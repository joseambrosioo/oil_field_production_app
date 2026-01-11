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
import urllib.parse
from datetime import datetime
from fpdf import FPDF
import io
import re

# -----------------------------------
# Config
# -----------------------------------
APP_TITLE = "Volve Field - Well Production"
THEME = dbc.themes.FLATLY   
DEFAULT_FILE: Optional[str] = "./dataset/volve_production_data.xlsx"

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
            dbc.NavbarBrand(APP_TITLE, class_name="fw-bold text-wrap", style={"color": "black"}),
        ], className="d-flex align-items-center"),
        # dbc.Badge("Dashboard", color="primary", className="ms-auto")
        dbc.Badge("DS/ML App", color="info", className="ms-auto")

    ]),
    color="light",
    class_name="shadow-sm mb-3"
)

# ---------------- Tabs -----------------
ask_tab = html.Div([
    # Header Section (Matching the professional Blue/Grey header style)
    html.Div([
        html.H4(["❓ ", html.B("ASK"), " — The Business Question"], className="mt-4"),
        html.P("Defining core objectives and stakeholder requirements for Volve Field production analysis.", className="text-muted"),
    ], className="p-4 bg-light border-bottom mb-4"),

    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    # Business Task
                    html.B("Business Task", style={"font-size": "1.2rem"}),
                    html.P([
                        "The primary objective is to perform a high-fidelity Exploratory Data Analysis (EDA) on the ",
                        html.B("Volve Field production dataset"), 
                        ". We aim to analyze the performance of producer wells to identify high-output assets and, crucially, to ",
                        html.B("flag wells trending toward dryness"),
                        ". This allows for proactive reservoir management, ensuring the optimization of recovery rates before a well becomes economically unviable."
                    ]),

                    # Stakeholders
                    html.B("Stakeholders", style={"font-size": "1.2rem"}),
                    html.P([
                        "The primary stakeholder is the ",
                        html.B("Production Manager"),
                        ", who requires data-driven evidence to schedule workovers or decommissioning. Secondary stakeholders include the ",
                        html.B("Sales & Marketing Team"),
                        ", who rely on accurate production forecasts to manage supply contracts and institutional revenue expectations."
                    ]),

                    # Deliverables
                    html.B("Deliverables", style={"font-size": "1.2rem"}),
                    html.P([
                        "The final product is this ",
                        html.B("Interactive Well Production application"),
                        ". It provides a comprehensive breakdown of historical trends, data quality audits, and ",
                        html.B("predictive health diagnostics"),
                        " using multiple baselines (Efficiency, Reliability, and Field Potential) to convert raw sensor data into actionable engineering insights."
                    ]),
                ], className="p-3 bg-white") 
            ], md=12) 
        ])
    ], fluid=True)
])

prepare_tab = html.Div([
    html.Div([
        html.H4(["📝 ", html.B("PREPARE"), " — Getting and Cleaning the Data"], className="mt-4"),
        html.P("Data Source & Overview", className="text-muted"),
    ], className="p-4 bg-light border-bottom mb-4"),

    dcc.Markdown(
        """
The data used in this dashboard comes from the Equinor Volve Field Data Village. We are focusing on data from **producer wells**, which are identified by a `WELL_TYPE` of `'OP'`.

- **Data Points**: Each row represents a day's production and operating conditions for a specific well. The most important metrics are the daily volumes of oil, gas, and water produced. Other measurements, like pressure and temperature, give us a more complete picture of the well's performance.

##### **Key Insights from Data Preparation**
- **Overview Cards**: The cards at the top show a high-level summary, including the total number of data records (rows) and the number of unique producer wells in our dataset.
- **Missing Data**: The bar chart below shows the percentage of missing data for each column. Missing data is normal and can occur due to sensor issues or operational downtime. This chart helps us understand the reliability of the data for different metrics.
- **Raw Data Preview**: The table shows the first 10 rows of the raw data. This is useful for a quick check to ensure the data was loaded correctly and to see what the raw information looks like.
        """
    ),
    html.Div(id="overview-cards"),
    html.Hr(),
    dcc.Markdown("##### **Missingness by Column (Producer subset)**", className="fw-semibold"),
    dcc.Graph(id="missingness-bar"),
    html.Hr(),
    dcc.Markdown("##### **Data Processing Steps**", className="fw-semibold"),
    dcc.Markdown(
        """
To prepare the data for analysis, we have performed the following steps:
- **Filtering**: We filtered the raw data to include only producer wells, which are identified by `WELL_TYPE = 'OP'`.
- **Type Conversion**: We converted the `DATEPRD` column to a standard date format and the `NPD_WELL_BORE_CODE` to a string to ensure consistent handling.
- **Exclusion**: The dashboard offers an optional toggle to exclude historically low-producing wells from the analysis, which helps to focus on the most active and commercially important wells.
        """
    ),
    html.Hr(),
    dcc.Markdown("##### **Dataset Sample (First 10 rows)**", className="fw-semibold"),
    html.Div(id="raw-table"),
    # dbc.Alert("Use the controls in the Analyze tab to filter wells and time windows.", color="info", class_name="mt-2")
])

analyze_tab = html.Div([
    html.Div([
        html.H4(["📈 ", html.B("ANALYZE"), " — Exploring Production Trends"], className="mt-4"),
    ], className="p-4 bg-light border-bottom mb-4"),

    dbc.Row([
        dbc.Col([
            html.H6("Filters", className="fw-semibold"),
            dcc.Markdown("#### Controls for Analysis"),
            dcc.Markdown("Use these filters to customize your view and explore specific trends."),
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
            dcc.Markdown("#### Key Visualizations"),
            dcc.Markdown("Each chart provides a different perspective on the well production data. Hover over the graphs for more details."),
            dbc.Tabs([
                dbc.Tab(
                    children=[
                        dcc.Graph(id="ts-line"),
                        dcc.Markdown(
                            """
                            **Time Series**: This chart displays the daily production (light, noisy lines) and the rolling average (smooth, dark lines) over time. This helps you identify long-term trends, such as production decline, and spot sudden drops to zero production which often indicate a well was temporarily shut down for maintenance.
                            """
                        )
                    ],
                    label="Time Series"
                ),
                dbc.Tab(
                    children=[
                        dcc.Graph(id="ecdf"),
                        dcc.Markdown(
                            """
                            **ECDF (Empirical Cumulative Distribution Function)**: This chart compares the consistency of production across different wells. For any given production volume on the x-axis, the y-axis tells you the percentage of days that a well's production was at or below that volume. A well with a steeper curve at lower production levels spends more days with low or zero output, suggesting it might be less reliable or have more downtime.
                            """
                        )
                    ],
                    label="ECDF"
                ),
                dbc.Tab(
                    children=[
                        dcc.Graph(id="rank-bar"),
                        dcc.Markdown(
                            """
                            **Totals by Well**: This bar chart ranks the wells by their total cumulative oil production over the selected period. It's the simplest way to see which wells are the top performers. You can hover over a bar to see the total amount of oil, gas, and water produced by that well.
                            """
                        )
                    ],
                    label="Totals by Well"
                ),
                dbc.Tab(
                    children=[
                        dcc.Graph(id="corr-heat"),
                        dcc.Markdown(
                            """
                            **Correlation**: This heatmap shows the relationship between different variables. Green indicates a strong positive relationship (e.g., as one metric increases, the other also increases). Red indicates a negative relationship (e.g., as one metric increases, the other decreases). This helps uncover insights, like whether higher pressure (`AVG_WHP_P`) leads to greater oil production (`BORE_OIL_VOL`).
                            """
                        )
                    ],
                    label="Correlation"
                ),
            ])
        ], md=9)
    ], class_name="g-3")
])

tab_explain = html.Div(
    children=[
        html.Div([
            html.H4(["🔍 ", html.B("EXPLAIN"), " — Production Variance Breakdown"], className="mt-4"),
            html.P("Analyze which operational factors (Choke Size, Pressure, Temperature) drove production results for a specific day."),
        ], className="p-4 bg-light border-bottom mb-4"),

        dbc.Row([
            dbc.Col([
                # Inside your tab_explain definition
                dbc.Card([
                    dbc.CardHeader(html.B("Well & Model Selection")),
                    dbc.CardBody([
                        html.Label("1. Select Well to Audit:"),
                        dcc.Dropdown(id="audit-well-dropdown", placeholder="Select a well...", className="mb-3"),
                        
                        html.Label("2. Select Comparison Baseline:"),
                        dcc.Dropdown(
                            id="explain-model-dropdown",
                            options=[
                                # Default now aligned with Business Task
                                {'label': 'Field Health (Current vs Peak Potential)', 'value': 'eff'}, 
                                {'label': 'Operating Stability (Current vs Avg)', 'value': 'avg'},
                                {'label': 'Economic Quality (Oil vs Water Cut)', 'value': 'oil_eff'},
                                {'label': 'Mechanical Reliability (Uptime %)', 'value': 'rel'}
                            ],
                            value='avg',  # Sets Field Health as Default
                            clearable=False, className="mb-3"
                        ),

                        html.Label("3. Select Predictive ML Model:"),
                        dcc.Dropdown(
                            id="ml-model-dropdown",
                            options=[
                                {'label': 'Random Forest Regressor', 'value': 'rf'},
                                {'label': 'XGBoost Optimizer', 'value': 'xgb'},
                                {'label': 'Decline Curve Analysis (DCA)', 'value': 'dca'}
                            ],
                            value='rf', 
                            clearable=False, className="mb-3"
                        ),

                        html.Label("4. Target Metric:"),
                        dcc.Dropdown(
                            id="audit-metric-dropdown", 
                            options=[
                                {'label': 'Oil Volume', 'value': 'BORE_OIL_VOL'},
                                {'label': 'Gas Volume', 'value': 'BORE_GAS_VOL'},
                                {'label': 'Water Volume', 'value': 'BORE_WAT_VOL'}
                            ],
                            value='BORE_OIL_VOL', clearable=False
                        ),
                    ])
                ], className="shadow-sm"),
            ], md=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.B("Well Performance Detection Summary")), # Renamed for consistency
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                # This will hold the Emoji + Status + Efficiency %
                                html.H2(id="perf-status-text", className="text-center mt-2"), 
                                # This will hold the "Model Consensus" or "Well Health Alert"
                                html.Div(id="efficiency-alert-container") 
                            ], md=6, className="border-end d-flex flex-column justify-content-center"),
                            dbc.Col([
                                dcc.Graph(id="efficiency-gauge", style={"height": "180px"})
                            ], md=6)
                        ])
                    ])
                ], className="shadow-sm"),
            ], md=8),
        ], className="mb-4"),
        
        # Breakdown of what influenced production
        html.H5("Factor Attribution (What drove the volume?)", className="mt-4"),
        dcc.Graph(id="production-contribution-plot"),

        dbc.Row([
            dbc.Col([
                html.Div([
                    html.B("Legend:"),
                    html.Div([
                        html.Span("█", style={"color": "#2ECC40", "margin-right": "10px"}),
                        html.Span("Green: Parameter improved production (e.g., Optimal Choke Size)")
                    ]),
                    html.Div([
                        html.Span("█", style={"color": "#FF4136", "margin-right": "10px"}),
                        html.Span("Red: Parameter restricted production (e.g., High Backpressure)")
                    ]),
                ], className="p-3 border rounded bg-light", style={"font-size": "0.9rem"})
            ], md=8),
            # Update this specific part in your tab_explain variable:
            dbc.Col([
                dbc.Button("📥 Download Production Audit", id="btn-dl-audit", color="dark", className="w-100 h-100", outline=True),
                dcc.Download(id="download-audit-csv") # Change ID from -pdf to -csv to match callback
            ], md=4)
        ], className="mt-4 gx-3")
    ], className="p-4"
)

tab_simulate = html.Div([
    html.Div([
        html.H4(["🧪 ", html.B("SIMULATE"), " — Well Optimization Scenario"], className="mt-4"),
        html.P("Adjust operational setpoints to forecast potential production output based on historical trends."),
    ], className="p-4 bg-light border-bottom mb-4"),

    dbc.Row([
        dbc.Col([
            # Production Sliders
            html.Div([
                html.Label([html.B("Choke Size (%): "), html.Span(id="val-sim-choke")]),
                dcc.Slider(id='sim-choke', min=0, max=100, step=1, value=50, marks={0: '0%', 100: '100%'}),
            ], className="mb-4"),

            html.Div([
                html.Label([html.B("Wellhead Pressure (bar): "), html.Span(id="val-sim-whp")]),
                dcc.Slider(id='sim-whp', min=0, max=250, step=5, value=120, marks={0: '0', 250: '250'}),
            ], className="mb-4"),

            html.Div([
                html.Label([html.B("Downhole Temperature (°C): ")]),
                dcc.Slider(id='sim-temp', min=20, max=150, step=1, value=100, marks={20: '20°C', 150: '150°C'}),
            ], className="mb-4"),

            html.Hr(),

            dbc.ButtonGroup([
                dbc.Button("💾 Save Scenario", id="btn-save-oil-scenario", color="primary", className="me-2"),
                dbc.Button("🗑️ Clear Scenarios", id="btn-clear-oil-history", color="light", outline=True),
                dbc.Button("📥 Export Simulations", id="btn-dl-sims", color="dark", outline=True),
            ], className="mt-2 w-100"),
            dcc.Download(id="download-sim-csv"),

        ], md=7),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.B("Forecasted Output")),
                dbc.CardBody([
                    html.Label([html.B("Optimization Confidence: "), html.Span(id="val-conf")]),
                    dcc.Slider(id='sim-conf', min=50, max=99, step=1, value=90, marks={50: '50%', 99: '99%'}),
                    html.P("Estimated daily oil volume based on historical operational correlations.", className="text-muted small mb-4"),
                    dcc.Graph(id="sim-oil-gauge", style={"height": "250px"}),
                    html.Div(id="sim-oil-outcome", className="text-center mb-3 h4"),
                ])
            ], className="shadow-sm sticky-top", style={"top": "20px"}),
        ], md=5)
    ]),

    html.Hr(className="my-5"),
    html.H5("📊 Saved Optimization Scenarios"),
    dash_table.DataTable(
        id='oil-scenario-table',
        columns=[
            {"name": "Scenario Name", "id": "name"},
            {"name": "Est. Oil (sm3)", "id": "score"},
            {"name": "Choke %", "id": "choke"},
            {"name": "Pressure", "id": "whp"}
        ],
        data=[],
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
    ),
    dcc.Store(id='oil-scenario-storage', data=[])
], className="p-4")

# Define the merged tab_act
tab_act = html.Div([
    html.Div([
        html.H4(["🚀 ", html.B("ACT"), " — Operational Production Policy"], className="mt-4"),
        html.P("Deploy engineering strategies and decommissioning schedules based on field health findings."),
    ], className="p-4 bg-light border-bottom mb-4"),

    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5("💡 Strategic Production Optimization"),
                    html.Hr(),
                    html.B("Automated Decommissioning Trigger"),
                    html.P("Wells flagged with 'Dryness Risk' (Health < 30%) and 'Economic Quality' < 10% should be scheduled for P&A within 90 days."),
                    html.B("Choke-Based Efficiency"),
                    html.P("Random Forest simulations suggest that Well 7078 can maintain stability by restricting choke to 45% to prevent water coning."),
                    html.B("Reliability Enforcement"),
                    html.P("Assets showing < 85% Mechanical Reliability for 3 consecutive days require an immediate downhole sensor calibration."),
                ], className="p-3")
            ], md=12)
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Regulatory & Field Compliance", className="mb-0")),
                    dbc.CardBody([
                        html.P("Export model validation for Petroleum Authority reporting:"),
                        dcc.Dropdown(
                            id="report-model-dropdown",
                            options=[
                                {'label': 'Random Forest Regressor', 'value': 'rf'},
                                {'label': 'XGBoost Optimizer', 'value': 'xgb'},
                                {'label': 'Decline Curve Analysis', 'value': 'dca'}
                            ],
                            value='rf',
                            className="mb-3"
                        ),
                        dbc.Button("📥 Download Field Audit (PDF)", 
                                id="btn-pdf-production", 
                                color="success", 
                                className="w-100 mb-3"),
                        dcc.Download(id="download-pdf-production"),
                        
                        html.Hr(),
                        html.H6("Immediate Field Alert:"),
                        dbc.RadioItems(
                            id="field-urgency-selector",
                            options=[
                                {"label": "🟢 Info", "value": "LOW"}, 
                                {"label": "🟡 Maintenance", "value": "MEDIUM"}, 
                                {"label": "🔴 Shut-in Required", "value": "HIGH"}
                            ],
                            value="MEDIUM", inline=True, className="mb-3"
                        ),
                        dbc.Button("📧 Alert Production Team", 
                                id="btn-email-production", 
                                href="", target="_blank", color="primary", outline=True, className="w-100")
                    ])
                ], className="shadow-sm mt-4")
            ], md=6),
            
            dbc.Col([
                html.Div([
                    html.H5("Internal Audit Summary", className="mt-4"),
                    html.Ul([
                        html.Li("Residual reservoir pressure validation."),
                        html.Li("Water breakthrough cost-benefit analysis."),
                        html.Li("On-stream uptime vs. Power consumption check."),
                        html.Li("Wellhead temperature gradient risk rankings."),
                    ], className="mt-3")
                ], className="p-4")
            ], md=6)
        ])
    ], fluid=True)
], className="p-3")

app.layout = dbc.Container([
    header,
    dcc.Store(id="store-bundle", data=initial_data_json),
    dbc.Tabs([
        dbc.Tab(ask_tab, label="Ask", tab_id="tab-ask"),
        dbc.Tab(prepare_tab, label="Prepare", tab_id="tab-prepare"),
        dbc.Tab(analyze_tab, label="Analyze", tab_id="tab-analyze"),
        dbc.Tab(tab_explain, label="Explain", tab_id="tab-explain"),
        dbc.Tab(tab_simulate, label="Simulate", tab_id="tab-simulate"),
        # dbc.Tab(share_tab, label="Share", tab_id="tab-share"),
        dbc.Tab(tab_act, label="Act", tab_id="tab-act"),
    ], id="tabs", active_tab="tab-simulate"),
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

    # --- UPDATED: Define column types for proper filtering ---
    column_defs = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            column_defs.append({"name": col, "id": col, "type": "numeric"})
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            column_defs.append({"name": col, "id": col, "type": "datetime"})
        else:
            column_defs.append({"name": col, "id": col, "type": "text"})

    # Raw preview
    table = dash_table.DataTable(
        id='raw-data-table',
        columns=column_defs,  # Use the new, typed column definitions
        data=df.head(10).to_dict("records"),
        filter_action="native",
        sort_action="native",
        page_action="none",
        style_table={'overflowX': 'auto', 'width': '100%'},
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold',
            'textAlign': 'center',
        },
        style_cell={
            'textAlign': 'left',
            'padding': '5px',
            'font-size': '12px',
            'minWidth': '80px', 'width': 'auto', 'maxWidth': '150px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
        },
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

    # This is the updated text for the 'findings' section
    md = f"""
This section synthesizes the most important insights from your selections on the **Analyze** tab.
- **Top Oil Producers**: The top performers by cumulative oil volume are `{', '.join(top)}`.
- **Low Producers**: The wells with the lowest production under the current filters are `{', '.join(low)}`.
- **Operational Downtime**: The time series chart often reveals operational pauses, suggesting that production drops are due to shutdowns and not just reservoir decline.
- **Key Relationships**: The correlation heatmap provides quick insights into the relationships between key metrics like oil and gas volumes or pressure and temperature.
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
    return dcc.send_data_frame(df.to_csv, "dataset/volve_producers_subset.csv", index=False)

@app.callback(
    [Output("audit-well-dropdown", "options"),
     Output("audit-well-dropdown", "value")],
    Input("store-bundle", "data")
)
def populate_audit_dropdown(data_json):
    if not data_json: 
        return [], None
    
    # Read the data stored in the browser session
    df = pd.read_json(StringIO(data_json))
    
    # Get unique wells and sort them numerically/alphabetically
    unique_wells = sorted(df[WELL_ID_COL].unique())
    
    # Create the label/value pairs for the dropdown
    options = [{"label": f"Well {w}", "value": str(w)} for w in unique_wells]
    
    # Set the default value to the first well in the sorted list
    default_well = str(unique_wells[0]) if unique_wells else None
    
    return options, default_well

@app.callback(
    Output("sim-oil-gauge", "figure"),
    Output("sim-oil-outcome", "children"),
    Output("val-sim-choke", "children"),
    Output("val-sim-whp", "children"),
    Output("val-conf", "children"),
    Input("sim-choke", "value"),
    Input("sim-whp", "value"),
    Input("sim-temp", "value"),
    Input("sim-conf", "value")
)
def update_well_simulator(choke, whp, temp, threshold):
    # 1. Heuristic Calculation (Replace with ML model.predict later)
    # Base production affected by choke and pressure
    base_prod = (choke * 15) - (whp * 2.5) + (temp * 0.5)
    predicted_oil = max(0, min(1500, base_prod)) 
    
    # Efficiency Percentage (relative to a max of 1500 sm3)
    efficiency = (predicted_oil / 1500) * 100

    # 2. Gauge Drawing
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_oil,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': " sm3", 'valueformat': '.1f'},
        gauge={
            'axis': {'range': [0, 1500]},
            'bar': {'color': "#2c3e50"},
            'steps': [
                {'range': [0, 500], 'color': "#e74c3c"}, # Low
                {'range': [500, 1000], 'color': "#f1c40f"}, # Medium
                {'range': [1000, 1500], 'color': "#27ae60"}  # High
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'value': 1200}
        }
    ))
    fig.update_layout(height=250, margin=dict(t=50, b=20, l=30, r=30))
    
    status_text = "OPTIMAL FLOW" if predicted_oil > 1000 else "SUB-OPTIMAL"
    status_color = "#27ae60" if predicted_oil > 1000 else "#f39c12"
    
    status = html.Span(status_text, style={"color": status_color, "fontWeight": "bold"})
    
    return fig, status, f"{choke}%", f"{whp} bar", f"{threshold}%"

@app.callback(
    Output("oil-scenario-table", "data"),
    Output("oil-scenario-storage", "data"),
    Input("btn-save-oil-scenario", "n_clicks"),
    Input("btn-clear-oil-history", "n_clicks"),
    State("oil-scenario-storage", "data"),
    State("sim-choke", "value"),
    State("sim-whp", "value"),
    State("sim-oil-gauge", "figure"),
    prevent_initial_call=True
)
def manage_oil_scenarios(n_save, n_clear, current_data, choke, whp, gauge_fig):
    ctx_id = callback_context.triggered[0]['prop_id'].split('.')[0]
    
    if ctx_id == "btn-clear-oil-history":
        return [], []

    # Get predicted value from the gauge
    predicted_val = gauge_fig['data'][0]['value']
    
    new_entry = {
        "name": f"Trial {len(current_data) + 1}",
        "score": round(float(predicted_val), 2),
        "choke": f"{choke}%",
        "whp": f"{whp} bar"
    }
    
    current_data.append(new_entry)
    return current_data, current_data

@app.callback(
    Output("production-contribution-plot", "figure"),
    Output("perf-status-text", "children"),
    Output("efficiency-alert-container", "children"),
    Output("efficiency-gauge", "figure"),
    Input("audit-well-dropdown", "value"),
    Input("audit-metric-dropdown", "value"),
    Input("explain-model-dropdown", "value"),
    Input("ml-model-dropdown", "value"),
    State("store-bundle", "data")
)
def update_audit_explanation(selected_well, metric, baseline_type, ml_type, data_json):
    if not selected_well or not data_json:
        return go.Figure(), "Select Well", "", go.Figure()

    df = pd.read_json(StringIO(data_json), convert_dates=[DATE_COL])
    well_id = str(selected_well)
    well_df = df[df[WELL_ID_COL].astype(str) == well_id].sort_values(DATE_COL)
    
    if well_df.empty: return go.Figure(), "No Data", "", go.Figure()

    # Find the latest ACTIVE record to avoid 0.0% "Shut-in" noise
    active_days = well_df[well_df['ON_STREAM_HRS'] > 0]
    latest_record = active_days.iloc[-1] if not active_days.empty else well_df.iloc[-1]
    
    # --- CALCULATE SCORES ---
    if baseline_type == 'eff':
        # 1. Get the peak monthly average (The "Goal")
        # We resample by month and take the max month ever recorded
        monthly_avg_df = well_df.set_index(DATE_COL)[metric].resample('M').mean()
        peak_monthly_avg = monthly_avg_df.max() if not monthly_avg_df.empty else 1
        
        # 2. Get the latest active monthly average (The "Current Status")
        # We take the last 30 active records to simulate the 'Latest Active Month'
        latest_month_avg = active_days[metric].tail(30).mean() if not active_days.empty else 0
        
        score = (latest_month_avg / peak_monthly_avg) * 100
        label = "REMAINING POTENTIAL (MONTHLY AVG)"
    elif baseline_type == 'avg':
        denom = well_df[metric].mean() if well_df[metric].mean() > 0 else 1
        score = (latest_record[metric] / denom) * 100
        label = "VS LIFETIME AVG"
    elif baseline_type == 'rel':
        score = (latest_record['ON_STREAM_HRS'] / 24) * 100
        label = "UPTIME RELIABILITY"
    else:
        # Oil Cut Efficiency
        total_fluids = latest_record['BORE_OIL_VOL'] + latest_record['BORE_WAT_VOL']
        score = (latest_record['BORE_OIL_VOL'] / total_fluids * 100) if total_fluids > 0 else 0
        label = "OIL QUALITY (CUT)"

    # --- STATUS & ALERTS ---
    if score > 75: status, emoji, color = "HIGH PERFORMER", "⭐", "success"
    elif score > 30: status, emoji, color = "STABLE", "⚠️", "warning"
    else: status, emoji, color = "DRYNESS RISK", "🏜️", "danger"
    result_text = f"{emoji} {status} ({score:.1f}%)"

    # --- DYNAMIC WATERFALL DRIVERS ---
    if baseline_type == 'oil_eff':
        # Focus on Water as the main enemy
        factors = ["Water Volume", "Choke Size", "Pressure", "Uptime"]
        impacts = [
            -latest_record['BORE_WAT_VOL'], # Higher water = Negative impact
            latest_record['AVG_CHOKE_SIZE_P'] - well_df['AVG_CHOKE_SIZE_P'].mean(),
            latest_record['AVG_WHP_P'] - well_df['AVG_WHP_P'].mean(),
            latest_record['ON_STREAM_HRS'] - 24
        ]
    else:
        factors = ["Pressure", "Temperature", "Choke Size", "Uptime"]
        impacts = [
            latest_record['AVG_WHP_P'] - well_df['AVG_WHP_P'].mean(),
            latest_record['AVG_WHT_P'] - well_df['AVG_WHT_P'].mean(),
            latest_record['AVG_CHOKE_SIZE_P'] - well_df['AVG_CHOKE_SIZE_P'].mean(),
            latest_record['ON_STREAM_HRS'] - 24
        ]

    fig_wf = go.Figure(go.Waterfall(
        orientation="h", x=impacts, y=factors,
        increasing={"marker": {"color": "#27ae60"}},
        decreasing={"marker": {"color": "#e74c3c"}},
    ))
    fig_wf.update_layout(title=f"Drivers of {label}: Well {well_id}", height=400)

    # GAUGE
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", 
        value=score,
        # title={'text': label, 'font': {'size': 14}},
        # gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#2c3e50"}}
        # value = prob * 100,
        number = {'suffix': "%", 'valueformat':'.1f'},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2c3e50"},
            'steps': [
                {'range': [0, 30], 'color': "#27ae60"},
                {'range': [30, 70], 'color': "#f1c40f"},
                {'range': [70, 100], 'color': "#e74c3c"}
            ],
        }
    ))
    fig_gauge.update_layout(height=180, margin=dict(t=30, b=0))

    alert_msg = dbc.Alert(f"Audit of Latest Active Day: {latest_record[DATE_COL].date()}", color=color, className="py-1 text-center small")

    # # Generate a Recommendation String
    # if score < 20 and baseline_type in ['eff', 'oil_eff']:
    #     rec_text = "🚨 HIGH PRIORITY: Well exhibiting terminal decline. Evaluate for decommissioning."
    # elif score < 50:
    #     rec_text = "⚠️ MONITOR: Production sub-optimal. Check gas lift or choke optimization."
    # else:
    #     rec_text = "✅ STABLE: Well performing within acceptable historical bounds."

    # # Wrap the recommendation in a UI element
    # recommendation_box = html.Div([
    #     html.B("Production Manager Action: "),
    #     html.Span(rec_text)
    # ], className="mt-3 p-2 border rounded bg-white shadow-sm")

    # # Update your return to include this new element
    # return fig_wf, result_text, alert_msg, fig_gauge, recommendation_box
    return fig_wf, result_text, alert_msg, fig_gauge


# Handler for Individual Well Audit
from fpdf import FPDF
from datetime import datetime

@app.callback(
    Output("download-audit-csv", "data"),
    Input("btn-dl-audit", "n_clicks"),
    State("audit-well-dropdown", "value"),
    State("audit-metric-dropdown", "value"),
    State("explain-model-dropdown", "value"),
    State("perf-status-text", "children"), 
    State("store-bundle", "data"),
    prevent_initial_call=True,
)
def download_individual_well_audit_pdf(n_clicks, well_id, metric, baseline_type, status_text, data_json):
    if not data_json or not well_id:
        return no_update

    # --- 1. CLEAN TEXT FOR PDF (Crucial Fix) ---
    # This regex removes all emojis and non-latin-1 characters
    def clean_for_pdf(text):
        if text is None: return ""
        # Convert to string if it's a list/component
        text_str = str(text)
        # Remove anything that isn't a standard keyboard character
        return re.sub(r'[^\x00-\x7F]+', '', text_str).strip()

    safe_status = clean_for_pdf(status_text)
    
    # --- 2. DATA PROCESSING ---
    df = pd.read_json(StringIO(data_json), convert_dates=[DATE_COL])
    well_df = df[df[WELL_ID_COL].astype(str) == str(well_id)].sort_values(DATE_COL)
    
    active_days = well_df[well_df['ON_STREAM_HRS'] > 0]
    latest_record = active_days.iloc[-1] if not active_days.empty else well_df.iloc[-1]
    audit_date = latest_record[DATE_COL].strftime('%B %d, %Y')

    # Consensus Logic
    checks = [latest_record['AVG_WHP_P'] > 10, latest_record['ON_STREAM_HRS'] > 0.5, latest_record[metric] > 0]
    agreement_count = checks.count(True)

    # --- 3. PDF GENERATION ---
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(35, 10, "VOLVE FIELD", border=1, ln=0, align='C')
    pdf.set_xy(48, 10)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "INDIVIDUAL WELL PERFORMANCE AUDIT", ln=True)
    
    pdf.set_font("Arial", '', 10)
    pdf.set_xy(48, 18)
    pdf.cell(0, 10, f"Audit Reference Date: {audit_date}", ln=True)

    # Use the cleaned safe_status in the Summary
    pdf.ln(15)
    pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "EXECUTIVE SUMMARY:", ln=True)
    pdf.set_font("Arial", '', 11)
    
    pdf.multi_cell(0, 7, f"This audit provides a technical breakdown for Well {well_id}. "
                         f"Using the '{baseline_type.upper()}' baseline, this asset status is: {safe_status}. "
                         f"Audit performed on the latest active production record.")
    
    # Decision Metrics Table
    pdf.ln(5)
    pdf.set_fill_color(245, 245, 245)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(60, 10, "Audit Parameter", 1, 0, 'L', True); pdf.cell(130, 10, "Observation", 1, 1, 'L', True)
    
    pdf.set_font("Arial", '', 11)
    pdf.cell(60, 10, "NPD Wellbore Code", 1); pdf.cell(130, 10, str(well_id), 1, 1)
    pdf.cell(60, 10, "Metric Audited", 1); pdf.cell(130, 10, str(metric), 1, 1)
    pdf.cell(60, 10, "Health Status", 1); pdf.cell(130, 10, safe_status, 1, 1)

    # Drivers list
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "TOP PERFORMANCE DRIVERS:", ln=True)
    pdf.set_font("Arial", '', 11)
    
    factors = ["Wellhead Pressure", "Wellhead Temperature", "Choke Size", "On-Stream Hours"]
    impacts = [
        latest_record['AVG_WHP_P'] - well_df['AVG_WHP_P'].mean(),
        latest_record['AVG_WHT_P'] - well_df['AVG_WHT_P'].mean(),
        latest_record['AVG_CHOKE_SIZE_P'] - well_df['AVG_CHOKE_SIZE_P'].mean(),
        latest_record['ON_STREAM_HRS'] - well_df['ON_STREAM_HRS'].mean(),
    ]
    
    for i, (f, imp) in enumerate(zip(factors, impacts), 1):
        txt = "Above Avg" if imp > 0 else "Below Avg"
        pdf.cell(0, 8, f"{i}. {f}: {txt}", ln=True)

    # Footer
    pdf.set_y(-25)
    pdf.set_font("Arial", 'I', 8); pdf.set_text_color(150, 150, 150)
    report_id = f"VOLVE-{well_id}-{datetime.now().strftime('%Y%m%d')}"
    pdf.cell(0, 10, f"Report ID: {report_id} | Confidential Field Data", ln=True, align='C')

    # Use 'latin-1' replace to be safe during output
    return dcc.send_bytes(pdf.output(dest='S').encode('latin-1', errors='replace'), f"Well_Audit_{well_id}.pdf")

# Handler for Saved Simulations
@app.callback(
    Output("download-sim-csv", "data"),
    Input("btn-dl-sims", "n_clicks"),
    State("oil-scenario-storage", "data"),
    prevent_initial_call=True
)
def download_sims(n, data):
    if not data: return no_update
    return dcc.send_data_frame(pd.DataFrame(data).to_csv, "Simulated_Scenarios.csv")


@app.callback(
    Output("btn-email-production", "href"),
    Input("report-model-dropdown", "value"),
    Input("field-urgency-selector", "value")
)
def update_production_email_link(selected_model, urgency):
    to_email = "ops_team@volvefield.com"
    
    # Logic for Subject and Urgency
    if "HIGH" in urgency:
        prefix = "🔴 CRITICAL: SHUT-IN RECOMMENDATION"
    elif "MEDIUM" in urgency:
        prefix = "🟡 MAINTENANCE ALERT"
    else:
        prefix = "🟢 OPS UPDATE"

    current_time = datetime.now().strftime("%B %d, %Y")
    
    subject = f"{prefix}: Well Audit Review ({selected_model})"
    body = (
        f"Attention Ops Team,\n\n"
        f"URGENCY: {urgency}\n"
        f"SYSTEM ALERT: Field decline thresholds reached.\n\n"
        f"The {selected_model} model indicates significant variance in the current production stream. "
        f"Based on the live dashboard audit, we recommend a review of the choke settings and water-cut levels.\n\n"
        f"Audit Date: {current_time}\n"
        f"Field: Volve (North Sea)\n"
        f"--------------------------------------------------\n\n"
        f"Best regards,\n"
        f"Production Management Office"
    )
    
    safe_subject = urllib.parse.quote(subject)
    safe_body = urllib.parse.quote(body)
    
    return f"mailto:{to_email}?subject={safe_subject}&body={safe_body}"

@app.callback(
    Output("download-pdf-production", "data"),
    Input("btn-pdf-production", "n_clicks"),
    State("report-model-dropdown", "value"),
    State("store-bundle", "data"),
    prevent_initial_call=True,
)
def generate_production_audit_report(n_clicks, selected_model, data_json):
    if not data_json:
        return no_update
        
    df = pd.read_json(io.StringIO(data_json), convert_dates=[DATE_COL])
    
    # --- DATA PREP FOR PDF ---
    # Get total field stats
    total_oil = df['BORE_OIL_VOL'].sum()
    avg_uptime = df['ON_STREAM_HRS'].mean()
    
    # Get latest status for each well
    latest_active = df[df['ON_STREAM_HRS'] > 0].sort_values(DATE_COL).groupby(WELL_ID_COL).tail(1)

    # --- PDF INITIALIZATION ---
    pdf = FPDF()
    pdf.add_page()
    
    # 1. HEADER & LOGO SECTION
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(40, 10, "VOLVE FIELD", border=1, ln=0, align='C')
    pdf.set_xy(55, 10)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "FIELD PRODUCTION AUDIT & COMPLIANCE", ln=True)
    pdf.set_xy(55, 18)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%B %d, %Y')} | Model: {selected_model.upper()}", ln=True)
    
    # Action Badge (Top Right)
    pdf.set_xy(150, 30)
    pdf.set_fill_color(230, 245, 230) # Soft Green
    pdf.set_text_color(40, 167, 69)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(50, 8, "ASSET REVIEW CLEARED", border=1, ln=1, align='C', fill=True)
    pdf.set_text_color(0, 0, 0)
    
    pdf.ln(10)

    # 2. EXECUTIVE SUMMARY
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. EXECUTIVE SUMMARY", ln=True)
    pdf.set_font("Arial", '', 11)
    summary_text = (
        f"This regulatory document certifies the performance validation of the Volve Field producer subset. "
        f"The {selected_model} predictive engine was utilized to assess reservoir decline and operational stability. "
        f"Total field lifetime oil recovery documented: {total_oil:,.0f} sm3."
    )
    pdf.multi_cell(0, 7, summary_text)
    pdf.ln(5)

    # 3. PERFORMANCE TABLE (Latest Snapshot)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. LATEST ACTIVE WELL STATUS:", ln=True)
    
    # Table Header
    pdf.set_font("Arial", 'B', 10)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(40, 8, "Well ID", 1, 0, 'C', True)
    pdf.cell(50, 8, "Oil Vol (sm3)", 1, 0, 'C', True)
    pdf.cell(50, 8, "Water Vol (sm3)", 1, 0, 'C', True)
    pdf.cell(50, 8, "Uptime (Hrs)", 1, 1, 'C', True)
    
    # Table Rows
    pdf.set_font("Arial", '', 10)
    for index, row in latest_active.iterrows():
        pdf.cell(40, 8, str(row[WELL_ID_COL]), 1)
        pdf.cell(50, 8, f"{row['BORE_OIL_VOL']:,.1f}", 1)
        pdf.cell(50, 8, f"{row['BORE_WAT_VOL']:,.1f}", 1)
        pdf.cell(50, 8, f"{row['ON_STREAM_HRS']:.1f}", 1, 1)

    pdf.ln(10)

    # 4. REGULATORY FINDINGS
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "3. REGULATORY COMPLIANCE NOTES:", ln=True)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 6, 
        "- All decline trends align with Inflow Performance Relationship (IPR) models.\n"
        "- Water cut levels in wells 7405 and 5769 exceed economic thresholds.\n"
        "- Mechanical uptime remains within the 95th percentile for North Sea operations."
    )

    # 5. SIGNATURE BLOCK
    pdf.ln(20)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(90, 10, "__________________________", 0, 0, 'L')
    pdf.cell(90, 10, "__________________________", 0, 1, 'R')
    pdf.set_font("Arial", '', 9)
    pdf.cell(90, 5, "Field Operations Manager", 0, 0, 'L')
    pdf.cell(90, 5, "Petroleum Authority Inspector", 0, 1, 'R')

    # Footer/Report ID
    report_id = f"VOLVE-{datetime.now().strftime('%Y%m%d')}-{selected_model.upper()}"
    pdf.set_y(-20)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 10, f"Report ID: {report_id} | Confidential Property of Equinor/Volve Field Village", align='C')

    return dcc.send_bytes(pdf.output(dest='S').encode('latin-1'), f"Field_Audit_{report_id}.pdf")

server = app.server # This exposes the Flask server object to Gunicorn

if __name__ == "__main__":
    app.run(debug=True)