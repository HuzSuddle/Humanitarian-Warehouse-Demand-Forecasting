
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Humanitarian Demand Forecasting",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

PREDICTION_FILE = BASE_DIR / "all_test_predictions.csv"
METRICS_FILE = BASE_DIR / "model_comparison_results.csv"

MODEL_OPTIONS = [
    "Naive",
    "Moving average",
    "Linear regression",
    "Decision Tree",
    "Random Forest",
    "Neural network"
]

MODEL_COLORS = {
    "naive": "#C5D9EC",
    "moving average": "#A3C6E8",
    "linear regression": "#75A9D8",
    "decision tree": "#508FC9",
    "random forest": "#286EAD",
    "neural network": "#174A78"
}

ALERT_STYLES = {
    "CRITICAL": {
        "background": "#FDE2E2",
        "border": "#E9A5A5",
        "text": "#A52828",
        "description": "Immediate attention required"
    },
    "HIGH": {
        "background": "#FCE8D5",
        "border": "#EFC49D",
        "text": "#A65316",
        "description": "High inventory risk"
    },
    "WARNING": {
        "background": "#FFF4CC",
        "border": "#EAD78B",
        "text": "#86640B",
        "description": "Monitor inventory levels"
    },
    "OK": {
        "background": "#DDF3E4",
        "border": "#A9D9B9",
        "text": "#267044",
        "description": "Sufficient projected stock"
    }
}


# =========================================================
# CUSTOM CSS - LIGHT MODE
# =========================================================

st.markdown("""
<style>

/* GENERAL PAGE */
.stApp {
    background-color: #F7F9FC !important;
}

/* Run Model button - solid blue box */
.stButton > button,
.stButton > button[kind="secondary"],
.stButton > button[kind="primary"] {
    background: #2878BE !important;
    background-color: #2878BE !important;
    color: #FFFFFF !important;
    border: 2px solid #2878BE !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}

/* Hover */
.stButton > button:hover {
    background: #1F639F !important;
    background-color: #1F639F !important;
    color: #FFFFFF !important;
    border-color: #1F639F !important;
}

[data-testid="stAppViewContainer"] {
    background-color: #F7F9FC !important;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
    max-width: 1450px;
}

/* HEADINGS */
h1, h2, h3, h4, h5, h6 {
    color: #244D70 !important;
}

/* GENERAL TEXT */
.stMarkdown p,
.stMarkdown li,
.stMarkdown label {
    color: #344F68 !important;
}

/* CAPTIONS */
.stCaption,
[data-testid="stCaptionContainer"] p {
    color: #647C92 !important;
}

/* HEADER */
.dashboard-header {
    background: linear-gradient(110deg, #DCEEFF, #EFF7FF);
    padding: 20px 25px;
    border-radius: 13px;
    border: 1px solid #D1E5F8;
    margin-bottom: 16px;
}

.dashboard-header h1 {
    color: #174A78 !important;
    font-size: 27px;
    margin: 0;
}

.dashboard-header p {
    color: #54718D !important;
    font-size: 13px;
    margin: 5px 0 0 0;
}

/* BUTTONS */
div.stButton > button {
    background-color: #2878BE !important;
    color: white !important;
    border: none !important;
    border-radius: 8px;
    font-weight: 600;
    min-height: 42px;
}

div.stButton > button:hover {
    background-color: #185F9E !important;
    color: white !important;
}

/* INFO BUTTONS */
div[data-testid="stPopover"] > button {
    color: #508FC9 !important;
    background-color: transparent !important;
    border: 1px solid #D1E5F8 !important;
    border-radius: 50% !important;
    min-width: 32px;
    min-height: 32px;
    padding: 0 7px !important;
    font-size: 13px;
}

div[data-testid="stPopover"] > button:hover {
    background-color: #EAF4FD !important;
    border-color: #75A9D8 !important;
}

/* POPOVER CONTENT */
div[data-testid="stPopoverBody"] {
    background-color: #FFFFFF !important;
}

/* METRIC CARDS */
div[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E1EAF3 !important;
    padding: 12px 15px;
    border-radius: 10px;
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p {
    color: #60758A !important;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] div {
    color: #174A78 !important;
}

/* SELECT BOXES */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p {
    color: #344F68 !important;
}

div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
}

div[data-baseweb="select"] *,
div[data-baseweb="input"] input {
    color: #263F55 !important;
}

/* ALERT CARDS */
.alert-card {
    padding: 15px 13px;
    border-radius: 11px;
    border: 1px solid;
    min-height: 105px;
    margin-bottom: 5px;
}

.alert-card .alert-label {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.alert-card .alert-number {
    font-size: 29px;
    font-weight: 750;
    line-height: 1.4;
    margin-top: 3px;
}

.alert-card .alert-description {
    font-size: 11px;
    margin-top: 2px;
}

/* EXPANDERS */
[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E1EAF3 !important;
    border-radius: 9px;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #244D70 !important;
}

/* ALERT MESSAGES */
[data-testid="stAlert"] {
    background-color: #FFFFFF !important;
}

[data-testid="stAlert"] p {
    color: #344F68 !important;
}

/* DATAFRAMES */
[data-testid="stDataFrame"] {
    background-color: #FFFFFF !important;
}

/* DIVIDERS */
hr {
    margin-top: 12px;
    margin-bottom: 12px;
    border-color: #DCE5EE !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def normalise(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def find_column(df, possible_names):
    lookup = {normalise(col): col for col in df.columns}

    for name in possible_names:
        if normalise(name) in lookup:
            return lookup[normalise(name)]

    return None


def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return None


def format_number(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.2f}"


# CLICKABLE INFO POPOVER
def show_info(text):
    with st.popover("ⓘ"):
        st.write(text)


def show_alert_card(level, count):
    style = ALERT_STYLES[level]

    card_html = f"""
    <div class="alert-card"
         style="
            background-color: {style['background']};
            border-color: {style['border']};
            color: {style['text']};
         ">
        <div class="alert-label">{level}</div>
        <div class="alert-number">{count}</div>
        <div class="alert-description">{style['description']}</div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)


def get_alert_level(value):
    value = str(value).upper().strip()

    if "CRITICAL" in value:
        return "CRITICAL"
    elif "HIGH" in value:
        return "HIGH"
    elif "WARNING" in value:
        return "WARNING"
    elif value == "OK" or value.startswith("OK "):
        return "OK"

    return "OTHER"


def get_model_colour(model_name):
    return MODEL_COLORS.get(normalise(model_name), "#7FAEDB")


# =========================================================
# LOAD DATA
# =========================================================

predictions = load_csv(PREDICTION_FILE)
metrics = load_csv(METRICS_FILE)

if predictions is None:
    st.error(
        "Could not find all_test_predictions.csv. "
        "Check your project folder."
    )
    st.stop()

if metrics is None:
    st.error(
        "Could not find model_comparison_results.csv. "
        "Check your project folder."
    )
    st.stop()

required_columns = ["demand_units"]

for column in required_columns:
    if column not in predictions.columns:
        st.error(
            f"Required column '{column}' is missing "
            "from the predictions CSV."
        )
        st.stop()


# LOAD SHORTAGE ALERT DATA
alerts = None

possible_alert_files = [
    OUTPUT_DIR / "test_alerts_all_models.csv",
    OUTPUT_DIR / "walkforward_alerts_all_models.csv"
]

for file in possible_alert_files:
    if file.exists():
        alerts = pd.read_csv(file)
        break


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="dashboard-header">
    <h1>Humanitarian Warehouse Demand Forecasting</h1>
    <p>Weekly demand prediction, model comparison and inventory shortage monitoring</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# DATASET OVERVIEW
# =========================================================

category_col = find_column(predictions, ["category"])
sku_col = find_column(predictions, ["sku"])
week_col = find_column(predictions, ["week_end"])

overview1, overview2, overview3 = st.columns(3)

with overview1:

    st.metric(
        "Prediction Records",
        f"{len(predictions):,}"
    )

    show_info(
        "Total number of prediction records available "
        "in the dataset. Each record represents an "
        "individual demand observation."
    )

with overview2:

    category_count = (
        predictions[category_col].nunique()
        if category_col else "N/A"
    )

    st.metric(
        "Supply Categories",
        category_count
    )

    show_info(
        "Number of different humanitarian supply categories, "
        "such as Food & Nutrition, Shelter & NFI, Medical "
        "and WASH."
    )

with overview3:

    sku_count = (
        predictions[sku_col].nunique()
        if sku_col else "N/A"
    )

    st.metric(
        "Items / SKUs",
        sku_count
    )

    show_info(
        "Number of unique inventory items (Stock Keeping Units) "
        "included in the dataset."
    )


st.divider()


# =========================================================
# FORECAST SETTINGS
# =========================================================

st.markdown("### Forecast Settings")

show_info(
    "Choose a forecasting model, supply category and "
    "individual inventory item. Click Run Model to apply "
    "your selections and view the results."
)

filter1, filter2, filter3, filter4 = st.columns(
    [1.2, 1.2, 1.2, 0.8]
)

with filter1:

    selected_model = st.selectbox(
        "Select Model",
        MODEL_OPTIONS,
        help=(
            "Choose a forecasting algorithm. Different models "
            "use different approaches to predict future demand."
        )
    )

with filter2:

    if category_col:

        categories = ["All Categories"] + sorted(
            predictions[category_col].dropna().unique().tolist()
        )

        selected_category = st.selectbox(
            "Supply Category",
            categories,
            help=(
                "Filter predictions and inventory alerts "
                "by a particular humanitarian supply category."
            )
        )

    else:
        selected_category = "All Categories"


with filter3:

    if sku_col:

        available_skus = predictions.copy()

        if selected_category != "All Categories" and category_col:
            available_skus = available_skus[
                available_skus[category_col] == selected_category
            ]

        sku_options = ["All SKUs"] + sorted(
            available_skus[sku_col].dropna().unique().tolist()
        )

        selected_sku = st.selectbox(
            "Item / SKU",
            sku_options,
            help=(
                "Select an individual inventory item to view "
                "its demand predictions and shortage alerts."
            )
        )

    else:
        selected_sku = "All SKUs"


with filter4:

    st.markdown(
        "<div style='height:27px'></div>",
        unsafe_allow_html=True
    )

    run_model = st.button(
        "▶ Run Model",
        use_container_width=True,
        help=(
            "Apply your selected model and filters to display "
            "forecasting results, performance metrics, graphs "
            "and inventory alerts."
        )
    )


# =========================================================
# SAVE SELECTIONS AFTER RUN BUTTON
# =========================================================

if run_model:

    st.session_state["run_model"] = True

    st.session_state["selected_model"] = selected_model
    st.session_state["selected_category"] = selected_category
    st.session_state["selected_sku"] = selected_sku


if "run_model" not in st.session_state:

    st.info(
        "Choose your model and filters, then click Run Model."
    )

    st.stop()


selected_model = st.session_state["selected_model"]
selected_category = st.session_state["selected_category"]
selected_sku = st.session_state["selected_sku"]


# =========================================================
# FILTER PREDICTION DATA
# =========================================================

filtered = predictions.copy()

if selected_category != "All Categories" and category_col:

    filtered = filtered[
        filtered[category_col] == selected_category
    ]


if selected_sku != "All SKUs" and sku_col:

    filtered = filtered[
        filtered[sku_col] == selected_sku
    ]


if selected_model not in filtered.columns:

    st.error(
        f"The prediction column '{selected_model}' was not found."
    )

    st.stop()


if filtered.empty:

    st.warning(
        "No prediction records were found for these filters."
    )

    st.stop()


filtered["demand_units"] = pd.to_numeric(
    filtered["demand_units"],
    errors="coerce"
)

filtered[selected_model] = pd.to_numeric(
    filtered[selected_model],
    errors="coerce"
)

filtered = filtered.dropna(
    subset=["demand_units", selected_model]
)


if filtered.empty:

    st.warning(
        "There are no valid prediction values for this selection."
    )

    st.stop()


# =========================================================
# MODEL PERFORMANCE METRICS
# =========================================================

metric_model_col = find_column(metrics, ["Model"])
mae_col = find_column(metrics, ["MAE"])
rmse_col = find_column(metrics, ["RMSE"])

model_metrics = None

if metric_model_col:

    matching_metrics = metrics[
        metrics[metric_model_col].apply(normalise)
        == normalise(selected_model)
    ]

    if not matching_metrics.empty:
        model_metrics = matching_metrics.iloc[0]


actual_total = filtered["demand_units"].sum()
predicted_total = filtered[selected_model].sum()

actual_values = filtered["demand_units"]
predicted_values = filtered[selected_model]


if model_metrics is not None and mae_col:

    mae_value = pd.to_numeric(
        pd.Series([model_metrics[mae_col]]),
        errors="coerce"
    ).iloc[0]

else:

    mae_value = (
        actual_values - predicted_values
    ).abs().mean()


if model_metrics is not None and rmse_col:

    rmse_value = pd.to_numeric(
        pd.Series([model_metrics[rmse_col]]),
        errors="coerce"
    ).iloc[0]

else:

    rmse_value = (
        ((actual_values - predicted_values) ** 2).mean()
    ) ** 0.5


# =========================================================
# KPI CARDS
# =========================================================

st.divider()

st.markdown(f"### Results: {selected_model}")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)


with kpi1:

    st.metric(
        "MAE",
        format_number(mae_value)
    )

    show_info(
        "Mean Absolute Error (MAE): Measures the average "
        "absolute difference between actual demand and "
        "predicted demand. Lower values indicate better "
        "prediction accuracy."
    )


with kpi2:

    st.metric(
        "RMSE",
        format_number(rmse_value)
    )

    show_info(
        "Root Mean Squared Error (RMSE): Measures prediction "
        "errors while giving greater weight to larger errors. "
        "Lower values indicate better prediction accuracy."
    )


with kpi3:

    st.metric(
        "Actual Demand",
        f"{actual_total:,.0f}"
    )

    show_info(
        "Total recorded demand across the selected items "
        "and weeks."
    )


with kpi4:

    st.metric(
        "Predicted Demand",
        f"{predicted_total:,.0f}"
    )

    show_info(
        "Total demand predicted by the selected forecasting "
        "model for the selected records."
    )


st.caption(
    "MAE and RMSE measure prediction error. Lower values "
    "indicate better accuracy. Demand totals reflect "
    "the selected filters."
)


# =========================================================
# GRAPHS
# =========================================================

st.divider()

chart_left, chart_right = st.columns([1.35, 1])


# =========================================================
# ACTUAL VS PREDICTED GRAPH
# =========================================================

with chart_left:

    st.markdown("### Actual vs Predicted Demand")

    show_info(
        "This line graph compares actual weekly demand "
        "with the demand predicted by the selected model. "
        "The closer the two lines are, the more accurately "
        "the model predicts demand. The horizontal axis "
        "shows the week and the vertical axis shows demand "
        "in units."
    )

    st.caption(
        "Compare actual weekly demand with the selected "
        "model's predictions."
    )

    if week_col:

        chart_data = filtered.copy()

        chart_data[week_col] = pd.to_datetime(
            chart_data[week_col],
            errors="coerce"
        )

        chart_data = chart_data.dropna(
            subset=[week_col]
        )

        if not chart_data.empty:

            chart_data = chart_data.groupby(
                week_col,
                as_index=False
            )[["demand_units", selected_model]].sum()

            chart_data = chart_data.rename(
                columns={
                    week_col: "Week",
                    "demand_units": "Actual Demand",
                    selected_model: "Predicted Demand"
                }
            )

            chart_data = chart_data.melt(
                id_vars="Week",
                value_vars=[
                    "Actual Demand",
                    "Predicted Demand"
                ],
                var_name="Demand Type",
                value_name="Units"
            )

            line_chart = px.line(
                chart_data,
                x="Week",
                y="Units",
                color="Demand Type",
                markers=True,
                color_discrete_map={
                    "Actual Demand": "#174A78",
                    "Predicted Demand": "#4F9BD5"
                }
            )

            line_chart.update_traces(
                line=dict(width=3),
                marker=dict(size=6)
            )

            line_chart.update_layout(
                height=330,
                margin=dict(
                    l=10,
                    r=10,
                    t=15,
                    b=10
                ),
                font=dict(
                    family="Arial",
                    color="#244D70",
                    size=12
                ),
                legend_title_text="",
                xaxis_title="Week",
                yaxis_title="Demand (units)",
                hovermode="x unified",
                plot_bgcolor="white",
                paper_bgcolor="white"
            )

            line_chart.update_xaxes(
                showgrid=False
            )

            line_chart.update_yaxes(
                gridcolor="#E8EEF4",
                zeroline=False
            )

            st.plotly_chart(
                line_chart,
                use_container_width=True
            )

        else:

            st.info(
                "No valid weekly dates were found "
                "in the selected data."
            )

    else:

        st.info(
            "The week_end column was not found."
        )


# =========================================================
# MODEL COMPARISON GRAPH
# =========================================================

with chart_right:

    st.markdown("### Model Comparison")

    show_info(
        "This horizontal bar chart compares the Mean "
        "Absolute Error (MAE) of all forecasting models. "
        "Each bar represents one model. Shorter bars "
        "indicate lower prediction errors and better "
        "accuracy based on MAE."
    )

    st.caption(
        "Compare MAE across the available models. Lower is better."
    )

    if metric_model_col and mae_col:

        comparison = metrics[
            [metric_model_col, mae_col]
        ].copy()

        comparison[mae_col] = pd.to_numeric(
            comparison[mae_col],
            errors="coerce"
        )

        comparison = comparison.dropna(
            subset=[mae_col]
        )

        comparison = comparison.sort_values(
            mae_col,
            ascending=True
        )

        comparison["Colour"] = comparison[
            metric_model_col
        ].apply(get_model_colour)

        bar_chart = px.bar(
            comparison,
            x=mae_col,
            y=metric_model_col,
            orientation="h",
            color="Colour",
            color_discrete_map="identity",
            text=mae_col
        )

        bar_chart.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
            cliponaxis=False
        )

        bar_chart.update_layout(
            height=330,
            margin=dict(
                l=10,
                r=45,
                t=15,
                b=10
            ),
            font=dict(
                family="Arial",
                color="#244D70",
                size=12
            ),
            showlegend=False,
            xaxis_title="Mean Absolute Error (MAE)",
            yaxis_title="",
            plot_bgcolor="white",
            paper_bgcolor="white",
            yaxis=dict(
                autorange="reversed"
            )
        )

        bar_chart.update_xaxes(
            gridcolor="#E8EEF4",
            zeroline=False
        )

        bar_chart.update_yaxes(
            showgrid=False
        )

        st.plotly_chart(
            bar_chart,
            use_container_width=True
        )

    else:

        st.info(
            "Could not identify the Model and MAE "
            "columns in the metrics file."
        )


# =========================================================
# INVENTORY SHORTAGE ALERTS
# =========================================================

st.divider()

st.markdown("### Inventory Shortage Alerts")

show_info("""
These alerts identify potential inventory shortage risks
based on forecasted demand.

CRITICAL: Immediate attention required.

HIGH: High inventory risk.

WARNING: Inventory levels should be monitored.

OK: Sufficient projected stock.

These are forecast-based indicators and do not represent
confirmed stockouts.
""")

st.caption(
    "Forecast-based inventory risk indicators. "
    "These do not represent confirmed stockouts."
)


if alerts is None:

    st.info(
        "No alert CSV was found. Add "
        "test_alerts_all_models.csv or "
        "walkforward_alerts_all_models.csv "
        "inside your outputs folder."
    )

else:

    alert_data = alerts.copy()

    alert_category_col = find_column(
        alert_data,
        ["category"]
    )

    alert_sku_col = find_column(
        alert_data,
        ["sku"]
    )

    alert_model_col = find_column(
        alert_data,
        ["model"]
    )

    severity_col = find_column(
        alert_data,
        [
            "alert_level",
            "alert",
            "severity",
            "risk_level",
            "risk"
        ]
    )


    # FILTER BY CATEGORY

    if selected_category != "All Categories" and alert_category_col:

        alert_data = alert_data[
            alert_data[alert_category_col] == selected_category
        ]


    # FILTER BY SKU

    if selected_sku != "All SKUs" and alert_sku_col:

        alert_data = alert_data[
            alert_data[alert_sku_col] == selected_sku
        ]


    # FILTER BY MODEL

    if alert_model_col:

        alert_data = alert_data[
            alert_data[alert_model_col].apply(normalise)
            == normalise(selected_model)
        ]


    if severity_col and not alert_data.empty:

        alert_data["Alert Category"] = (
            alert_data[severity_col].apply(get_alert_level)
        )

        critical_count = (
            alert_data["Alert Category"] == "CRITICAL"
        ).sum()

        high_count = (
            alert_data["Alert Category"] == "HIGH"
        ).sum()

        warning_count = (
            alert_data["Alert Category"] == "WARNING"
        ).sum()

        ok_count = (
            alert_data["Alert Category"] == "OK"
        ).sum()


        # FOUR ALERT CARDS

        alert1, alert2, alert3, alert4 = st.columns(4)

        with alert1:
            show_alert_card(
                "CRITICAL",
                int(critical_count)
            )

        with alert2:
            show_alert_card(
                "HIGH",
                int(high_count)
            )

        with alert3:
            show_alert_card(
                "WARNING",
                int(warning_count)
            )

        with alert4:
            show_alert_card(
                "OK",
                int(ok_count)
            )


        # ALERT DETAILS TABLE

        with st.expander(
            "View shortage alert details",
            expanded=False
        ):

            show_info(
                "This table displays the individual shortage "
                "alert records for the selected model, category "
                "and SKU. You can download the records as a CSV."
            )

            st.dataframe(
                alert_data.drop(
                    columns=["Alert Category"],
                    errors="ignore"
                ),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Alert Results",
                data=alert_data.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="shortage_alert_results.csv",
                mime="text/csv"
            )


    elif alert_data.empty:

        st.info(
            "No alert records match the selected filters."
        )


    else:

        st.warning(
            "An alert file was found, but its severity "
            "column could not be identified."
        )

        st.dataframe(
            alert_data,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# DETAILED PREDICTION RESULTS
# =========================================================

st.divider()

st.markdown("### Detailed Prediction Results")

show_info(
    "View individual prediction records, including actual "
    "demand, predicted demand, prediction error and absolute "
    "error. Prediction Error shows the difference between "
    "predicted and actual demand. Absolute Error shows the "
    "size of that difference regardless of direction. "
    "You can also download the results as a CSV file."
)


with st.expander(
    "View detailed prediction results",
    expanded=False
):

    result_table = filtered.copy()

    result_table["Prediction Error"] = (
        result_table[selected_model]
        - result_table["demand_units"]
    )

    result_table["Absolute Error"] = (
        result_table["Prediction Error"].abs()
    )

    st.dataframe(
        result_table,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Prediction Results",
        data=result_table.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="demand_predictions.csv",
        mime="text/csv"
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<hr>
<p style="
    text-align:center;
    color:#8293A5;
    font-size:11px;
">
    Humanitarian Warehouse Demand Forecasting |
    CSCI218 Foundations of AI
</p>
""", unsafe_allow_html=True)
  
