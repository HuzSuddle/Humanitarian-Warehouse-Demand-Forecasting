
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Humanitarian Demand Forecasting",
    page_icon=None,
    layout="wide"
)


# =========================================================
# FILE PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def find_file(filename):
    """Look for a CSV in the project folder or outputs folder."""
    possible_paths = [
        BASE_DIR / filename,
        OUTPUT_DIR / filename
    ]

    for path in possible_paths:
        if path.exists():
            return path

    return None


PREDICTION_FILE = find_file("all_test_predictions.csv")
METRICS_FILE = find_file("model_comparison_results.csv")

ALERT_FILES = [
    find_file("test_alerts_all_models.csv"),
    find_file("walkforward_alerts_all_models.csv")
]

ALERT_FILES = [file for file in ALERT_FILES if file is not None]


# =========================================================
# MODEL SETTINGS
# =========================================================

MODEL_OPTIONS = [
    "Naive",
    "Moving average",
    "Linear regression",
    "Decision Tree",
    "Random Forest",
    "Neural network"
]

MODEL_COLOURS = {
    "Naive": "#A6A6A6",
    "Moving average": "#8CB6D9",
    "Linear regression": "#2878B5",
    "Decision Tree": "#5398C8",
    "Random Forest": "#164B7A",
    "Neural network": "#A8C9E8"
}


# =========================================================
# CUSTOM DESIGN
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #F8FAFC;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.dashboard-header {
    background-color: #E8F3FC;
    padding: 25px 28px;
    border-radius: 12px;
    margin-bottom: 20px;
}

.dashboard-title {
    color: #174B75;
    font-size: 27px;
    font-weight: 750;
    margin-bottom: 5px;
}

.dashboard-subtitle {
    color: #526B80;
    font-size: 14px;
}

.section-heading {
    color: #174B75;
    font-size: 19px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 10px;
}

div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #E2E8F0;
    padding: 15px;
    border-radius: 10px;
}

div.stButton > button {
    background-color: #2878B5;
    color: white;
    border: none;
    border-radius: 7px;
    padding: 8px 22px;
    font-weight: 600;
}

div.stButton > button:hover {
    background-color: #164B7A;
    color: white;
}

.alert-card {
    padding: 13px 15px;
    border-radius: 9px;
    text-align: left;
    min-height: 83px;
}

.alert-name {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.4px;
}

.alert-count {
    font-size: 27px;
    font-weight: 750;
    margin-top: 5px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def load_csv(path):
    """Load a CSV safely."""
    if path is None:
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as error:
        st.error(f"Could not read {path.name}: {error}")
        return pd.DataFrame()


def normalise(column):
    return str(column).strip().lower().replace("_", " ").replace("-", " ")


def find_column(data, possible_names):
    """Find a column using alternative possible names."""
    if data.empty:
        return None

    for column in data.columns:
        if normalise(column) in possible_names:
            return column

    return None


def format_number(value):
    if pd.isna(value):
        return "N/A"

    return f"{value:,.2f}"


def section_title(title):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True
    )


def style_alerts(data, severity_column):
    """Apply colours to severity cells in the alert table."""

    def colour_severity(value):
        value = str(value).upper()

        if "CRITICAL" in value:
            return "background-color: #F8D7DA; color: #842029;"

        if "HIGH" in value:
            return "background-color: #FCE1CC; color: #984B13;"

        if "WARNING" in value:
            return "background-color: #FFF0BD; color: #806000;"

        if value == "OK":
            return "background-color: #D5EDDC; color: #23643B;"

        return ""

    return data.style.map(
        colour_severity,
        subset=[severity_column]
    )


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def get_data():
    predictions = load_csv(PREDICTION_FILE)
    metrics = load_csv(METRICS_FILE)

    alerts = pd.DataFrame()

    for alert_file in ALERT_FILES:
        current_alerts = load_csv(alert_file)

        if not current_alerts.empty:
            alerts = pd.concat(
                [alerts, current_alerts],
                ignore_index=True
            )

    return predictions, metrics, alerts


predictions, metrics, alerts = get_data()


# =========================================================
# CHECK DATA
# =========================================================

if predictions.empty:
    st.error(
        "The prediction file could not be found or is empty. "
        "Please check that all_test_predictions.csv is in your "
        "project folder or outputs folder."
    )
    st.stop()


# Identify important columns

category_col = find_column(predictions, {"category"})
sku_col = find_column(predictions, {"sku"})
date_col = find_column(predictions, {"week end", "date", "week"})
actual_col = find_column(predictions, {"demand units", "actual", "actual demand"})
item_col = find_column(predictions, {"item name", "item"})


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="dashboard-header">
    <div class="dashboard-title">
        Humanitarian Warehouse Demand Forecasting
    </div>
    <div class="dashboard-subtitle">
        Weekly demand predictions and inventory shortage monitoring
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.title("Dashboard Filters")

if category_col:
    categories = sorted(
        predictions[category_col].dropna().astype(str).unique()
    )

    selected_category = st.sidebar.selectbox(
        "Supply Category",
        ["All Categories"] + categories
    )
else:
    selected_category = "All Categories"


if sku_col:
    skus = sorted(
        predictions[sku_col].dropna().astype(str).unique()
    )

    selected_sku = st.sidebar.selectbox(
        "SKU",
        ["All SKUs"] + skus
    )
else:
    selected_sku = "All SKUs"


selected_model = st.sidebar.selectbox(
    "Forecasting Model",
    MODEL_OPTIONS,
    index=MODEL_OPTIONS.index("Random Forest")
)


# =========================================================
# FILTER PREDICTIONS
# =========================================================

filtered_data = predictions.copy()

if category_col and selected_category != "All Categories":
    filtered_data = filtered_data[
        filtered_data[category_col].astype(str) == selected_category
    ]

if sku_col and selected_sku != "All SKUs":
    filtered_data = filtered_data[
        filtered_data[sku_col].astype(str) == selected_sku
    ]


if filtered_data.empty:
    st.warning("No prediction data matches the selected filters.")
    st.stop()


# =========================================================
# RUN MODEL
# =========================================================

section_title("Forecast Overview")

run_model = st.button(
    "Run Model",
    type="primary"
)

if "model_has_run" not in st.session_state:
    st.session_state.model_has_run = False

if run_model:
    st.session_state.model_has_run = True
    st.session_state.selected_model = selected_model

if "selected_model" not in st.session_state:
    st.session_state.selected_model = selected_model

if st.session_state.selected_model != selected_model:
    st.session_state.selected_model = selected_model


if st.session_state.model_has_run:
    st.success(f"Showing results for {selected_model}.")
else:
    st.info(
        "Choose a forecasting model and click Run Model "
        "to display its results."
    )


# =========================================================
# KPI METRICS
# =========================================================

prediction_col = find_column(
    filtered_data,
    {normalise(selected_model)}
)

actual_values = (
    pd.to_numeric(filtered_data[actual_col], errors="coerce")
    if actual_col else pd.Series(dtype=float)
)

predicted_values = (
    pd.to_numeric(filtered_data[prediction_col], errors="coerce")
    if prediction_col else pd.Series(dtype=float)
)

mae_value = None
rmse_value = None

if not metrics.empty:
    metric_model_col = find_column(metrics, {"model"})
    mae_col = find_column(metrics, {"mae"})
    rmse_col = find_column(metrics, {"rmse"})

    if metric_model_col:
        selected_metrics = metrics[
            metrics[metric_model_col].astype(str).str.lower()
            == selected_model.lower()
        ]

        if not selected_metrics.empty:
            if mae_col:
                mae_value = pd.to_numeric(
                    selected_metrics.iloc[0][mae_col],
                    errors="coerce"
                )

            if rmse_col:
                rmse_value = pd.to_numeric(
                    selected_metrics.iloc[0][rmse_col],
                    errors="coerce"
                )


actual_average = (
    actual_values.mean()
    if not actual_values.empty else None
)

predicted_average = (
    predicted_values.mean()
    if not predicted_values.empty else None
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        "Model MAE",
        format_number(mae_value) if mae_value is not None else "N/A"
    )

with kpi2:
    st.metric(
        "Model RMSE",
        format_number(rmse_value) if rmse_value is not None else "N/A"
    )

with kpi3:
    st.metric(
        "Average Actual Demand",
        format_number(actual_average)
    )

with kpi4:
    st.metric(
        "Average Predicted Demand",
        format_number(predicted_average)
    )


# =========================================================
# CHARTS
# =========================================================

section_title("Demand Forecast Analysis")

chart1, chart2 = st.columns([1, 1], gap="large")


# Actual vs Predicted

with chart1:
    st.markdown("**Actual vs Predicted Demand**")

    if date_col and actual_col and prediction_col:

        chart_data = filtered_data.copy()

        chart_data[actual_col] = pd.to_numeric(
            chart_data[actual_col],
            errors="coerce"
        )

        chart_data[prediction_col] = pd.to_numeric(
            chart_data[prediction_col],
            errors="coerce"
        )

        chart_data[date_col] = pd.to_datetime(
            chart_data[date_col],
            errors="coerce"
        )

        chart_data = chart_data.dropna(
            subset=[date_col, actual_col, prediction_col]
        )

        chart_data = chart_data.sort_values(date_col)

        if not chart_data.empty:
            plot_data = chart_data[[date_col, actual_col, prediction_col]]

            plot_data = plot_data.melt(
                id_vars=[date_col],
                value_vars=[actual_col, prediction_col],
                var_name="Demand Type",
                value_name="Demand"
            )

            plot_data["Demand Type"] = plot_data["Demand Type"].replace({
                actual_col: "Actual Demand",
                prediction_col: "Predicted Demand"
            })

            fig = px.line(
                plot_data,
                x=date_col,
                y="Demand",
                color="Demand Type",
                markers=True,
                color_discrete_map={
                    "Actual Demand": "#164B7A",
                    "Predicted Demand": "#6BAED6"
                }
            )

            fig.update_layout(
                height=430,
                margin=dict(l=10, r=10, t=20, b=10),
                legend_title_text="",
                xaxis_title="Week",
                yaxis_title="Demand Units"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No valid data available for this chart.")

    else:
        st.info("Required date, actual demand, or prediction columns are missing.")


# Model Comparison

with chart2:
    st.markdown("**Model Comparison (MAE)**")

    if not metrics.empty:
        metric_model_col = find_column(metrics, {"model"})
        mae_col = find_column(metrics, {"mae"})

        if metric_model_col and mae_col:
            comparison_data = metrics.copy()

            comparison_data[mae_col] = pd.to_numeric(
                comparison_data[mae_col],
                errors="coerce"
            )

            comparison_data = comparison_data.dropna(
                subset=[mae_col]
            )

            comparison_data = comparison_data[
                comparison_data[metric_model_col].isin(MODEL_OPTIONS)
            ]

            if not comparison_data.empty:
                comparison_data["Colour"] = comparison_data[
                    metric_model_col
                ].map(MODEL_COLOURS)

                fig = px.bar(
                    comparison_data,
                    x=metric_model_col,
                    y=mae_col,
                    color=metric_model_col,
                    color_discrete_map=MODEL_COLOURS,
                    text_auto=".2f"
                )

                fig.update_layout(
                    height=430,
                    showlegend=False,
                    margin=dict(l=10, r=10, t=20, b=10),
                    xaxis_title="Model",
                    yaxis_title="Mean Absolute Error"
                )

                fig.update_xaxes(     tickangle=-30,     tickfont=dict(size=10),     automargin=True )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )
            else:
                st.info("No model comparison results available.")
        else:
            st.info("The metrics file is missing the Model or MAE column.")
    else:
        st.info("Model comparison metrics are unavailable.")


# =========================================================
# INVENTORY AND SHORTAGE ALERTS
# =========================================================

section_title("Inventory and Shortage Alerts")

if alerts.empty:
    st.info(
        "No alert CSV files were found. Add test_alerts_all_models.csv "
        "or walkforward_alerts_all_models.csv to your outputs folder."
    )

else:
    alert_data = alerts.copy()

    # Find relevant alert columns

    severity_col = find_column(
        alert_data,
        {"severity", "alert level", "risk level", "status", "alert"}
    )

    alert_model_col = find_column(
        alert_data,
        {"model", "forecast model"}
    )

    alert_category_col = find_column(
        alert_data,
        {"category"}
    )

    alert_sku_col = find_column(
        alert_data,
        {"sku"}
    )


    # Filter alerts by selected model when possible

    if alert_model_col:
        matching_model = alert_data[
            alert_data[alert_model_col].astype(str).str.lower()
            == selected_model.lower()
        ]

        if not matching_model.empty:
            alert_data = matching_model.copy()

    # Apply category and SKU filters

    if alert_category_col and selected_category != "All Categories":
        alert_data = alert_data[
            alert_data[alert_category_col].astype(str) == selected_category
        ]

    if alert_sku_col and selected_sku != "All SKUs":
        alert_data = alert_data[
            alert_data[alert_sku_col].astype(str) == selected_sku
        ]

    if alert_data.empty:
        st.info("No alerts match the selected filters.")

    elif severity_col:

        # Standardise severity labels before counting

        alert_data[severity_col] = (
            alert_data[severity_col]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        # Count each severity

        critical_count = alert_data[severity_col].str.contains(
            "CRITICAL", na=False
        ).sum()

        high_count = alert_data[severity_col].str.contains(
            "HIGH", na=False
        ).sum()

        warning_count = alert_data[severity_col].str.contains(
            "WARNING", na=False
        ).sum()

        ok_count = alert_data[severity_col].str.fullmatch(
            "OK", na=False
        ).sum()

        # Coloured alert cards

        alert1, alert2, alert3, alert4 = st.columns(4)

        alert_cards = [
            (alert1, "CRITICAL", int(critical_count), "#F8D7DA", "#842029"),
            (alert2, "HIGH", int(high_count), "#FCE1CC", "#984B13"),
            (alert3, "WARNING", int(warning_count), "#FFF0BD", "#806000"),
            (alert4, "OK", int(ok_count), "#D5EDDC", "#23643B")
        ]

        for column, name, count, background, text_colour in alert_cards:
            with column:
                st.markdown(f"""
                <div class="alert-card"
                     style="background:{background}; color:{text_colour};">
                    <div class="alert-name">{name}</div>
                    <div class="alert-count">{count}</div>
                </div>
                """, unsafe_allow_html=True)

        # Alert table

        st.markdown("")
        with st.expander("View Detailed Inventory Alerts"):
            st.dataframe(
                style_alerts(alert_data, severity_col),
                use_container_width=True,
                hide_index=True
            )

        # Download alerts

        st.download_button(
            label="Download Alert Results",
            data=alert_data.to_csv(index=False).encode("utf-8"),
            file_name="inventory_alerts.csv",
            mime="text/csv"
        )

    else:
        st.warning(
            "The alert data was loaded, but no severity column was found. "
            "Please check the alert CSV column names."
        )

        with st.expander("View Alert Data"):
            st.dataframe(
                alert_data,
                use_container_width=True,
                hide_index=True
            )


# =========================================================
# PREDICTION RESULTS
# =========================================================

section_title("Prediction Results")

with st.expander("View Prediction Table"):
    st.dataframe(
        filtered_data,
        use_container_width=True,
        hide_index=True
    )

st.download_button(
    label="Download Prediction Results",
    data=filtered_data.to_csv(index=False).encode("utf-8"),
    file_name="filtered_predictions.csv",
    mime="text/csv"
)


# =========================================================
# MODEL PERFORMANCE DETAILS
# =========================================================

section_title("Model Performance Details")

if not metrics.empty:
    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="Download Model Comparison Results",
        data=metrics.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison_results.csv",
        mime="text/csv"
    )
else:
    st.info("Model performance metrics are not available.")


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#718096; font-size:12px;">
        Humanitarian Warehouse Demand Forecasting Dashboard
        <br>
        CSCI218 – Foundations of Artificial Intelligence
    </div>
    """,
    unsafe_allow_html=True
)
