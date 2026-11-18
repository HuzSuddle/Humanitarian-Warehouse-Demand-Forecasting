
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

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
    "Naive": "#A9C7E8",
    "Moving average": "#7FAEDB",
    "Linear regression": "#548FCB",
    "Decision Tree": "#3679B9",
    "Random Forest": "#205D9A",
    "Neural network": "#123F70"
}

ALERT_COLORS = {
    "CRITICAL": "#FDE2E2",
    "HIGH": "#FCE8D5",
    "WARNING": "#FFF4CC",
    "OK": "#DDF3E4"
}

# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #F7F9FC;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
        max-width: 1450px;
    }

    .dashboard-header {
        background: linear-gradient(110deg, #DCEEFF, #EFF7FF);
        padding: 20px 25px;
        border-radius: 13px;
        border: 1px solid #D1E5F8;
        margin-bottom: 16px;
    }

    .dashboard-header h1 {
        color: #174A78;
        font-size: 27px;
        margin: 0;
        padding: 0;
    }

    .dashboard-header p {
        color: #54718D;
        font-size: 13px;
        margin: 5px 0 0 0;
    }

    div.stButton > button {
        background-color: #2878BE;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        min-height: 42px;
    }

    div.stButton > button:hover {
        background-color: #185F9E;
        color: white;
        border: none;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E1EAF3;
        padding: 12px 15px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(30, 70, 110, 0.04);
    }

    div[data-testid="stMetricLabel"] {
        color: #60758A;
        font-size: 12px;
    }

    div[data-testid="stMetricValue"] {
        color: #174A78;
        font-size: 23px;
    }

    h2, h3 {
        color: #244D70;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
    }

    .small-note {
        color: #718399;
        font-size: 12px;
    }

    hr {
        margin-top: 12px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def normalise(value):
    return str(value).strip().lower().replace("_", " ").replace("-", " ")


def find_column(df, possible_names):
    """Find a column even if its capitalisation or spacing differs."""
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


def section_title(title, explanation):
    st.markdown(f"### {title}")
    st.caption(explanation)


def style_alerts(df, severity_col):
    """Apply red, orange, yellow and green backgrounds to alert levels."""

    def colour_severity(value):
        level = str(value).upper().strip()

        for alert, colour in ALERT_COLORS.items():
            if alert in level:
                return f"background-color: {colour}; color: #263746; font-weight: 600;"

        return ""

    return df.style.map(colour_severity, subset=[severity_col])


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

predictions = load_csv(PREDICTION_FILE)
metrics = load_csv(METRICS_FILE)

if predictions is None:
    st.error("Could not find all_test_predictions.csv. Please check your project folder.")
    st.stop()

if metrics is None:
    st.error("Could not find model_comparison_results.csv. Please check your project folder.")
    st.stop()

# Optional shortage alert data
alerts = None

possible_alert_files = [
    OUTPUT_DIR / "test_alerts_all_models.csv",
    OUTPUT_DIR / "walkforward_alerts_all_models.csv"
]

for file in possible_alert_files:
    if file.exists():
        alerts = pd.read_csv(file)
        break


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="dashboard-header">
    <h1>Humanitarian Warehouse Demand Forecasting</h1>
    <p>Weekly demand prediction, model comparison and inventory shortage monitoring</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

category_col = find_column(predictions, ["category"])
sku_col = find_column(predictions, ["sku"])
week_col = find_column(predictions, ["week_end"])

overview1, overview2, overview3 = st.columns(3)

with overview1:
    st.metric("Prediction Records", f"{len(predictions):,}")

with overview2:
    st.metric(
        "Supply Categories",
        predictions[category_col].nunique() if category_col else "N/A"
    )

with overview3:
    st.metric(
        "Items / SKUs",
        predictions[sku_col].nunique() if sku_col else "N/A"
    )

st.divider()


# --------------------------------------------------
# FILTERS
# --------------------------------------------------

st.markdown("### Forecast Settings")

filter1, filter2, filter3, filter4 = st.columns([1.2, 1.2, 1.2, 0.8])

with filter1:
    selected_model = st.selectbox(
        "Select Model",
        MODEL_OPTIONS
    )

with filter2:
    if category_col:
        categories = ["All Categories"] + sorted(
            predictions[category_col].dropna().unique().tolist()
        )
        selected_category = st.selectbox("Supply Category", categories)
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

        selected_sku = st.selectbox("Item / SKU", sku_options)
    else:
        selected_sku = "All SKUs"

with filter4:
    st.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
    run_model = st.button("▶ Run Model", use_container_width=True)


# --------------------------------------------------
# RUN SELECTED MODEL
# --------------------------------------------------

if run_model:
    st.session_state["run_model"] = True
    st.session_state["selected_model"] = selected_model
    st.session_state["selected_category"] = selected_category
    st.session_state["selected_sku"] = selected_sku


if "run_model" not in st.session_state:
    st.info("Choose your model and filters, then click **Run Model** to view the results.")
    st.stop()


# Use the saved selections after the button is clicked
selected_model = st.session_state["selected_model"]
selected_category = st.session_state["selected_category"]
selected_sku = st.session_state["selected_sku"]


# --------------------------------------------------
# FILTER PREDICTION DATA
# --------------------------------------------------

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
    st.error(f"The prediction column for {selected_model} was not found.")
    st.stop()

if filtered.empty:
    st.warning("No prediction records were found for these filters.")
    st.stop()

filtered[selected_model] = pd.to_numeric(
    filtered[selected_model], errors="coerce"
)

filtered["demand_units"] = pd.to_numeric(
    filtered["demand_units"], errors="coerce"
)

filtered = filtered.dropna(
    subset=["demand_units", selected_model]
)

if filtered.empty:
    st.warning("There are no valid prediction values for this selection.")
    st.stop()


# --------------------------------------------------
# MODEL PERFORMANCE METRICS
# --------------------------------------------------

metric_model_col = find_column(metrics, ["Model"])
mae_col = find_column(metrics, ["MAE"])
rmse_col = find_column(metrics, ["RMSE"])
mae_vs_naive_col = find_column(metrics, ["MAE vs naive"])

model_metrics = None

if metric_model_col:
    matching_metrics = metrics[
        metrics[metric_model_col].apply(normalise) == normalise(selected_model)
    ]

    if not matching_metrics.empty:
        model_metrics = matching_metrics.iloc[0]

actual_total = filtered["demand_units"].sum()
predicted_total = filtered[selected_model].sum()

actual_values = filtered["demand_units"]
predicted_values = filtered[selected_model]

mae_value = (
    model_metrics[mae_col]
    if model_metrics is not None and mae_col
    else (actual_values - predicted_values).abs().mean()
)

rmse_value = (
    model_metrics[rmse_col]
    if model_metrics is not None and rmse_col
    else (((actual_values - predicted_values) ** 2).mean()) ** 0.5
)


# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------

st.divider()
st.markdown(f"### Results: {selected_model}")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("MAE", format_number(mae_value))

with kpi2:
    st.metric("RMSE", format_number(rmse_value))

with kpi3:
    st.metric("Actual Demand", f"{actual_total:,.0f}")

with kpi4:
    st.metric("Predicted Demand", f"{predicted_total:,.0f}")

st.caption(
    "MAE and RMSE measure prediction error. Lower values indicate better prediction accuracy. "
    "Demand totals refer to the currently selected filters."
)


# --------------------------------------------------
# GRAPHS: SIDE BY SIDE
# --------------------------------------------------

st.divider()

chart_left, chart_right = st.columns([1.35, 1])

# Actual vs Predicted
with chart_left:
    section_title(
        "Actual vs Predicted Demand",
        "Compare the actual weekly demand with the selected model's predictions."
    )

    chart_data = filtered.copy()

    if week_col:
        chart_data[week_col] = pd.to_datetime(
            chart_data[week_col], errors="coerce"
        )
        chart_data = chart_data.sort_values(week_col)

        chart_data = chart_data.groupby(week_col, as_index=False)[
            ["demand_units", selected_model]
        ].sum()

        chart_data = chart_data.rename(columns={
            week_col: "Week",
            "demand_units": "Actual Demand",
            selected_model: "Predicted Demand"
        })

        chart_data = chart_data.melt(
            id_vars="Week",
            value_vars=["Actual Demand", "Predicted Demand"],
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

        line_chart.update_layout(
            height=330,
            margin=dict(l=10, r=10, t=15, b=10),
            legend_title_text="",
            xaxis_title="Week",
            yaxis_title="Demand (units)",
            hovermode="x unified"
        )

        st.plotly_chart(line_chart, use_container_width=True)

    else:
        st.info("The week_end column was not found, so the weekly graph cannot be displayed.")


# Model comparison
with chart_right:
    section_title(
        "Model Comparison",
        "Compare MAE across all available models. Lower MAE is better."
    )

    if metric_model_col and mae_col:
        comparison = metrics[[metric_model_col, mae_col]].copy()
        comparison[mae_col] = pd.to_numeric(
            comparison[mae_col], errors="coerce"
        )
        comparison = comparison.dropna(subset=[mae_col])
        comparison = comparison.sort_values(mae_col, ascending=True)

        comparison["Colour"] = comparison[metric_model_col].map(
            lambda name: MODEL_COLORS.get(
                next(
                    (m for m in MODEL_COLORS if normalise(m) == normalise(name)),
                    None
                ),
                "#7FAEDB"
            )
        )

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
            margin=dict(l=10, r=45, t=15, b=10),
            showlegend=False,
            xaxis_title="Mean Absolute Error (MAE)",
            yaxis_title="",
            yaxis=dict(autorange="reversed")
        )

        st.plotly_chart(bar_chart, use_container_width=True)
    else:
        st.info("Model comparison columns could not be identified in the metrics file.")


# --------------------------------------------------
# INVENTORY SHORTAGE ALERTS
# --------------------------------------------------

st.divider()

section_title(
    "Inventory Shortage Alerts",
    "Forecast-based inventory risk indicators. These are not confirmed stockouts."
)

if alerts is None:
    st.info(
        "No alert CSV was found. Add test_alerts_all_models.csv or "
        "walkforward_alerts_all_models.csv inside your outputs folder to display alerts."
    )

else:
    alert_data = alerts.copy()

    alert_category_col = find_column(alert_data, ["category"])
    alert_sku_col = find_column(alert_data, ["sku"])
    alert_model_col = find_column(alert_data, ["model"])
    severity_col = find_column(
        alert_data,
        ["alert_level", "alert", "severity", "risk_level", "risk"]
    )

    # Filter alerts using the selected category and SKU
    if selected_category != "All Categories" and alert_category_col:
        alert_data = alert_data[
            alert_data[alert_category_col] == selected_category
        ]

    if selected_sku != "All SKUs" and alert_sku_col:
        alert_data = alert_data[
            alert_data[alert_sku_col] == selected_sku
        ]

    # Filter by selected model if the alert file contains a model column
    if alert_model_col:
        matching_alerts = alert_data[
            alert_data[alert_model_col].apply(normalise) == normalise(selected_model)
        ]

        if not matching_alerts.empty:
            alert_data = matching_alerts

    if severity_col and not alert_data.empty:
        alert_data[severity_col] = (
            alert_data[severity_col].astype(str).str.upper().str.strip()
        )

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

        alert1, alert2, alert3, alert4 = st.columns(4)

        with alert1:
            st.metric("🔴 Critical", int(critical_count))

        with alert2:
            st.metric("🟠 High", int(high_count))

        with alert3:
            st.metric("🟡 Warning", int(warning_count))

        with alert4:
            st.metric("🟢 OK", int(ok_count))

        # Colour-coded alert table
        with st.expander("View shortage alert details", expanded=True):
            st.dataframe(
                style_alerts(alert_data, severity_col),
                use_container_width=True,
                hide_index=True
            )

        st.download_button(
            "Download Alert Results",
            data=alert_data.to_csv(index=False).encode("utf-8"),
            file_name="shortage_alert_results.csv",
            mime="text/csv"
        )

    elif alert_data.empty:
        st.info("No alert records match the selected filters.")

    else:
        st.warning(
            "An alert file was found, but its severity column could not be identified. "
            "The available alert data is shown below."
        )
        st.dataframe(alert_data, use_container_width=True, hide_index=True)


# --------------------------------------------------
# PREDICTION TABLE
# --------------------------------------------------

st.divider()

with st.expander("View detailed prediction results", expanded=False):
    result_table = filtered.copy()

    result_table["Prediction Error"] = (
        result_table[selected_model] - result_table["demand_units"]
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
        data=result_table.to_csv(index=False).encode("utf-8"),
        file_name="demand_predictions.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<hr>
<p style="text-align:center; color:#8293A5; font-size:11px;">
    Humanitarian Warehouse Demand Forecasting | CSCI218 Foundations of AI
</p>
""", unsafe_allow_html=True)
