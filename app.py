
import streamlit as st
import pandas as pd
from pathlib import Path
import re

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Humanitarian Demand Forecasting",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
    .main-title {
        font-size: 30px;
        font-weight: 700;
        color: #17324D;
        margin-bottom: 2px;
    }

    .subtitle {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 12px;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background-color: #17324D;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 9px 20px;
        font-weight: 600;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background-color: #285477;
        color: white;
    }

    [data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 14px;
        border-radius: 10px;
    }

    [data-testid="stMetricLabel"] {
        color: #475569;
        font-size: 13px;
    }

    .section-note {
        color: #64748b;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MODEL COLUMN MAPPING
# --------------------------------------------------

model_options = {
    "Naive": "Naive",
    "Moving Average": "Moving average",
    "Linear Regression": "Linear regression",
    "Decision Tree": "Decision Tree",
    "Random Forest": "Random Forest",
    "Neural Network": "Neural network"
}


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def normalise(value):
    """Make names easier to compare despite spaces/capitalisation."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_column(dataframe, possible_names):
    """Find a column even if its capitalisation or spacing differs."""
    lookup = {
        normalise(column): column
        for column in dataframe.columns
    }

    for name in possible_names:
        match = lookup.get(normalise(name))
        if match:
            return match

    return None


def section_heading(title, explanation):
    """Display a section title with a clickable information popover."""
    heading_col, info_col = st.columns([0.92, 0.08])

    with heading_col:
        st.markdown(f"### {title}")

    with info_col:
        with st.popover("ⓘ"):
            st.write(explanation)


def metric_card(label, value, explanation):
    """Display a metric with a clickable explanation."""
    with st.container(border=True):
        st.metric(label, value)

        with st.popover("ⓘ About this metric"):
            st.write(explanation)


# --------------------------------------------------
# LOAD PROJECT DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    predictions = pd.read_csv(
        BASE_DIR / "all_test_predictions.csv"
    )

    metrics = pd.read_csv(
        BASE_DIR / "model_comparison_results.csv"
    )

    predictions["week_end"] = pd.to_datetime(
        predictions["week_end"],
        errors="coerce"
    )

    return predictions, metrics


@st.cache_data
def load_alert_data():
    """
    Load saved alert results if the notebook output files exist.
    These files are optional.
    """

    possible_files = [
        (
            BASE_DIR / "outputs" / "test_alerts_all_models.csv",
            "Test alert results"
        ),
        (
            BASE_DIR / "outputs" / "walkforward_alerts_all_models.csv",
            "Walk-forward alert results"
        )
    ]

    for file_path, description in possible_files:
        if file_path.exists():
            return pd.read_csv(file_path), description

    return None, None


try:
    predictions, metrics = load_data()

except Exception as e:
    st.error(f"Could not load project files: {e}")
    st.stop()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">Humanitarian Warehouse '
    'Demand Forecasting</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-based demand forecasting '
    'and model evaluation for humanitarian supplies</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

overview_col1, overview_col2, overview_col3 = st.columns(3)

overview_col1.caption(
    f"**{len(predictions):,}** test records"
)

overview_col2.caption(
    f"**{predictions['category'].nunique()}** supply categories"
)

overview_col3.caption(
    f"**{predictions['sku'].nunique()}** items analysed"
)


# --------------------------------------------------
# MODEL AND FILTER SELECTION
# --------------------------------------------------

st.markdown("### Forecast Settings")

with st.form("model_selection"):

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        selected_model = st.selectbox(
            "Forecasting Model",
            list(model_options.keys())
        )

    categories = ["All Categories"] + sorted(
        predictions["category"].dropna().unique().tolist()
    )

    with filter_col2:
        selected_category = st.selectbox(
            "Supply Category",
            categories
        )

    filtered_for_sku = predictions.copy()

    if selected_category != "All Categories":
        filtered_for_sku = filtered_for_sku[
            filtered_for_sku["category"] == selected_category
        ]

    sku_options = ["All Items"] + sorted(
        filtered_for_sku["sku"].dropna().unique().tolist()
    )

    with filter_col3:
        selected_sku = st.selectbox(
            "Item / SKU",
            sku_options
        )

    run_button = st.form_submit_button(
        "▶ Run Model",
        use_container_width=True
    )


# --------------------------------------------------
# SAVE SELECTED SETTINGS
# --------------------------------------------------

if run_button:
    st.session_state["has_run"] = True
    st.session_state["selected_model"] = selected_model
    st.session_state["selected_category"] = selected_category
    st.session_state["selected_sku"] = selected_sku


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

if st.session_state.get("has_run", False):

    model_name = st.session_state["selected_model"]
    prediction_column = model_options[model_name]

    category = st.session_state["selected_category"]
    sku = st.session_state["selected_sku"]

    results = predictions.copy()

    if category != "All Categories":
        results = results[
            results["category"] == category
        ]

    if sku != "All Items":
        results = results[
            results["sku"] == sku
        ]

    results = results.dropna(
        subset=["week_end", "demand_units", prediction_column]
    )

    st.divider()

    st.markdown(f"## {model_name} Forecast Results")

    if results.empty:
        st.warning("No results are available for this selection.")

    else:

        # --------------------------------------------------
        # FIND SELECTED MODEL PERFORMANCE
        # --------------------------------------------------

        metrics_model_column = find_column(
            metrics,
            ["Model"]
        )

        model_result = pd.DataFrame()

        if metrics_model_column:
            model_result = metrics[
                metrics[metrics_model_column].apply(normalise)
                == normalise(model_name)
            ]

        # --------------------------------------------------
        # KPI CARDS
        # --------------------------------------------------

        total_actual = results["demand_units"].sum()
        total_predicted = results[prediction_column].sum()

        mae_value = "N/A"
        rmse_value = "N/A"

        if not model_result.empty:
            metric_row = model_result.iloc[0]

            mae_value = f"{metric_row['MAE']:.2f}"
            rmse_value = f"{metric_row['RMSE']:.2f}"

        st.markdown("### Model Performance")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        with kpi1:
            metric_card(
                "MAE",
                mae_value,
                "Mean Absolute Error. It measures the average absolute "
                "difference between actual demand and predicted demand. "
                "A lower MAE means predictions are closer to actual demand."
            )

        with kpi2:
            metric_card(
                "RMSE",
                rmse_value,
                "Root Mean Squared Error. It measures prediction error "
                "while giving larger errors more weight. A lower RMSE "
                "indicates better predictive performance."
            )

        with kpi3:
            metric_card(
                "Actual Demand",
                f"{total_actual:,.0f}",
                "The total recorded demand units across the selected "
                "test records."
            )

        with kpi4:
            metric_card(
                "Predicted Demand",
                f"{total_predicted:,.0f}",
                "The total demand units predicted by the selected model "
                "across the selected test records."
            )

        # --------------------------------------------------
        # MAE IMPROVEMENT VS NAIVE
        # --------------------------------------------------

        if not model_result.empty:
            improvement = metric_row.get("MAE vs naive", None)

            st.caption(
                f"**MAE improvement vs Naive:** {improvement}"
            )

        # --------------------------------------------------
        # ACTUAL VS PREDICTED CHART
        # --------------------------------------------------

        st.divider()

        section_heading(
            "Actual vs Predicted Demand",
            "This chart compares the recorded demand with the "
            "selected model's saved predictions over the test period. "
            "When All Items is selected, demand is aggregated by week."
        )

        if sku == "All Items":
            chart_data = (
                results.groupby("week_end")[
                    ["demand_units", prediction_column]
                ]
                .sum()
                .reset_index()
            )
        else:
            chart_data = results[
                ["week_end", "demand_units", prediction_column]
            ].copy()

            chart_data = chart_data.sort_values("week_end")

        chart_data = chart_data.rename(
            columns={
                "demand_units": "Actual Demand",
                prediction_column: "Predicted Demand"
            }
        )

        chart_data = chart_data.set_index("week_end")

        st.line_chart(
            chart_data[
                ["Actual Demand", "Predicted Demand"]
            ],
            use_container_width=True
        )

        st.caption(
            "The chart uses the saved test-set predictions; "
            "it does not train a new model when you click Run Model."
        )

        # --------------------------------------------------
        # MODEL COMPARISON CHART
        # --------------------------------------------------

        st.divider()

        section_heading(
            "Model Comparison",
            "This chart compares the saved Mean Absolute Error (MAE) "
            "of the available models. Lower MAE indicates a smaller "
            "average prediction error. The comparison uses the "
            "overall model evaluation results."
        )

        if {"Model", "MAE"}.issubset(metrics.columns):

            comparison_data = metrics[
                ["Model", "MAE"]
            ].copy()

            comparison_data["MAE"] = pd.to_numeric(
                comparison_data["MAE"],
                errors="coerce"
            )

            comparison_data = comparison_data.dropna(
                subset=["MAE"]
            )

            st.bar_chart(
                comparison_data,
                x="Model",
                y="MAE",
                use_container_width=True
            )

        else:
            st.info(
                "Model comparison data is not available "
                "in the expected format."
            )

        # --------------------------------------------------
        # INVENTORY SHORTAGE ALERTS
        # --------------------------------------------------

        st.divider()

        section_heading(
            "Inventory Shortage Alerts",
            "These alerts come from the saved inventory alert outputs "
            "generated by the project notebook. They are forecast-based "
            "inventory risk indicators, not confirmed real-world "
            "stockouts. The notebook uses projected stock and demand "
            "to classify inventory risk."
        )

        alert_data, alert_description = load_alert_data()

        if alert_data is None:

            st.info(
                "No saved shortage alert CSV was found. "
                "To display this section, add the notebook's "
                "test_alerts_all_models.csv or "
                "walkforward_alerts_all_models.csv file "
                "inside the outputs folder."
            )

        elif alert_data.empty:

            st.info("The saved alert file contains no records.")

        else:

            st.caption(f"Data source: {alert_description}")

            alert_results = alert_data.copy()

            # Apply category filter if the alert file contains it
            alert_category_col = find_column(
                alert_results,
                ["category", "supply_category"]
            )

            if category != "All Categories" and alert_category_col:
                alert_results = alert_results[
                    alert_results[alert_category_col] == category
                ]

            # Apply SKU filter if the alert file contains it
            alert_sku_col = find_column(
                alert_results,
                ["sku", "item_code", "item"]
            )

            if sku != "All Items" and alert_sku_col:
                alert_results = alert_results[
                    alert_results[alert_sku_col].astype(str) == str(sku)
                ]

            severity_col = find_column(
                alert_results,
                ["alert_level", "alert", "severity", "risk_level"]
            )

            if alert_results.empty:

                st.info(
                    "No alert records match the selected category "
                    "and item filters."
                )

            else:

                if severity_col:

                    severity_values = (
                        alert_results[severity_col]
                        .astype(str)
                        .str.upper()
                    )

                    severity1, severity2, severity3, severity4 = (
                        st.columns(4)
                    )

                    severity1.metric(
                        "Critical",
                        int(severity_values.str.contains("CRITICAL").sum())
                    )

                    severity2.metric(
                        "High",
                        int(severity_values.str.contains("HIGH").sum())
                    )

                    severity3.metric(
                        "Warning",
                        int(severity_values.str.contains("WARNING").sum())
                    )

                    severity4.metric(
                        "OK",
                        int(
                            severity_values.str.fullmatch(
                                "OK|LOW RISK|SAFE"
                            ).sum()
                        )
                    )

                st.dataframe(
                    alert_results,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "Download Shortage Alerts",
                    data=alert_results.to_csv(index=False),
                    file_name="shortage_alerts.csv",
                    mime="text/csv"
                )

        # --------------------------------------------------
        # PREDICTION RESULTS TABLE
        # --------------------------------------------------

        st.divider()

        section_heading(
            "Prediction Results",
            "This table shows the actual demand and the selected "
            "model's predicted demand for each test record. Prediction "
            "Error is calculated as predicted demand minus actual demand. "
            "Absolute Error shows the size of the error without its sign."
        )

        table = results.copy()

        table["Prediction Error"] = (
            table[prediction_column] - table["demand_units"]
        )

        table["Absolute Error"] = (
            table["Prediction Error"].abs()
        )

        table = table.rename(
            columns={
                "week_end": "Week Ending",
                "sku": "Item Code",
                "item_name": "Item Name",
                "category": "Category",
                "demand_units": "Actual Demand",
                prediction_column: "Predicted Demand"
            }
        )

        display_columns = [
            "Week Ending",
            "Item Code",
            "Item Name",
            "Category",
            "Actual Demand",
            "Predicted Demand",
            "Prediction Error",
            "Absolute Error"
        ]

        table = table[display_columns].sort_values(
            "Week Ending"
        )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Prediction Results",
            data=table.to_csv(index=False),
            file_name=f"{model_name.replace(' ', '_')}_results.csv",
            mime="text/csv"
        )

else:

    st.info(
        "Choose a forecasting model and filters, "
        "then click Run Model to view the results."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "CSCI218 | Foundations of Artificial Intelligence | "
    "Humanitarian Warehouse Demand Forecasting"
)
