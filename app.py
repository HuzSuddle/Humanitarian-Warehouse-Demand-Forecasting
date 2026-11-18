
import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import re

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Humanitarian Demand Forecasting",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

# Colour palette
NAVY = "#17324D"
BLUE = "#4A90D9"
TEAL = "#35A99A"
CORAL = "#E99A75"
PURPLE = "#9382C5"
GOLD = "#E7B85C"

PALETTE = [
    BLUE, TEAL, CORAL, PURPLE, GOLD,
    "#5B9BD5", "#78B7A8", "#D989A5"
]

MODEL_OPTIONS = {
    "Naive": "Naive",
    "Moving Average": "Moving average",
    "Linear Regression": "Linear regression",
    "Decision Tree": "Decision Tree",
    "Random Forest": "Random Forest",
    "Neural Network": "Neural network"
}


# ==================================================
# 2. PAGE STYLING
# ==================================================

st.markdown("""
<style>

/* Main page */
.stApp {
    background-color: #F4F7FB;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 0.8rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1500px;
}

/* Header */
.dashboard-header {
    background: linear-gradient(110deg, #DCEEFF, #EAF5FF);
    border: 1px solid #C9E2F7;
    border-radius: 14px;
    padding: 17px 22px;
    margin-bottom: 12px;
}

.dashboard-title {
    color: #17324D;
    font-size: 27px;
    font-weight: 750;
    margin: 0;
}

.dashboard-subtitle {
    color: #526B82;
    font-size: 13px;
    margin-top: 4px;
}

/* Section headings */
h2, h3 {
    color: #17324D !important;
}

/* Filter area */
div[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid #DCE5EF;
    border-radius: 12px;
    padding: 12px 15px 8px 15px;
}

/* Buttons */
div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background: #2878BD;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 650;
    min-height: 38px;
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background: #1D6099;
    color: white;
    border: none;
}

/* KPI cards */
.kpi-card {
    border-radius: 12px;
    padding: 12px 14px;
    min-height: 91px;
    border: 1px solid rgba(23, 50, 77, 0.08);
}

.kpi-label {
    font-size: 12px;
    font-weight: 650;
    color: #40566B;
    margin-bottom: 5px;
}

.kpi-value {
    font-size: 25px;
    font-weight: 750;
    color: #17324D;
}

.kpi-hint {
    font-size: 10px;
    color: #60758A;
    margin-top: 2px;
}

/* Chart and table panels */
.panel {
    background: white;
    border: 1px solid #DFE7F0;
    border-radius: 12px;
    padding: 12px 14px;
}

/* Small muted text */
.small-note {
    color: #718096;
    font-size: 11px;
}

/* Alert cards */
.alert-card {
    border-radius: 10px;
    padding: 10px 12px;
    text-align: center;
    border: 1px solid rgba(0,0,0,0.06);
}

.alert-number {
    font-size: 23px;
    font-weight: 750;
    color: #26394B;
}

.alert-label {
    font-size: 11px;
    font-weight: 700;
    color: #45596D;
}

/* Reduce default vertical gaps */
div[data-testid="stVerticalBlock"] {
    gap: 0.65rem;
}

/* Footer */
.footer {
    text-align: center;
    color: #8090A0;
    font-size: 11px;
    padding: 5px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. HELPER FUNCTIONS
# ==================================================

def normalise(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_column(dataframe, possible_names):
    lookup = {
        normalise(column): column
        for column in dataframe.columns
    }

    for name in possible_names:
        found = lookup.get(normalise(name))
        if found:
            return found

    return None


def info_popover(explanation):
    with st.popover("ⓘ"):
        st.write(explanation)


def show_kpi(label, value, background, hint=""):
    st.markdown(
        f"""
        <div class="kpi-card" style="background:{background};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-hint">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_alert_card(label, number, background):
    st.markdown(
        f"""
        <div class="alert-card" style="background:{background};">
            <div class="alert-number">{number}</div>
            <div class="alert-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# 4. LOAD MAIN DATA
# ==================================================

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
    alert_files = [
        BASE_DIR / "outputs" / "test_alerts_all_models.csv",
        BASE_DIR / "outputs" / "walkforward_alerts_all_models.csv"
    ]

    for file_path in alert_files:
        if file_path.exists():
            return pd.read_csv(file_path), file_path.name

    return None, None


try:
    predictions, metrics = load_data()

except Exception as error:
    st.error(f"Could not load project files: {error}")
    st.stop()


# ==================================================
# 5. HEADER
# ==================================================

st.markdown("""
<div class="dashboard-header">
    <div class="dashboard-title">
        Humanitarian Warehouse Demand Forecasting
    </div>
    <div class="dashboard-subtitle">
        AI-based demand forecasting and inventory risk analysis
        for humanitarian supplies
    </div>
</div>
""", unsafe_allow_html=True)


# ==================================================
# 6. DATASET SUMMARY
# ==================================================

summary1, summary2, summary3 = st.columns(3)

summary1.markdown(
    f"<span style='color:#526B82;font-size:12px'>"
    f"📊 <b>{len(predictions):,}</b> test records</span>",
    unsafe_allow_html=True
)

summary2.markdown(
    f"<span style='color:#526B82;font-size:12px'>"
    f"📦 <b>{predictions['category'].nunique()}</b> supply categories</span>",
    unsafe_allow_html=True
)

summary3.markdown(
    f"<span style='color:#526B82;font-size:12px'>"
    f"🏷️ <b>{predictions['sku'].nunique()}</b> items analysed</span>",
    unsafe_allow_html=True
)


# ==================================================
# 7. FILTERS
# ==================================================

with st.form("model_selection"):

    filter1, filter2, filter3, filter4 = st.columns(
        [1.2, 1.2, 1.2, 0.8],
        vertical_alignment="bottom"
    )

    with filter1:
        selected_model = st.selectbox(
            "Forecasting Model",
            list(MODEL_OPTIONS.keys())
        )

    with filter2:
        category_options = ["All Categories"] + sorted(
            predictions["category"].dropna().unique().tolist()
        )

        selected_category = st.selectbox(
            "Supply Category",
            category_options
        )

    filtered_skus = predictions.copy()

    if selected_category != "All Categories":
        filtered_skus = filtered_skus[
            filtered_skus["category"] == selected_category
        ]

    sku_options = ["All Items"] + sorted(
        filtered_skus["sku"].dropna().unique().tolist()
    )

    with filter3:
        selected_sku = st.selectbox(
            "Item / SKU",
            sku_options
        )

    with filter4:
        st.write("")
        run_button = st.form_submit_button(
            "▶ Run Model",
            use_container_width=True
        )


if run_button:
    st.session_state["has_run"] = True
    st.session_state["selected_model"] = selected_model
    st.session_state["selected_category"] = selected_category
    st.session_state["selected_sku"] = selected_sku


# ==================================================
# 8. DASHBOARD RESULTS
# ==================================================

if not st.session_state.get("has_run", False):

    st.info(
        "Choose a model and filters, then click Run Model "
        "to display the forecasting results."
    )

else:

    model_name = st.session_state["selected_model"]
    prediction_column = MODEL_OPTIONS[model_name]
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

    if results.empty:
        st.warning("No records are available for this selection.")
        st.stop()

    # Overall model performance
    model_column = find_column(metrics, ["Model"])

    model_result = pd.DataFrame()

    if model_column:
        model_result = metrics[
            metrics[model_column].apply(normalise)
            == normalise(model_name)
        ]

    mae_value = "N/A"
    rmse_value = "N/A"

    if not model_result.empty:
        metric_row = model_result.iloc[0]
        mae_value = f"{float(metric_row['MAE']):.2f}"
        rmse_value = f"{float(metric_row['RMSE']):.2f}"

    total_actual = results["demand_units"].sum()
    total_predicted = results[prediction_column].sum()

    # --------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------

    st.markdown(
        "<div style='font-size:17px;font-weight:700;color:#17324D;"
        "margin:12px 0 8px 0'>Forecast Overview</div>",
        unsafe_allow_html=True
    )

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        show_kpi(
            "MAE",
            mae_value,
            "#E3F0FF",
            "Mean Absolute Error"
        )
        info_popover(
            "Mean Absolute Error measures the average absolute "
            "difference between actual and predicted demand. "
            "Lower values indicate more accurate predictions. "
            "This is the overall model score from the evaluation CSV."
        )

    with k2:
        show_kpi(
            "RMSE",
            rmse_value,
            "#E1F4EF",
            "Root Mean Squared Error"
        )
        info_popover(
            "Root Mean Squared Error gives more weight to larger "
            "prediction errors. Lower values are better. "
            "This is the overall model score from the evaluation CSV."
        )

    with k3:
        show_kpi(
            "Actual Demand",
            f"{total_actual:,.0f}",
            "#FFF0E5",
            "Total recorded demand units"
        )
        info_popover(
            "Total actual demand across the selected records "
            "in the test dataset."
        )

    with k4:
        show_kpi(
            "Predicted Demand",
            f"{total_predicted:,.0f}",
            "#EEEAFE",
            "Total forecast demand units"
        )
        info_popover(
            "Total demand predicted by the selected model "
            "across the selected test records."
        )

    if not model_result.empty:
        st.caption(
            f"Selected model: **{model_name}**  |  "
            f"MAE improvement vs Naive: "
            f"**{metric_row.get('MAE vs naive', 'N/A')}**"
        )

    # --------------------------------------------------
    # CHARTS SIDE BY SIDE
    # --------------------------------------------------

    st.markdown(
        "<div style='font-size:17px;font-weight:700;color:#17324D;"
        "margin:14px 0 5px 0'>Forecast Visualisation</div>",
        unsafe_allow_html=True
    )

    chart_left, chart_right = st.columns(2, gap="medium")

    # Actual vs predicted
    with chart_left:

        heading_col, info_col = st.columns([0.88, 0.12])

        with heading_col:
            st.markdown("**Actual vs Predicted Demand**")

        with info_col:
            info_popover(
                "Compares actual recorded demand with the selected "
                "model's saved predictions over the test period. "
                "When All Items is selected, values are added together "
                "for each week."
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

        chart_data = chart_data.rename(
            columns={
                "demand_units": "Actual Demand",
                prediction_column: "Predicted Demand"
            }
        )

        chart_long = chart_data.melt(
            id_vars=["week_end"],
            value_vars=["Actual Demand", "Predicted Demand"],
            var_name="Demand Type",
            value_name="Units"
        )

        demand_chart = (
            alt.Chart(chart_long)
            .mark_line(
                strokeWidth=2.8,
                point=alt.OverlayMarkDef(size=25)
            )
            .encode(
                x=alt.X(
                    "week_end:T",
                    title="Week Ending",
                    axis=alt.Axis(labelAngle=-35)
                ),
                y=alt.Y("Units:Q", title="Demand Units"),
                color=alt.Color(
                    "Demand Type:N",
                    scale=alt.Scale(
                        domain=["Actual Demand", "Predicted Demand"],
                        range=[BLUE, TEAL]
                    ),
                    legend=alt.Legend(orient="bottom")
                ),
                tooltip=[
                    alt.Tooltip("week_end:T", title="Week"),
                    alt.Tooltip("Demand Type:N"),
                    alt.Tooltip("Units:Q", format=",.0f")
                ]
            )
            .properties(height=245)
            .configure_view(strokeOpacity=0)
            .configure_axis(
                gridColor="#E9EEF4",
                labelColor="#526579",
                titleColor="#526579"
            )
            .configure_legend(
                labelColor="#526579",
                titleColor="#526579"
            )
        )

        st.altair_chart(
            demand_chart,
            use_container_width=True
        )

    # Model comparison
    with chart_right:

        heading_col, info_col = st.columns([0.88, 0.12])

        with heading_col:
            st.markdown("**Model Comparison (MAE)**")

        with info_col:
            info_popover(
                "Compares the overall Mean Absolute Error of the "
                "available forecasting models. Lower bars mean "
                "smaller average prediction errors. These values "
                "come from the saved model comparison results."
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

            model_order = comparison_data.sort_values(
                "MAE"
            )["Model"].tolist()

            comparison_chart = (
                alt.Chart(comparison_data)
                .mark_bar(
                    cornerRadiusEnd=5,
                    size=25
                )
                .encode(
                    x=alt.X(
                        "Model:N",
                        sort=model_order,
                        title=None,
                        axis=alt.Axis(
                            labelAngle=-25,
                            labelLimit=150
                        )
                    ),
                    y=alt.Y(
                        "MAE:Q",
                        title="Mean Absolute Error"
                    ),
                    color=alt.Color(
                        "Model:N",
                        scale=alt.Scale(
                            domain=model_order,
                            range=PALETTE[:len(model_order)]
                        ),
                        legend=None
                    ),
                    tooltip=[
                        alt.Tooltip("Model:N"),
                        alt.Tooltip("MAE:Q", format=".2f")
                    ]
                )
                .properties(height=245)
                .configure_view(strokeOpacity=0)
                .configure_axis(
                    gridColor="#E9EEF4",
                    labelColor="#526579",
                    titleColor="#526579"
                )
            )

            st.altair_chart(
                comparison_chart,
                use_container_width=True
            )

        else:
            st.info("Model comparison data is unavailable.")

    # --------------------------------------------------
    # INVENTORY SHORTAGE ALERTS
    # --------------------------------------------------

    st.markdown("---")

    alert_heading, alert_info = st.columns([0.92, 0.08])

    with alert_heading:
        st.markdown(
            "<div style='font-size:17px;font-weight:700;"
            "color:#17324D'>Inventory Shortage Alerts</div>",
            unsafe_allow_html=True
        )

    with alert_info:
        info_popover(
            "These are forecast-based inventory risk alerts from "
            "the notebook's saved alert outputs. They are not "
            "confirmed real-world stockouts. The notebook uses "
            "projected stock and predicted demand to classify risk."
        )

    alert_data, alert_filename = load_alert_data()

    if alert_data is None:

        st.info(
            "No saved alert CSV found. Add the notebook's alert "
            "CSV files to the outputs folder to display this section."
        )

    elif alert_data.empty:

        st.info("The saved alert file contains no records.")

    else:

        alert_results = alert_data.copy()

        alert_model_col = find_column(
            alert_results,
            ["model", "model_name"]
        )

        if alert_model_col:
            matching_model = alert_results[
                alert_results[alert_model_col].apply(normalise)
                == normalise(model_name)
            ]

            if not matching_model.empty:
                alert_results = matching_model

        alert_category_col = find_column(
            alert_results,
            ["category", "supply_category"]
        )

        if category != "All Categories" and alert_category_col:
            alert_results = alert_results[
                alert_results[alert_category_col] == category
            ]

        alert_sku_col = find_column(
            alert_results,
            ["sku", "item_code"]
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

            st.info("No alert records match the current filters.")

        elif severity_col:

            severity = (
                alert_results[severity_col]
                .astype(str)
                .str.upper()
                .str.strip()
            )

            a1, a2, a3, a4 = st.columns(4)

            with a1:
                show_alert_card(
                    "CRITICAL",
                    int(severity.str.contains("CRITICAL").sum()),
                    "#FBE2E2"
                )

            with a2:
                show_alert_card(
                    "HIGH",
                    int(severity.str.fullmatch("HIGH").sum()),
                    "#FCEAD9"
                )

            with a3:
                show_alert_card(
                    "WARNING",
                    int(severity.str.contains("WARNING").sum()),
                    "#FFF4D6"
                )

            with a4:
                show_alert_card(
                    "OK",
                    int(severity.str.fullmatch("OK").sum()),
                    "#DDF3E8"
                )

            st.caption(f"Source: {alert_filename}")

            st.dataframe(
                alert_results,
                use_container_width=True,
                hide_index=True,
                height=220
            )

            st.download_button(
                "Download Alert Results",
                data=alert_results.to_csv(index=False),
                file_name="shortage_alerts.csv",
                mime="text/csv"
            )

        else:

            st.caption(f"Source: {alert_filename}")

            st.dataframe(
                alert_results,
                use_container_width=True,
                hide_index=True,
                height=220
            )

    # --------------------------------------------------
    # PREDICTION RESULTS TABLE
    # --------------------------------------------------

    st.markdown("---")

    table_heading, table_info = st.columns([0.92, 0.08])

    with table_heading:
        st.markdown(
            "<div style='font-size:17px;font-weight:700;"
            "color:#17324D'>Prediction Results</div>",
            unsafe_allow_html=True
        )

    with table_info:
        info_popover(
            "The table displays actual and predicted demand for "
            "each selected test record. Prediction Error is "
            "predicted demand minus actual demand. Absolute Error "
            "shows the size of the error without its sign."
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

    table = table[display_columns].sort_values("Week Ending")

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        height=270
    )

    st.download_button(
        "⬇ Download Prediction Results",
        data=table.to_csv(index=False),
        file_name=f"{model_name.replace(' ', '_')}_results.csv",
        mime="text/csv"
    )


# ==================================================
# 9. FOOTER
# ==================================================

st.markdown("""
<div class="footer">
    CSCI218 | Foundations of Artificial Intelligence |
    Humanitarian Warehouse Demand Forecasting
</div>
""", unsafe_allow_html=True)
