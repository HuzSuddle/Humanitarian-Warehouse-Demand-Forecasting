
import streamlit as st
import pandas as pd
from pathlib import Path

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
    }
    .subtitle {
        color: #64748b;
        font-size: 15px;
    }
    div.stButton > button {
        background-color: #17324D;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background-color: #285477;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD EXISTING CSV FILES
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

st.write("")
st.divider()


# --------------------------------------------------
# PROJECT SUMMARY
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Test Records",
    f"{len(predictions):,}"
)

col2.metric(
    "Supply Categories",
    predictions["category"].nunique()
)

col3.metric(
    "Items Analysed",
    predictions["sku"].nunique()
)

st.divider()


# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

st.subheader("Forecasting Model")

model_options = {
    "Naive": "Naive",
    "Moving Average": "Moving average",
    "Linear Regression": "Linear regression",
    "Decision Tree": "Decision Tree",
    "Random Forest": "Random Forest",
    "Neural Network": "Neural network"
}

with st.form("model_selection"):

    selected_model = st.selectbox(
        "Choose a model",
        list(model_options.keys())
    )

    categories = ["All Categories"] + sorted(
        predictions["category"].dropna().unique().tolist()
    )

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

    selected_sku = st.selectbox(
        "Select Item",
        sku_options
    )

    run_button = st.form_submit_button(
        "▶ Run Model",
        use_container_width=True
    )


# --------------------------------------------------
# DISPLAY RESULTS AFTER BUTTON CLICK
# --------------------------------------------------

if run_button:

    st.session_state["has_run"] = True
    st.session_state["selected_model"] = selected_model
    st.session_state["selected_category"] = selected_category
    st.session_state["selected_sku"] = selected_sku


if st.session_state.get("has_run", False):

    model_name = st.session_state["selected_model"]
    prediction_column = model_options[model_name]

    category = st.session_state["selected_category"]
    sku = st.session_state["selected_sku"]

    # Filter prediction records

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

    st.subheader(f"{model_name} Results")

    if results.empty:
        st.warning("No results available for this selection.")

    else:

        # --------------------------------------------------
        # PERFORMANCE METRICS
        # --------------------------------------------------

        model_result = metrics[
            metrics["Model"].str.lower() == model_name.lower()
        ]

        if not model_result.empty:

            metric_row = model_result.iloc[0]

            st.markdown("### Model Performance")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "MAE",
                f"{metric_row['MAE']:.2f}"
            )

            c2.metric(
                "RMSE",
                f"{metric_row['RMSE']:.2f}"
            )

            c3.metric(
                "MAE Improvement vs Naive",
                str(metric_row["MAE vs naive"])
            )

        else:
            st.info("Summary metrics are not available for this model.")

        # --------------------------------------------------
        # ACTUAL VS PREDICTED GRAPH
        # --------------------------------------------------

        st.divider()
        st.markdown("### Actual vs Predicted Demand")

        if sku == "All Items":

            # Aggregate demand by week
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
            chart_data[[
                "Actual Demand",
                "Predicted Demand"
            ]],
            use_container_width=True
        )

        st.caption(
            "The chart compares recorded demand against "
            "the selected model's saved predictions."
        )

        # --------------------------------------------------
        # PREDICTION TABLE
        # --------------------------------------------------

        st.divider()
        st.markdown("### Prediction Results")

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

        st.dataframe(
            table[display_columns].sort_values("Week Ending"),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Prediction Results",
            data=table[display_columns].to_csv(index=False),
            file_name=f"{model_name.replace(' ', '_')}_results.csv",
            mime="text/csv"
        )

        # --------------------------------------------------
        # SHORTAGE ANALYSIS
        # --------------------------------------------------

        st.divider()
        st.markdown("### Demand Analysis")

        total_actual = results["demand_units"].sum()
        total_predicted = results[prediction_column].sum()

        a, b = st.columns(2)

        a.metric(
            "Total Actual Demand",
            f"{total_actual:,.0f}"
        )

        b.metric(
            "Total Predicted Demand",
            f"{total_predicted:,.0f}"
        )

        st.caption(
            "These figures summarise demand across the selected "
            "records. They are not inventory shortage quantities."
        )

else:

    st.info(
        "Select a forecasting model and filters above, "
        "then click Run Model to view its results."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "CSCI218 | Foundations of Artificial Intelligence | "
    "Humanitarian Warehouse Demand Forecasting"
)
