# Humanitarian Warehouse Demand Forecasting System

A machine learning-based demand forecasting system designed to support humanitarian warehouse planning. The project uses historical data provided by Dubai Humanitarian to predict weekly demand for relief items, compare forecasting models and identify potential stock shortages through an interactive Streamlit dashboard.

---

## 1. Project Overview

Humanitarian warehouses must maintain sufficient supplies to respond to emergencies and support affected communities. However, estimating future demand is challenging. Underestimating demand can result in shortages of essential supplies, while overestimating demand can lead to excess inventory and inefficient resource allocation.

The Humanitarian Warehouse Demand Forecasting System applies machine learning to historical warehouse data to predict weekly demand for individual relief items. It also estimates projected stock levels and highlights items that may require attention.

The project focuses on four humanitarian supply categories:

- Food & Nutrition
- Shelter & NFI
- Medical
- WASH (Water, Sanitation and Hygiene)

The system combines data preprocessing, feature engineering, forecasting model comparison, shortage detection and an interactive Streamlit dashboard.

**Data source:** Historical datasets provided by Dubai Humanitarian.

## 2. Objectives

The main objectives are to:

- Analyse historical humanitarian dispatch and inventory data.
- Clean and transform the data into a suitable format for machine learning.
- Identify historical weekly demand patterns across relief items.
- Forecast future weekly demand for individual items.
- Compare baseline forecasting methods and machine learning models.
- Evaluate forecasting performance using consistent metrics.
- Estimate projected inventory levels and identify potential shortages.
- Present forecasts and shortage alerts through an interactive dashboard.

## 3. System Features

| Feature | Description |
|---|---|
| Data preprocessing | Cleans and prepares dispatch, inventory, item, calendar and emergency-related data. |
| Feature engineering | Creates historical demand, lag, rolling and contextual features for forecasting. |
| Weekly demand forecasting | Predicts future demand for individual humanitarian relief items. |
| Model comparison | Evaluates six forecasting approaches using a chronological test dataset. |
| Model evaluation | Compares predictions using MAE and RMSE. |
| Shortage detection | Estimates projected stock using predicted demand, starting inventory and incoming stock. |
| Risk classification | Groups potential stock shortages into CRITICAL, HIGH, WARNING and OK levels. |
| Interactive dashboard | Displays forecasts, model comparisons and shortage alerts using Streamlit and Plotly. |

## 4. System Workflow

The system follows the workflow below:

1. **Data input:** Load the historical dispatch, inventory, item, calendar and emergency-related datasets.
2. **Data preprocessing:** Clean records, handle inconsistencies and prepare the datasets for analysis.
3. **Feature engineering:** Generate weekly demand observations and forecasting features, including lagged demand and rolling averages.
4. **Model training:** Train and evaluate baseline and machine learning forecasting models.
5. **Forecast evaluation:** Compare predictions against actual demand using MAE and RMSE.
6. **Shortage detection:** Calculate projected stock and classify potential inventory risks.
7. **Dashboard visualisation:** Present forecasts, model performance and shortage alerts through the Streamlit application.

## 5. Dataset

The project uses historical datasets provided by Dubai Humanitarian.

### Data sources

| Dataset | Purpose |
|---|---|
| Dispatch records | Provides historical records of dispatched humanitarian supplies. |
| Item details | Provides information about individual relief items and their categories. |
| Weekly inventory snapshots | Provides historical warehouse inventory information. |
| Calendar information | Provides time-related features for forecasting. |
| Emergency-related information | Provides contextual information about emergencies that may influence demand. |

### Dataset summary

| Description | Records |
|---|---:|
| Original dispatch records | 92,678 |
| Records after cleaning | 92,653 |
| Final processed dataset | 2,114 rows |
| Number of features in processed dataset | 49 |
| Number of relief items | 14 |
| Number of supply categories | 4 |

The final processed dataset contains weekly observations for individual relief items.

### Forecasting target

The target variable is **weekly demand**, measured in units. Historical dispatch records are used to derive demand because inventory snapshots alone may not reflect actual demand during stockout periods.

### Training and testing

The dataset is divided chronologically:

- **Training set:** First 80% of observations (1,680 rows).
- **Testing set:** Final 20% of observations (434 rows).

A chronological split is used rather than a random split to preserve the time order of the observations. This allows the models to be evaluated on later observations that were not used during training.

Only information available at the relevant prediction point should be used to generate forecasting features.

**Data availability:** The current implementation uses historical datasets and is not connected to a live warehouse management system.

## 6. Data Preprocessing and Feature Engineering

Data preprocessing prepares the historical records for forecasting.

The process includes:

- Loading and combining the required historical datasets.
- Cleaning duplicate and inconsistent records.
- Handling missing values where required.
- Converting and preparing relevant data types.
- Organising demand observations by item and week.
- Preparing categorical and numerical features for model training.

### Feature engineering

The forecasting dataset includes features designed to capture historical demand patterns and contextual influences.

| Feature | Purpose |
|---|---|
| Lag 1 | Captures demand from the previous week. |
| Lag 2 | Captures demand from two weeks earlier. |
| Lag 4 | Captures demand from four weeks earlier. |
| Lag 52 | Captures demand from the corresponding period one year earlier, where available. |
| Rolling mean | Represents average demand over a previous period. |
| Calendar features | Capture time-related patterns. |
| Emergency features | Represent emergency-related contextual information. |
| Item and category features | Help models distinguish between different relief items and supply categories. |

Lagged and rolling features should use historical observations only, preventing information from the prediction target period from entering the model inputs.

## 7. Forecasting Models

The project compares six forecasting approaches.

| Model | Description | Purpose |
|---|---|---|
| Naive Forecast | Uses the previous week's demand as the next week's prediction. | Provides a simple baseline. |
| Four-Week Moving Average | Uses the average demand from the previous four weeks. | Provides a smoothed baseline. |
| Linear Regression | Models linear relationships between input features and demand. | Provides an interpretable machine learning baseline. |
| Decision Tree | Uses decision-based splits to capture non-linear relationships. | Identifies non-linear patterns and feature interactions. |
| Random Forest | Combines predictions from multiple decision trees. | Improves prediction stability through an ensemble approach. |
| Dense Neural Network | Learns relationships between input features through interconnected neural network layers. | Evaluates a neural network approach to demand forecasting. |

### Model configurations

The following configurations are used in the reported experiments:

| Model | Configuration |
|---|---|
| Decision Tree | `max_depth=8`, `min_samples_leaf=5` |
| Random Forest | `n_estimators=300`, `min_samples_leaf=3` |
| Random Forest repeated runs | Random seeds 0–4 |

The Naive and Four-Week Moving Average models serve as reference baselines. The remaining models learn relationships between the engineered input features and weekly demand.

**Additional model:** Gradient Boosting is used separately for shortage-alert walk-forward backtesting. It is not included in the primary six-model forecasting comparison.

## 8. Model Evaluation

Forecasting performance is evaluated using a chronological test dataset.

### Evaluation metrics

**Mean Absolute Error (MAE)**

Measures the average absolute difference between actual and predicted demand.


A lower MAE indicates that predictions are closer to actual demand on average.

**Root Mean Squared Error (RMSE)**

Measures prediction error while giving greater weight to larger errors.

A lower RMSE indicates better predictive performance, particularly when large errors are important.

### Model comparison

All six primary forecasting approaches are evaluated using the same chronological test observations and metrics to support a fair comparison.

Random Forest is also evaluated across five random seeds (0–4) to examine variation between repeated runs.

### Shortage-alert evaluation

Shortage-alert behaviour is evaluated separately using walk-forward backtesting over historical observations.

This approach evaluates shortage detection across successive historical periods rather than relying only on the primary forecasting test period.

Precision and recall are used to assess alert performance where actual shortage events are available.

## 9. Shortage Detection

The system uses predicted demand and inventory information to estimate whether a relief item may experience a shortage.

### Projected stock calculation


The projected stock level is considered alongside demand coverage to classify potential shortage risks.

### Risk classifications

| Risk level | Description |
|---|---|
| CRITICAL | Indicates a high-priority potential shortage. |
| HIGH | Indicates a significant shortage risk. |
| WARNING | Indicates that inventory may require attention. |
| OK | Indicates that projected inventory is sufficient under the current calculation. |

These classifications are intended to support warehouse planning and prioritisation. They do not automatically trigger stock purchases or warehouse decisions.

## 10. Dashboard

The project includes an interactive dashboard developed using Streamlit and Plotly.

The dashboard presents forecasting and inventory-related information in a visual format.

Its main purpose is to help users:

- Explore weekly demand forecasts.
- Compare actual and predicted demand.
- Review forecasting model performance.
- Identify relief items with potential shortage risks.
- View projected stock and shortage classifications.

To launch the dashboard, follow the installation instructions below.

## 11. Technologies and Dependencies

| Technology | Purpose |
|---|---|
| Python | Main programming language. |
| Pandas | Data manipulation and preprocessing. |
| NumPy | Numerical operations. |
| Scikit-learn | Machine learning models and evaluation metrics. |
| TensorFlow / Keras | Dense Neural Network implementation. |
| Matplotlib | Data visualisation. |
| Plotly | Interactive charts and visualisations. |
| Streamlit | Interactive dashboard development. |
| Jupyter Notebook / Google Colab | Model development and experimentation. |
| Git and GitHub | Version control and team collaboration. |

The project's Python package dependencies are maintained in `requirements.txt`.

For reproducible installation, use the dependency versions specified in that file rather than installing unpinned latest versions.

## 12. Repository Structure

The main application and dependency file are located in the project root.

```text
Humanitarian-Warehouse-Demand-Forecasting/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── clean_weekly_demand.csv
│
├── notebooks/
│   └── Model development and experimentation notebooks
│
├── outputs/
│   └── Generated predictions and evaluation results

```

The directory names and notebook/output contents should match the files committed to the repository.

### Main project components

| Component | Responsibility |
|---|---|
| `app.py` | Runs the Streamlit dashboard. |
| `requirements.txt` | Lists the Python dependencies required to run the project. |
| `data/` | Contains the processed forecasting dataset, where included. |
| `notebooks/` | Contains the model development, preprocessing and evaluation work. |
| `outputs/` | Stores generated predictions and evaluation results, where included. |
| `tests/` | Contains automated software tests, if implemented. |

## 13. Installation and Setup

### Prerequisites

- Python 3.x (use the version tested with the project).
- pip.
- Git.
- Access to the required dataset and generated output files.

### Step 1: Clone the repository

```bash
git clone https://github.com/HuzSuddle/Humanitarian-Warehouse-Demand-Forecasting.git
```

### Step 2: Navigate to the project directory

```bash
cd Humanitarian-Warehouse-Demand-Forecasting
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### Step 4: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Run the dashboard

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that address in a browser to access the dashboard.

**Note:** Ensure that the required input datasets and generated output files are available in the expected repository locations before launching the application.

## 14. Configuration and Reproducibility

Reproducibility is important when comparing forecasting models and evaluating their performance.

The following experimental settings should be maintained consistently:

| Setting | Value |
|---|---|
| Dataset split | Chronological 80:20 |
| Training observations | 1,680 |
| Testing observations | 434 |
| Forecasting metrics | MAE and RMSE |
| Decision Tree maximum depth | 8 |
| Decision Tree minimum samples per leaf | 5 |
| Random Forest estimators | 300 |
| Random Forest minimum samples per leaf | 3 |
| Random Forest random seeds | 0, 1, 2, 3, 4 |

### Random seeds

Random seeds are used for repeated Random Forest experiments to make the model runs reproducible.

Where other stochastic models are used, their random states should also be explicitly controlled and documented.

### Dependency management

The `requirements.txt` file records the packages required by the project. Using the same Python and package versions helps reduce differences between development and execution environments.

### Reproducing the experiment

To reproduce the project:

1. Clone the repository.
2. Set up the Python environment.
3. Install the listed dependencies.
4. Obtain the required datasets.
5. Run the model development and evaluation notebooks in the documented order.
6. Generate the required prediction and evaluation outputs.
7. Launch the Streamlit dashboard.

The exact Python version, dependency versions, model settings and random seeds should be retained with the project files.

## 15. Testing and Validation

Testing is important to ensure that the application handles data and forecasting outputs correctly.

### Forecasting validation

The forecasting models are evaluated using a chronological test dataset and MAE and RMSE.

Shortage-alert behaviour is assessed separately using walk-forward backtesting.

### Recommended software test coverage

The following cases should be covered by automated tests where applicable:

| Test area | Expected behaviour |
|---|---|
| Missing input files | Display a clear error rather than failing unexpectedly. |
| Missing values | Handle or report missing values appropriately. |
| Duplicate records | Apply the intended duplicate-handling process. |
| Invalid data types | Detect or handle invalid values before model processing. |
| Forecast output | Ensure predictions are numeric and valid. |
| Projected stock | Verify the projected stock calculation. |
| Risk classification | Verify that projected stock and demand coverage produce the expected risk category. |
| Dashboard output files | Display a useful message if required prediction or metric files are unavailable. |

### Running automated tests

If automated tests are included in the repository, run them using the project's configured test framework.

For example, if the project uses `pytest`:

```bash
pip install pytest
pytest -v
```

**Important:** This command is applicable only after the test suite has been implemented. Model evaluation on a test dataset is not the same as automated software testing.

## 16. Limitations

- Forecasting performance depends on the quality and availability of historical data.
- Unexpected emergencies may cause demand patterns to change.
- Historical demand patterns may not fully represent future humanitarian requirements.
- Forecast accuracy may vary between relief items and supply categories.
- The system relies on the availability of inventory and incoming-stock information for shortage estimation.
- The current implementation uses historical datasets rather than live warehouse system integration.
- The system provides decision support and does not automatically make inventory or procurement decisions.

## 17. Future Improvements

Potential future improvements include:

- Integrating live warehouse inventory and dispatch data.
- Extending the system to support additional relief items and categories.
- Improving shortage detection using additional historical and contextual information.
- Expanding automated testing and validation coverage.
- Introducing centralised configuration management for model parameters and experiment settings.
- Improving dashboard usability and reporting capabilities.
- Exploring additional forecasting models and ensemble approaches.

---

## Project Outcome

The Humanitarian Warehouse Demand Forecasting System demonstrates how machine learning can support humanitarian warehouse planning through weekly demand prediction and early identification of potential stock shortages.

By combining historical data processing, forecasting models, model evaluation and an interactive dashboard, the project provides a decision-support approach to help warehouse teams make more informed inventory planning decisions.
