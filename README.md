# Humanitarian Warehouse Demand Forecasting System

## Overview

This project uses machine learning to predict weekly demand for humanitarian relief supplies. It aims to help warehouse staff plan inventory, identify potential shortages and make better stock management decisions.

The system uses historical data provided by Dubai Humanitarian to forecast demand across four supply categories: Food & Nutrition, Shelter & NFI, Medical, and WASH.

## Features

- Data cleaning and preprocessing.
- Weekly demand forecasting for individual relief items.
- Comparison of different machine learning models.
- Model evaluation using MAE and RMSE.
- Shortage detection based on predicted demand and available inventory.
- Interactive Streamlit dashboard for viewing forecasts and shortage alerts.

## Machine Learning Models

The project uses the following forecasting models:

- Naive Forecast
- 4-Week Moving Average
- Linear Regression
- Decision Tree
- Random Forest
- Dense Neural Network

Gradient Boosting is also used as an additional model for shortage-alert backtesting.

## Shortage Detection

The system compares predicted demand with available inventory and incoming stock to estimate projected stock levels.

**Projected Stock = Starting Stock + Incoming Stock − Predicted Demand**

Shortage risks are classified into four levels:

- **CRITICAL**
- **HIGH**
- **WARNING**
- **OK**

## Dataset

The dataset includes:

- Historical dispatch records
- Item details
- Weekly inventory snapshots
- Calendar information
- Emergency-related information

The final processed dataset contains 2,114 rows and 49 columns, covering 14 relief items across four supply categories.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Plotly
- Streamlit
- Google Colab / Jupyter Notebook

## Running the Dashboard

1. Clone the repository:

   ```bash
   git clone https://github.com/HuzSuddle/Humanitarian-Warehouse-Demand-Forecasting.git
   ```

2. Open the project folder:

   ```bash
   cd Humanitarian-Warehouse-Demand-Forecasting
   ```

3. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit dashboard:

   ```bash
   streamlit run app.py
   ```


## Project Objective

The main objective is to use machine learning and data analysis to support humanitarian warehouse planning by predicting future demand and identifying potential stock shortages.

This project was developed as part of a university group project.
