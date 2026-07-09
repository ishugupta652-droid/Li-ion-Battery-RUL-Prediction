# NASA Li-ion Battery Remaining Useful Life (RUL) Prediction

This repository contains a comprehensive Data Science pipeline developed for the **MSE643** coursework. The project focuses on analyzing degradation in NASA Lithium-ion batteries to predict their State-of-Health (SoH) and Remaining Useful Life (RUL) using advanced Machine Learning and Deep Learning techniques.

## Project Overview

Battery degradation is a highly complex, non-linear phenomenon. In critical engineering applications (e.g., aerospace and electric vehicles), predicting when a battery will fail is essential for safety. 

This project tackles battery degradation in two phases:
1. **Binary Classification (State):** Determining whether a battery is currently "Healthy" or "Degraded" based on its cycle number and ambient temperature.
2. **Time-Series Regression (RUL):** Forecasting the exact number of charge/discharge cycles remaining before the battery reaches its End-of-Life (EOL) threshold.

## Methodology

### 1. Data Engineering & Preprocessing
* Dynamically calculated the End-of-Life (EOL) threshold for every unique battery in the dataset (defined as 75% of its initial starting capacity).
* Engineered two target variables: `State` (1 for Healthy, 0 for Degraded) and `RUL` (countdown of remaining cycles).

### 2. Time-Series Forecasting (Deep Learning)
* Built a **Long Short-Term Memory (LSTM)** Neural Network.
* Reformatted chronological data into 10-cycle sliding windows, forcing the model to rely on historical capacity memory to predict future capacity fade.

### 3. Classification & Handling Imbalanced Data
* Evaluated 7 baseline Classical Machine Learning models (Logistic Regression, SVM, KNN, Random Forest, etc.).
* **The Problem:** Models achieved a deceptively high accuracy of 86% but suffered from a 0.00 Recall for degraded batteries due to an Imbalanced Dataset (86% of the dataset was healthy).
* **The Solution:** Implemented **SMOTE (Synthetic Minority Over-sampling Technique)** on a Random Forest classifier. This successfully boosted the Recall for degraded batteries from 10% to 77%, prioritizing safety (catching failing batteries) over raw accuracy.

### 4. Mathematical Optimization (GridSearchCV)
* Utilized `imblearn Pipeline` to safely cross-validate the SMOTE-enhanced Random Forest, mathematically proving its limit.
* Applied exhaustive GridSearch optimization to an **XGBoost Regressor**, tuning `learning_rate`, `max_depth`, and `n_estimators`.
* **Result:** Successfully reduced the final Root Mean Squared Error (RMSE) for the RUL countdown from **37.39** down to **30.29**.
* **Feature Importance:** Extracted XGBoost feature importance to prove that `Capacity` and `ambient_temperature` are the dominant indicators of impending failure when the model is heavily regularized.

## Repository Contents

* `Project_MSE643.ipynb`: The complete end-to-end Jupyter Notebook containing all data engineering, visualizations, model training, and evaluations.
* `metadata.csv`: The dataset containing the NASA battery cycle logs.
* `requirements.txt`: The required Python libraries to run the notebook.

## How to Run

1. Clone this repository.
2. Install the required dependencies using: `pip install -r requirements.txt`
3. Open `Project_MSE643.ipynb` in Jupyter Notebook or VSCode.
4. Execute "Restart Kernel and Run All Cells".
