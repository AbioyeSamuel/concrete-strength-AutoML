#  Predictive ML Algorithms for Concrete Strength

This repository contains machine learning models used to predict **real-time concrete compressive strength** based on material composition and curing age. It includes both classic regressors and automated machine learning (AutoML) tools such as TPOT, integrated with powerful algorithms like CatBoost, XGBoost, SVR, MLP, ANN, KNN, GBR, and Lasso Regression.

---

## Project Structure

concrete-strength-predictor/
│
├── gbr_algorithm.py # Gradient Boosting Regressor script
├── xgboost_model.py # XGBoost implementation
├── catboost_model.py # CatBoost implementation
├── svr_model.py # Support Vector Regression
├── ann_model.py # Artificial Neural Network
├── knn_model.py # K-Nearest Neighbors
├── lasso_model.py # Lasso Regression
├── tpot_automl.py # TPOT AutoML integration
├── utils.py # Data preprocessing and shared utilities
├── dataset/ # Input dataset
├── visualizations/ # Charts, heatmaps, and result plots
└── README.md # Project documentation

##  Features

- Predicts concrete compressive strength using multiple ML algorithms
- Includes **AutoML with TPOT**
- Supports **feature engineering, data normalization**, and **correlation heatmaps**
- Produces visualizations like performance plots and 3D bar charts
- Models benchmarked for performance (R² > 99% on best models)

---

##  Setup Guide

Follow the steps below to set up the project on your local machine.

### 1. Install an IDE

Download and install **Visual Studio Code**:  

https://code.visualstudio.com/

### 2. Install Python

Download and install **Python** (version 3.8 or later):  

https://www.python.org/downloads/  
Ensure you **check the box** to add Python to PATH during installation.

### 3. Clone the Repository

In your terminal:

git clone https://github.com/AbioyeSamuel/concrete-strength-AutoML.git
cd concrete-strength-predictor

### 4. Create a Virtual Environment

python -m venv venv

### 5. Activate the Environment
Windows:

.\venv\Scripts\activate
Mac/Linux:

source venv/bin/activate

### 6. Install Required Packages

pip install -r requirements.txt

If requirements.txt is not available, install manually:

pip install numpy pandas scikit-learn matplotlib seaborn xgboost catboost tpot

To run any model, use:

python gbr_algorithm.py         # For Gradient Boosting
python xgboost_model.py         # For XGBoost
python catboost_model.py        # For CatBoost
python svr_model.py             # For SVR
python ann_model.py             # For ANN
python knn_model.py             # For KNN
python lasso_model.py           # For Lasso
python tpot_automl.py           # For AutoML using TPOT

### Visualization Outputs
Heatmaps for correlation analysis

3D bar plots for performance comparison

Prediction vs Actual plots for all models

Accuracy results stored and compared automatically