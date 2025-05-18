import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# For automated machine learning using TPOT
import tpot
from tpot.old_config_utils import convert_config_dict_to_choicepipeline
from tpot import TPOTRegressor

import shap

if __name__ == "__main__":
    ###############################################
    # Data Loading and Initial Exploration
    ###############################################
    current_directory = os.getcwd()
    print("Current Working Directory:", current_directory)

    # Use the output.xlsx file as the data source
    excel_file_name = "output.xlsx"
    excel_file_path = os.path.join(current_directory, excel_file_name)
    print("Excel File Path:", excel_file_path)

    df = pd.read_excel(excel_file_path)
    print("Data Preview:")
    print(df.head())

    # Remove units from column names and any extra text for consistency.
    # Here, we assume the first several columns (except target) are the features.
    # Adjust the slicing if needed.
    input_variables = df.iloc[:, :9]
    input_variables.columns = [col.split(' (')[0].replace('Concrete ', '').lower() for col in input_variables.columns]

    # Plot Pearson correlation heatmap for the input features
    plt.figure(figsize=(15, 12))
    corr_matrix = input_variables.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='plasma', fmt=".2f", annot_kws={"size": 12})
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.title("Pearson Correlation Heatmap")
    plt.show()
    print("Pearson Correlation Matrix:")
    print(corr_matrix)

    # Compute and plot Kendall's tau correlation heatmap
    # kendall_corr_matrix = input_variables.corr(method='kendall')
    # plt.figure(figsize=(15, 12))
    # sns.heatmap(kendall_corr_matrix, annot=True, cmap='cividis', fmt=".2f", annot_kws={"size": 16}, cbar_kws={"shrink": 0.8})
    # plt.xticks(rotation=90, ha='right', fontsize=14)
    # plt.yticks(rotation=0, fontsize=14)
    # plt.title("Kendall's Tau Correlation Heatmap")
    # plt.show()
    # print("Kendall's Tau Correlation Matrix:")
    # print(kendall_corr_matrix)

    ###############################################
    # Step 1: Explore and Clean the Data
    ###############################################
    # Check for missing values and drop any rows with missing data
    missing_values = df.isnull().sum()
    print("\nMissing Values:\n", missing_values)
    df = df.dropna()

    # Visualize outliers using boxplots (excluding the target column)
    # Adjust target column name as needed (here assumed as "Concrete compressive strength (MPa, megapascals) ")
    target_col = 'CS\n(MPa)'
    plt.figure(figsize=(10, 7))
    sns.boxplot(data=df.drop(target_col, axis=1))
    plt.title('Boxplot of Features')
    plt.show()

    print("\nData Types:")
    print(df.dtypes)
    print("\nSummary Statistics:")
    print(df.describe())

    # Plot distribution of the target variable
    plt.figure(figsize=(8, 6))
    sns.histplot(df[target_col], bins=30, kde=True)
    plt.xlabel('Compressive Strength')
    plt.ylabel('Frequency')
    plt.title('Target Distribution')
    plt.show()

    ###############################################
    # Step 2: Prepare the Data
    ###############################################
    # Separate features and target variable
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Split the data into training and testing sets (10% test size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Feature scaling using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ###############################################
    # Step 3: Automated Hyperparameter Tuning with TPOT (SVR)
    ###############################################
    # Configure TPOT to search only over SVR parameters.
    # You can extend this config with additional SVR hyperparameters if needed.
    tpot_config = {
        'sklearn.svm.SVR': {
            'kernel': ['linear', 'rbf'],
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.2],
            'gamma': ['scale', 'auto'],  # Important for rbf
            'cache_size': [200],
        }
    }

    # Convert config_dict to search_space
    search_space = convert_config_dict_to_choicepipeline(tpot_config)

    tpot = TPOTRegressor(search_space=search_space, generations=5, population_size=40, cv=5, random_state=42,
                        n_jobs=1, verbose=2)

    print("Starting TPOT AutoML tuning for SVR...")
    tpot.fit(X_train_scaled, y_train)
    print("TPOT tuning complete.")

    best_model = tpot.fitted_pipeline_
    print("Best SVR Model from TPOT:")
    print(best_model)

    ###############################################
    # Step 4: Model Evaluation
    ###############################################
    # Predict on the test set using the best model from TPOT
    y_pred = best_model.predict(X_test_scaled)

    # Calculate evaluation metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f'\nEvaluation Metrics:')
    print(f'R-squared (R²): {r2}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Mean Absolute Error (MAE): {mae}')

    # Cross-validation results using KFold
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, scoring='r2', cv=kf)
    print(f"Cross-Validation R² Scores: {cv_scores}")
    print(f"Mean R² Score: {cv_scores.mean()}")
    print(f"Standard Deviation of R² Scores: {cv_scores.std()}")

    # Visualize cross-validation results
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, n_splits + 1), cv_scores, marker='o', label='R² Score per Fold', color='blue')
    plt.axhline(cv_scores.mean(), color='red', linestyle='--', label='Mean R² Score')
    plt.fill_between(
        range(1, n_splits + 1),
        cv_scores.mean() - cv_scores.std(),
        cv_scores.mean() + cv_scores.std(),
        color='red',
        alpha=0.2,
        label='Standard Deviation Range'
    )
    plt.title('Cross-Validation R² Scores', fontsize=15)
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid()
    plt.show()

    ###############################################
    # Step 5: User Input for Real-Time Prediction
    ###############################################
    feature_names = X.columns.tolist()
    user_input = {}
    for feature in feature_names:
        user_input[feature] = float(input(f'Enter value for {feature}: '))

    user_df = pd.DataFrame([user_input])
    user_input_scaled = scaler.transform(user_df)
    user_pred_strength = best_model.predict(user_input_scaled)
    print(f'\nPredicted Compressive Strength: {user_pred_strength[0]}')

    ###############################################
    # Step 6: Measure Training Time (Refit on Training Data)
    ###############################################
    start_time = time.time()
    best_model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    print(f"Training Time: {training_time:.2f} seconds")

    ###############################################
    # Step 7: Visualization: Actual vs. Predicted
    ###############################################
    # Scatter plot for test set with fitted regression line
    # plt.figure(figsize=(10, 6))
    # plt.scatter(y_test, y_pred, color='green', label='Test Set', alpha=0.7)
    # plt.xlabel('Actual Compressive Strength')
    # plt.ylabel('Predicted Compressive Strength')

    # slope, intercept, _, _, _ = linregress(y_test, y_pred)
    # fit_line = slope * y_test + intercept
    # plt.plot(y_test, fit_line, '--', color='red', linewidth=2, label='Fitted Line')
    # equation_text = f'Fitted Equation: y = {slope:.2f}x + {intercept:.2f}'
    # plt.text(0.5, 0.92, equation_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    # r2_text = f'R-squared (R²): {r2:.3f}'
    # plt.text(0.5, 0.85, r2_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    # plt.legend()
    # plt.show()

    # Plotting actual vs. predicted values by sample number
    sample_indices = np.arange(len(y_test))
    sorted_indices = np.argsort(sample_indices)
    sorted_y_test = y_test.values[sorted_indices]
    sorted_y_pred = y_pred[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.scatter(sorted_indices, sorted_y_test, color='red', label='Actual', alpha=0.7)
    plt.scatter(sorted_indices, sorted_y_pred, color='green', label='Predicted', alpha=0.7)
    plt.plot(sorted_indices, sorted_y_test, color='red', linestyle='-', linewidth=1)
    plt.plot(sorted_indices, sorted_y_pred, color='green', linestyle='-', linewidth=1)
    plt.xlabel('Sample Number')
    plt.ylabel('Compressive Strength')
    plt.legend()
    plt.show()

    # # Bar chart comparing actual vs. predicted for a random sample of 10
    # sample_size = min(10, len(y_test))
    # random.seed(100)
    # sample_indices = random.sample(range(len(y_test)), sample_size)
    # actual_strength = y_test.values[sample_indices]
    # predicted_strength = y_pred[sample_indices]

    # bar_width = 0.35
    # plt.figure(figsize=(14, 8))
    # plt.bar(range(sample_size), actual_strength, color='blue', width=bar_width, label='Actual')
    # plt.bar([i + bar_width for i in range(sample_size)], predicted_strength, color='orange', width=bar_width, label='Predicted')
    # plt.xlabel('Sample Number', fontsize=15)
    # plt.ylabel('Compressive Strength', fontsize=15)
    # plt.xticks([i + bar_width / 2 for i in range(sample_size)], sample_indices)
    # plt.legend(fontsize=12)
    # plt.tight_layout()
    # plt.show()

    ###############################################
    # (Optional) SHAP Analysis
    ###############################################
    # To interpret the SVR model's predictions, you can use SHAP.
    # Note: SVR is not tree-based; thus, you may need to use KernelExplainer.
    explainer = shap.KernelExplainer(best_model.predict, X_train_scaled[:100])
    shap_values = explainer.shap_values(X_test_scaled[:50], nsamples=100)

    # Summary plot for feature importance
    shap.summary_plot(shap_values, X_test, feature_names=X.columns.tolist())
