import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from mpl_toolkits.mplot3d import Axes3D
import shap

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# For automated machine learning using TPOT
import tpot
print(tpot.__version__)  # Should print 1.0.0
from tpot import TPOTRegressor
from tpot.old_config_utils import convert_config_dict_to_choicepipeline
import inspect
print(type(TPOTRegressor()))
print(inspect.getfile(tpot.TPOTRegressor))
print(inspect.signature(TPOTRegressor.__init__))

# Also import XGBRegressor so that TPOT uses the correct estimator for SHAP analysis later.
from xgboost import XGBRegressor


###############################################
# Data Loading and Initial Exploration
###############################################

if __name__ == "__main__":
    # Get the current working directory
    current_directory = os.getcwd()
    print("Current Working Directory:", current_directory)
    print(f"Script is running in: {os.getcwd()}")
    
    # Construct the relative path to the Excel file
    excel_file_name = "output.xlsx"
    excel_file_path = os.path.join(current_directory, excel_file_name)
    print("Excel File Path:", excel_file_path)
    
    # Load the Excel file into a DataFrame
    df = pd.read_excel(excel_file_path)
    print(df.head())
    
    # Remove units from column names and "Concrete" from the column name "Concrete compressive strength"
    # (Assuming the first 9 columns are the features; adjust if needed)
    input_variables = df.iloc[:, :9]
    input_variables.columns = [col.split(' (')[0].replace('Concrete ', '').lower() for col in input_variables.columns]
    
    # Plot correlation heatmap (Pearson)
    corr_matrix = input_variables.corr()
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='plasma', fmt=".2f", annot_kws={"size": 12})
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.show()
    print(corr_matrix)
    
    # Compute Kendall's tau correlation for each pair of features
    # kendall_corr_matrix = input_variables.corr(method='kendall')
    # plt.figure(figsize=(15, 12))
    # sns.heatmap(kendall_corr_matrix, annot=True, cmap='cividis', fmt=".2f", annot_kws={"size": 16}, cbar_kws={"shrink": 0.8})
    # plt.xticks(rotation=90, ha='right', fontsize=14)
    # plt.yticks(rotation=0, fontsize=14)
    # plt.show()
    # print("Kendall's Tau Correlation Matrix:\n", kendall_corr_matrix)
    
    ###############################################
    # Step 1: Explore and Clean the Data
    ###############################################
    
    # Check for missing values and drop rows with missing values
    missing_values = df.isnull().sum()
    print("\nMissing Values:\n", missing_values)
    df = df.dropna()
    
    # Visualize outliers using boxplots for features (excluding the target)
    plt.figure(figsize=(10, 7))
    # Removing unit-laden parameter names for cleaner boxplot labels
    params_without_units = [param.split(' ')[0] for param in df.drop('CS\n(MPa)', axis=1).columns]
    sns.boxplot(data=df.drop('CS\n(MPa)', axis=1))
    plt.xticks(range(len(params_without_units)), params_without_units)
    plt.show()
    
    # Check data types and summary statistics
    print("\nData Types:\n", df.dtypes)
    summary_stats = df.describe()
    print("\nSummary Statistics:\n", summary_stats)
    
    # Distribution of the target variable (Compressive Strength)
    plt.figure(figsize=(8, 6))
    sns.histplot(df['CS\n(MPa)'], bins=30, kde=True)
    plt.xlabel('Compressive Strength')
    plt.ylabel('Frequency')
    plt.show()
    
    ###############################################
    # Step 2: Prepare the Data
    ###############################################
    
    # Separate features and target variable
    X = df.drop('CS\n(MPa)', axis=1)
    y = df['CS\n(MPa)']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Step 3: Feature Scaling using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ###############################################
    # Step 4: Automated Hyperparameter Tuning with TPOT
    ###############################################
    
    # TPOTRegressor configured to only consider XGBRegressor with specified hyperparameters.
    tpot_config = {
        'xgboost.XGBRegressor': {
            'n_estimators': [100, 150, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'objective': ['reg:squarederror'],
            'random_state': [42]
        }
    }
    # Convert config_dict to search_space
    search_space = convert_config_dict_to_choicepipeline(tpot_config)

    # Initialize TPOTRegressor; you can adjust generations and population_size as desired
    tpot = TPOTRegressor(search_space=search_space, generations=5, population_size=40, cv=5, random_state=42,
                        n_jobs=1, verbose=2)
    
    # Fit the TPOT pipeline on the scaled training data
    print("Starting TPOT AutoML tuning...")
    tpot.fit(X_train_scaled, y_train)
    print("TPOT tuning complete.")
    
    # Get the best pipeline (estimator)
    best_model = tpot.fitted_pipeline_
    print("Best Model from TPOT:")
    print(best_model)
    
    # Optional: Export the TPOT pipeline code (if you wish to keep a standalone pipeline)
    # tpot.export('tpot_best_pipeline.py')
    
    ###############################################
    # Step 5: Model Evaluation
    ###############################################
    
    # Predict on the test set using the best model
    y_pred = best_model.predict(X_test_scaled)
    
    # Calculate evaluation metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f'R-squared (R2): {r2}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    
    # Perform cross-validation on the best model using KFold
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, scoring='r2', cv=kf)
    print(f"Cross-Validation R² Scores: {cv_scores}")
    print(f"Mean R² Score: {cv_scores.mean()}")
    print(f"Standard Deviation of R² Scores: {cv_scores.std()}")
    
    # Plot cross-validation R² scores
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
    # Step 6: User Input for Real-Time Prediction
    ###############################################
    
    # Extract feature names from the DataFrame
    feature_names = X.columns.tolist()
    
    # Take user input for all features (this example uses command-line input)
    user_input = {}
    for feature in feature_names:
        user_input[feature] = float(input(f'Enter value for {feature}: '))
    
    user_df = pd.DataFrame([user_input])
    user_input_scaled = scaler.transform(user_df)
    user_pred_strength = best_model.predict(user_input_scaled)
    print(f'Predicted Compressive Strength: {user_pred_strength[0]}')
    
    ###############################################
    # Step 7: Measure Training Time (Refit on Training Data)
    ###############################################
    
    start_time = time.time()
    best_model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    print(f"Training Time: {training_time:.2f} seconds")
    
    ###############################################
    # Step 8: Visualization: Actual vs. Predicted
    ###############################################
    
    # Scatter plot for Test Set with fitted line
    # plt.figure(figsize=(10, 6))
    # plt.scatter(y_test, y_pred, color='green', label='Test Set', alpha=0.7)
    # plt.xlabel('Actual Compressive Strength')
    # plt.ylabel('Predicted Compressive Strength')
    
    # slope, intercept, _, _, _ = linregress(y_test, y_pred)
    # fit_line = slope * y_test + intercept
    # plt.plot(y_test, fit_line, '--', color='red', linewidth=2, label='Fitted Line')
    
    # equation_text = f'Fitted Line: y = {slope:.2f}x + {intercept:.2f}'
    # plt.text(0.5, 0.92, equation_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    # r2_text = f'R-squared (R²): {r2:.3f}'
    # plt.text(0.5, 0.85, r2_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    
    # plt.legend()
    # plt.show()
    
    # Scatter plot by sample number
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
    
    # Bar chart comparing actual vs. predicted for a random sample of 10
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
    # Step 9: SHAP Analysis
    ###############################################
    
    # For SHAP analysis, we refit an XGBRegressor using the best parameters found by TPOT
    # (Alternatively, you can use best_model if it is based on XGBRegressor)
    # Here we train a new model for clarity.
    model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, subsample=0.8, random_state=0)
    model.fit(X_train_scaled, y_train)
    
    # Initialize SHAP explainer and compute SHAP values for the test set
    explainer = shap.Explainer(model, X_train_scaled)
    shap_values = explainer(X_test_scaled)
    
    # Assign actual feature names to the SHAP values object
    shap_values.feature_names = input_variables.columns
    
    # Summary plot (bar chart of feature importance)
    shap.summary_plot(shap_values, X_test, feature_names=input_variables.columns, plot_type="bar")
    
    # Detailed summary plot
    shap.summary_plot(shap_values, X_test, feature_names=input_variables.columns)
    
    # Bar plot of feature importance
    shap.plots.bar(shap_values)
    
    # SHAP summary plot on the entire dataset (for cross-validation insights)
    plt.figure(figsize=(8, 6))
    # This works only if the final step in the pipeline is indeed an XGBRegressor
    explainer = shap.TreeExplainer(best_model.named_steps['xgbregressor'])
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, feature_names=input_variables.columns)
    
    # Relative Importance of Features using SHAP values
    shap_importance = np.abs(shap_values).mean(axis=0)
    relative_importance = 100.0 * (shap_importance / shap_importance.max())
    sorted_idx = np.argsort(relative_importance)
    
    # Short forms mapping for feature names
    feature_short_forms = {
        "Cement": "cem",
        "Blast": "bfs",  # For "Blast furnace slag"
        "Fly": "fa",     # For "Fly ash"
        "Water": "wtr",
        "Superplasticizer": "sp",
        "Coarse": "cag",   # For "Coarse aggregate"
        "Fine": "fag",     # For "Fine aggregate"
        "Age": "age",
    }
    
    sorted_features = [feature_short_forms.get(feature.split(' ')[0], feature.split(' ')[0]) for feature in X.columns[sorted_idx]]
    sorted_relative_importance = relative_importance[sorted_idx]
    
    # 3D Plot of Feature Importance
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    x_pos = np.arange(len(sorted_features)) * 1.5  # spacing between bars
    y_pos = np.zeros(len(sorted_features))
    z_pos = np.zeros(len(sorted_features))
    bar_width = 0.4
    bar_depth = 0.3
    bar_height = sorted_relative_importance
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(sorted_features)))
    ax.bar3d(x_pos, y_pos, z_pos, bar_width, bar_depth, bar_height, color=colors, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_features, rotation=45, ha='right', fontsize=10)
    ax.set_yticks([])
    ax.set_xlabel('Features', fontsize=12, labelpad=30)
    ax.set_zlabel('Relative Importance (%)', fontsize=12, labelpad=10)
    
    for i in range(len(sorted_relative_importance)):
        ax.text(x_pos[i], y_pos[i], bar_height[i] + 2, f"{bar_height[i]:.2f}%", color='skyblue', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.show()