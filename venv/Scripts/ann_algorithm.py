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
from sklearn.neural_network import MLPRegressor

# For automated machine learning using TPOT
import tpot
from tpot.old_config_utils import convert_config_dict_to_choicepipeline
from tpot import TPOTRegressor


if __name__ == "__main__":
    ###############################################
    # Data Loading and Initial Exploration
    ###############################################

    current_directory = os.getcwd()
    print("Current Working Directory:", current_directory)

    excel_file_name = "output.xlsx"
    excel_file_path = os.path.join(current_directory, excel_file_name)
    print("Excel File Path:", excel_file_path)

    df = pd.read_excel(excel_file_path)
    print(df.head())

    # Remove units from column names and "Concrete" from names
    input_variables = df.iloc[:, :9]
    input_variables.columns = [col.split(' (')[0].replace('Concrete ', '').lower() for col in input_variables.columns]

    # Plot Pearson correlation heatmap
    corr_matrix = input_variables.corr()
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='plasma', fmt=".2f", annot_kws={"size": 12})
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.show()
    print(corr_matrix)

    # Compute Kendall's tau correlation
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

    missing_values = df.isnull().sum()
    print("\nMissing Values:\n", missing_values)
    df = df.dropna()

    plt.figure(figsize=(10, 7))
    params_without_units = [param.split(' ')[0] for param in df.drop('CS\n(MPa)', axis=1).columns]
    sns.boxplot(data=df.drop('CS\n(MPa)', axis=1))
    plt.xticks(range(len(params_without_units)), params_without_units)
    plt.show()

    print("\nData Types:\n", df.dtypes)
    summary_stats = df.describe()
    print("\nSummary Statistics:\n", summary_stats)

    plt.figure(figsize=(8, 6))
    sns.histplot(df['CS\n(MPa)'], bins=30, kde=True)
    plt.xlabel('Compressive Strength')
    plt.ylabel('Frequency')
    plt.show()

    ###############################################
    # Step 2: Prepare the Data
    ###############################################

    X = df.drop('CS\n(MPa)', axis=1)
    y = df['CS\n(MPa)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ###############################################
    # Step 3: Automated Hyperparameter Tuning with TPOT
    ###############################################

    # Configure TPOT to search only over MLPRegressor parameters
    tpot_config = {
        'sklearn.neural_network.MLPRegressor': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [300, 500],
            'random_state': [42]
        }
    }

    # Convert config_dict to search_space
    search_space = convert_config_dict_to_choicepipeline(tpot_config)

    tpot = TPOTRegressor(search_space=search_space, generations=5, population_size=40, cv=5, random_state=42,
                        n_jobs=1, verbose=2)

    print("Starting TPOT AutoML tuning...")
    tpot.fit(X_train_scaled, y_train)
    print("TPOT tuning complete.")

    best_model = tpot.fitted_pipeline_
    print("Best Model from TPOT:")
    print(best_model)

    ###############################################
    # Step 4: Model Evaluation
    ###############################################

    y_pred = best_model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f'R-squared (R2): {r2}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Mean Absolute Error (MAE): {mae}')

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, scoring='r2', cv=kf)
    print(f"Cross-Validation R² Scores: {cv_scores}")
    print(f"Mean R² Score: {cv_scores.mean()}")
    print(f"Standard Deviation of R² Scores: {cv_scores.std()}")

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
    print(f'Predicted Compressive Strength: {user_pred_strength[0]}')

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
    # Step 8: SHAP Analysis
    ###############################################

    # For SHAP analysis, we refit an MLPRegressor (or use best_model if it is an MLPRegressor)
    # Note: For non-tree models, SHAP will use a KernelExplainer
    model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                         alpha=0.0001, learning_rate='adaptive', max_iter=300, random_state=42)
    model.fit(X_train_scaled, y_train)

    explainer = shap.Explainer(model, X_train_scaled)
    shap_values = explainer(X_test_scaled)

    # Assign feature names to the SHAP values
    shap_values.feature_names = input_variables.columns

    # SHAP summary plots
    shap.summary_plot(shap_values, X_test, feature_names=input_variables.columns, plot_type="bar")
    shap.summary_plot(shap_values, X_test, feature_names=input_variables.columns)
    shap.plots.bar(shap_values)

    plt.figure(figsize=(8, 6))
    explainer = shap.Explainer(best_model, X_train_scaled)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, feature_names=input_variables.columns)

    shap_importance = np.abs(shap_values).mean(axis=0)
    relative_importance = 100.0 * (shap_importance / shap_importance.max())
    sorted_idx = np.argsort(relative_importance)

    feature_short_forms = {
        "Cement": "cem",
        "Blast": "bfs",
        "Fly": "fa",
        "Water": "wtr",
        "Superplasticizer": "sp",
        "Coarse": "cag",
        "Fine": "fag",
        "Age": "age",
    }

    sorted_features = [feature_short_forms.get(feature.split(' ')[0], feature.split(' ')[0]) for feature in X.columns[sorted_idx]]
    sorted_relative_importance = relative_importance[sorted_idx]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    x_pos = np.arange(len(sorted_features)) * 1.5
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

