# import pandas as pd 
# import numpy as np
# from sklearn.model_selection import cross_val_score, train_test_split, KFold, GridSearchCV
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import shap

# # Load the Excel file into a DataFrame
# current_directory = os.getcwd()
# excel_file_name = "hpc_compressive_strength.xlsx"
# excel_file_path = os.path.join(current_directory, excel_file_name)
# df = pd.read_excel(excel_file_path)

# # Clean and prepare the data
# input_variables = df.iloc[:, :9]
# input_variables.columns = [col.split(' (')[0].replace('Concrete ', '').lower() for col in input_variables.columns]

# # Remove missing values and separate features and target variable
# df = df.dropna()
# X = df.drop('Concrete compressive strength (MPa, megapascals) ', axis=1)
# y = df['Concrete compressive strength (MPa, megapascals) ']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# # Feature Scaling
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Define parameter grid for GridSearchCV
# param_grid = {
#     'n_estimators': [100, 150, 200],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 5, 7],
#     'subsample': [0.8, 1.0],
# }
# xgb_model = XGBRegressor()

# # Grid Search for Hyperparameter Tuning
# grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)
# grid_search.fit(X_train_scaled, y_train)

# # Best model from GridSearchCV
# best_model = grid_search.best_estimator_

# # Cross-validation for SHAP analysis
# n_splits = 5
# kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
# shap_values_per_fold = []

# for fold_num, (train_index, val_index) in enumerate(kf.split(X_train_scaled), 1):
#     X_train_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[val_index]
#     y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
#     # Train the model on the fold
#     best_model.fit(X_train_fold, y_train_fold)
    
#     # Calculate SHAP values for the validation set
#     explainer = shap.Explainer(best_model, X_train_fold)
#     shap_values = explainer(X_val_fold)
    
#     # Append SHAP values for the current fold (keep the 2D shape)
#     shap_values_per_fold.append(shap_values.values)
    
#     # Print SHAP values per fold
#     print(f"SHAP Values for Fold {fold_num}:")
#     print(shap_values.values)
#     print("\n")

# # Stack SHAP values across all folds (stacking preserves the shape)
# shap_values_concat = np.vstack(shap_values_per_fold)

# # Compute the average SHAP values across all folds (this will be a 2D matrix)
# avg_shap_values = np.mean(shap_values_concat, axis=0)

# # Print the average SHAP values across the folds
# print("Average SHAP Values across all 5 Folds:")
# print(avg_shap_values)

# # Convert to a DataFrame to maintain feature names and ensure 2D shape
# avg_shap_values_df = pd.DataFrame(avg_shap_values.reshape(1, -1), columns=X_train.columns.values)

# # Plot SHAP summary
# shap.summary_plot(avg_shap_values_df.values, X_train, plot_type="bar")


# # Final model evaluation on the test set
# y_pred = best_model.predict(X_test_scaled)
# r2 = r2_score(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)

# # Cross-validation results
# cv_scores = cross_val_score(best_model, X_train_scaled, y_train, scoring='r2', cv=kf)
# print(f"Cross-Validation R² Scores: {cv_scores}")
# print(f"Mean R² Score: {cv_scores.mean()}")
# print(f"Standard Deviation of R² Scores: {cv_scores.std()}")

# # Model performance metrics
# print(f'R-squared (R2): {r2}')
# print(f'Mean Squared Error (MSE): {mse}')
# print(f'Mean Absolute Error (MAE): {mae}')

# # Visualize predictions against actual values
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred, color='green', label='Test Set', alpha=0.7)
# plt.xlabel('Actual Compressive Strength')
# plt.ylabel('Predicted Compressive Strength')
# plt.legend()
# plt.show()

import docx
import pandas as pd


print("python-docx is installed successfully!")

# Load the Word document
doc_path = "docc.docx"  # Replace with your actual file path
doc = docx.Document(doc_path)

# Extract table data
data = []
for table in doc.tables:
    for row in table.rows:
        data.append([cell.text.strip() for cell in row.cells])

# Convert to DataFrame
df = pd.DataFrame(data)

# Rename columns (Ensure the first row is used as headers)
df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)

# Save to Excel
excel_path = "output.xlsx"
df.to_excel(excel_path, index=False)

print(f"Excel file saved: {excel_path}")
