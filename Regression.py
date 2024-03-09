import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

file_path = "files/California_Houses.csv"
data = pd.read_csv(file_path)  # data is a dataframe
# Split the balanced dataset into training and the rest
training_set, remaining_data = train_test_split(data, test_size=0.3, random_state=42)
#  split the remaining into validation and testing sets
#  validation set should be reserved strictly for evaluating
#  the model's performance after you've trained it using the training set and adjusted the hyperparameters separately.
validation_set, test_set = train_test_split(
    remaining_data, test_size=0.5, random_state=42
)
# Define features (X) and target variable (y)
x_train = training_set.drop(columns=["Median_House_Value"])
y_train = training_set["Median_House_Value"]

x_validation = validation_set.drop(columns=["Median_House_Value"])
y_validation = validation_set["Median_House_Value"]

x_test = test_set.drop(columns=["Median_House_Value"])
y_test = test_set["Median_House_Value"]

# Apply Linear Regression
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)
# Evaluate Linear Regression on validation set( Calculate Mean Square Error for Linear Regression)
linear_val_predictions = linear_model.predict(x_validation)
linear_val_mse = mean_squared_error(y_validation, linear_val_predictions)
print("***************************************************")
print("Linear Regression Validation MSE:", linear_val_mse)
# Calculate Mean Absolute Error for Linear Regression
linear_val_mae = mean_absolute_error(y_validation, linear_val_predictions)
print("Linear Regression Validation MAE:", linear_val_mae)
print("***************************************************")

# Apply Lasso Regression
lasso_model = Lasso(alpha=0.1, max_iter=10000)  # You can adjust the alpha parameter
lasso_model.fit(x_train, y_train)

# Evaluate Lasso Regression on validation set
lasso_val_predictions = lasso_model.predict(x_validation)
lasso_val_mse = mean_squared_error(y_validation, lasso_val_predictions)
print("***************************************************")
print("Lasso Regression Validation MSE:", lasso_val_mse)
# Calculate Mean Absolute Error for Lasso Regression
lasso_val_mae = mean_absolute_error(y_validation, lasso_val_predictions)
print("Lasso Regression Validation MAE:", lasso_val_mae)
print("***************************************************")

# Apply Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(x_train, y_train)

# Evaluate Ridge Regression on validation set
ridge_val_predictions = ridge_model.predict(x_validation)
ridge_val_mse = mean_squared_error(y_validation, ridge_val_predictions)
print("Ridge Regression Validation MSE:", ridge_val_mse)
# Calculate Mean Absolute Error for Ridge Regression
ridge_val_predictions = ridge_model.predict(x_validation)
ridge_val_mae = mean_absolute_error(y_validation, ridge_val_predictions)
print("Ridge Regression Validation MAE:", ridge_val_mae)
print("***************************************************")

# Calculate Mean Squared Error for Linear Regression on test set
linear_test_predictions = linear_model.predict(x_test)
linear_test_mse = mean_squared_error(y_test, linear_test_predictions)
print("Linear Regression Test MSE:", linear_test_mse)
# Calculate Mean Absolute Error for Linear Regression on test set
linear_test_mae = mean_absolute_error(y_test, linear_test_predictions)
print("Linear Regression Test MAE:", linear_test_mae)
print("***************************************************")

# After selecting the best model based on validation performance, you can evaluate it on the test set
lasso_test_predictions = lasso_model.predict(x_test)
lasso_test_mse = mean_squared_error(y_test, lasso_test_predictions)

print("Lasso Regression Test MSE:", lasso_test_mse)
# Calculate Mean Absolute Error for Lasso Regression on test set
lasso_test_mae = mean_absolute_error(y_test, lasso_test_predictions)
print("Lasso Regression Test MAE:", lasso_test_mae)
# Calculate Mean Squared Error for Ridge Regression on test set
ridge_test_predictions = ridge_model.predict(x_test)
ridge_test_mse = mean_squared_error(y_test, ridge_test_predictions)
print("***************************************************")

print("Ridge Regression Test MSE:", ridge_test_mse)
# Calculate Mean Absolute Error for Ridge Regression on test set
ridge_test_mae = mean_absolute_error(y_test, ridge_test_predictions)
print("Ridge Regression Test MAE:", ridge_test_mae)
print("***************************************************")

# Validation Set Evaluation
# Find the best-performing model based on validation set MSE
best_model_mse = min(linear_val_mse, min(lasso_val_mse, ridge_val_mse))
best_model_mae = min(linear_val_mae, min(lasso_val_mae, ridge_val_mae))
if best_model_mse == linear_val_mse:
    print("Best model based on Validation Set: Linear regression")
    print("MSE:", best_model_mse)
    print("MAE:", best_model_mae)
elif best_model_mse == lasso_val_mse:
    print("Best model based on Validation Set: Lasso regression")
    print("MSE:", best_model_mse)
    print("MAE:", best_model_mae)
else:
    print("Best model based on Validation Set: Ridge regression")
    print("MSE:", best_model_mse)
    print("MAE:", best_model_mae)

print("***************************************************")

# Test Set Evaluation
# Evaluate the best-performing model on the test set
if best_model_mse == linear_val_mse:
    print("Best model based on Test Set: Linear regression")
    print("MSE:", linear_test_mse)
    print("MAE:", linear_test_mae)
elif best_model_mse == lasso_val_mse:
    print("Best model based on Test Set: Lasso regression")
    print("MSE:", lasso_test_mse)
    print("MAE:", lasso_test_mae)
else:  # Ridge Regression
    print("Best model based on Test Set: Ridge regression")
    print("MSE:", ridge_test_mse)
    print("MAE:", ridge_test_mae)

print("***************************************************")
print("***************************************************")
