import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




def evaluate_regression_model(y_test, y_pred):
    """
    Evaluates a regression model by plotting predicted vs actual values and printing regression metrics.
    
    Args:
        y_test (array-like): The true test labels.
        y_pred (array-like): The predicted labels.
    
    Returns:
        None
    """
    # Calculate regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Print regression metrics
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R2 ): {r2:.4f}")
    
    # Plot predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.show()


def evaluate_model(y_test_scaled, y_pred_scaled, y_scaler):
    """
    Evaluate the performance of a regression model by calculating MAE, RMSE, and R² score.
    
    Args:
    y_test_scaled (np.ndarray): Scaled true target values.
    y_pred_scaled (np.ndarray): Scaled predicted target values.
    y_scaler (StandardScaler): Scaler used to inverse transform the target values.

    Returns:
    dict: Dictionary containing MAE, RMSE, and R² score.
    """
    # Inverse transform the predictions and actual values
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_test_original = y_scaler.inverse_transform(y_test_scaled)

    # Calculate metrics
    mae = mean_absolute_error(y_test_original, y_pred)
    rmse = mean_squared_error(y_test_original, y_pred, squared=False)
    r2 = r2_score(y_test_original, y_pred)
    
    # Return the metrics in a dictionary
    return mae, rmse, r2, y_pred, y_test_original


def evaluate_model_confidence(y_test, y_pred):

    """
    Evaluate the performance of a model by calculating the confidence score.

    Args:
    y_test (np.ndarray): Test dataset
    y_pred (np.ndarray): Predicted values

    Returns:
    pd.DataFrame: DataFrame containing the actual, predicted, residual, and confidence scores.
    float: Average confidence score
    
    """
    
    results_df = pd.DataFrame({
        'Actual': y_test.flatten(),
        'Predicted': y_pred.flatten(),
    })

    # Calculate the residuals
    results_df['Residual'] = abs(results_df['Actual'] - results_df['Predicted'])

    # Calculate confidence scores
    results_df['Confidence'] = 1 / (1 + results_df['Residual'])

    # Calculate average confidence
    avg_confidence = results_df['Confidence'].mean() * 100  # Convert to percentage

    # Sort the DataFrame by confidence
    results_df = results_df.sort_values(by='Confidence', ascending=False)

    # Print the metrics and average confidence
    print(f'Average Confidence: {avg_confidence:.2f}%')

    return results_df, avg_confidence