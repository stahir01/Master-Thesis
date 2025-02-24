import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, f1_score, precision_score,
    mean_absolute_error, mean_squared_error, r2_score, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
import shap
from IPython.display import display




#from googlesearch import search


def convert_weight_to_kg(dataset, *parameters, lbs=False, grams=False):
    """
    Convert weight to kg
    Args:
        dataset: pd.DataFrame
        parameters: list of columns to convert 
        lbs: if True, converts from lbs to kg
        grams: if True, converts from grams to kg
    Returns:
        dataset: pd.DataFrame
    """

    for param in parameters:
        dataset[param] = dataset[param].apply(pd.to_numeric, errors='coerce')
        # Convert lbs to kg
        if lbs:
            dataset[param] = dataset[param].apply(lambda x: x * 0.453592)
        
        # Convert grams to kg
        if grams:
            dataset[param] = dataset[param].apply(lambda x: x * 0.001)
    
    return dataset

def convert_temp_to_celsius(dataset, *parameters, fahrenheit=False):
    """
    Convert temperature to Celsius
    Args:
        dataset: pd.DataFrame
        parameters: list of columns to convert 
        fahrenheit: if True, converts from fahrenheit to celsius
    Returns:
        dataset: pd.DataFrame
    """
    for param in parameters:
        dataset[param] = dataset[param].apply(pd.to_numeric, errors='coerce')
        # Convert fahrenheit to celsius
        if fahrenheit:
            dataset[param] = dataset[param].apply(lambda x: (x - 32) * 5/9)
    
    return dataset



def convert_length_to_meters(dataset, *parameters, km = False, ft=False, cm=False):
    """
    Convert length from feet or centimeters to meters
    Args:
        dataset: pd.DataFrame
        parameters: list of columns to convert 
        ft: if True, converts from feet to meters
        cm: if True, converts from centimeters to meters
    Returns:
        dataset: pd.DataFrame
    """
    for param in parameters:
        dataset[param] = dataset[param].apply(pd.to_numeric, errors='coerce')

        # Convert km to meters
        if km:
            dataset[param] = dataset[param].apply(lambda x: x * 1000)

        # Convert feet to meters
        if ft:
            dataset[param] = dataset[param].apply(lambda x: x * 0.3048)
        
        # Convert cm to meters
        if cm:
            dataset[param] = dataset[param].apply(lambda x: x * 0.01)
    
    return dataset

def convert_speed(dataset, *parameters, knots=False):
    """
    Convert speed from knots to kilometers per hour (km/h).
    
    Args:
        dataset: pd.DataFrame
        parameters: list of columns to convert
        knots_to_kmh: True if the speed is in knots and needs to be converted to km/h, False otherwise
        
    Returns:
        dataset: pd.DataFrame
    """
    for param in parameters:
        dataset[param] = dataset[param].apply(pd.to_numeric, errors='coerce')
        
        # Convert knots to km/h
        if knots:
            dataset[param] = dataset[param].apply(lambda x: x * 1.852)
    
    return dataset



def convert_loading(dataset, *parameters, kg_m2=False, lbs_ft2=False):
    """
    Convert loading parameters to newtons per square meter (N/m^2).
    
    Args:
        dataset: pd.DataFrame
        parameters: list of columns to convert
        kg_m2: True if the parameters are in kg/m^2
        lbs_ft2: True if the parameters are in lbs/ft^2
        
    Returns:
        dataset: pd.DataFrame
    """
    for param in parameters:
        dataset[param] = dataset[param].apply(pd.to_numeric, errors='coerce')
        
        # Convert kg/m^2 to N/m^2
        if kg_m2:
            dataset[param] = dataset[param].apply(lambda x: x * 9.80665)
        
        # Convert lbs/ft^2 to N/m^2
        if lbs_ft2:
            dataset[param] = dataset[param].apply(lambda x: x * 47.8803)
    
    return dataset

def convert_knots_to_mach(dataset, *parameters):
    """
    Convert speed from knots to Mach number.

    Args:
        dataset: pd.DataFrame
        parameters: list of columns to convert
    
    Returns:
        dataset: pd.DataFrame
    """
    for param in parameters:
        dataset[param] = dataset[param].apply(pd.to_numeric, errors='coerce')
        dataset[param] = dataset[param].apply(lambda x: x * 0.0015)

    return dataset

def convert_range(dataset, *parameters, nm=False, km=False):
    """
    Convert range from nautical miles to meters.
    
    Args:
        dataset: pd.DataFrame
        parameters: list of columns to convert
        nm: True if the range is in nautical miles
        km: True if the range is in kilometers
        
    Returns:
        dataset: pd.DataFrame
    """
    for param in parameters:
        dataset[param] = dataset[param].apply(pd.to_numeric, errors='coerce')
        
        # Convert nautical miles to meters
        if nm:
            dataset[param] = dataset[param].apply(lambda x: x * 1609.344)
        
        # Convert kilometers to meters
        if km:
            dataset[param] = dataset[param].apply(lambda x: x * 1000)
    
    return dataset



def aircraft_name(dataset, column_name):
    """
    Cleanup aircraft name

    Args:
    dataset: pd.DataFrame
    column_name(str): Name of the column in the dataset

    Returns:
    column_name: cleaned column
    """

    #Remove any special character at the end of the name
    for i in range(len(dataset[column_name])):
        # Remove bracket from the name of the aircraft
        idx_opening_bracket = dataset[column_name][i].find('(')
        if idx_opening_bracket != -1:
            dataset[column_name][i] = dataset[column_name][i][:idx_opening_bracket].strip()

        # Remove everything after the comma
        idx_comma = dataset[column_name][i].find(',', idx_opening_bracket)
        if idx_comma != -1:
            dataset[column_name][i] = dataset[column_name][i][:idx_comma].strip()

        #Remove special character at the end
        special_characters = "!@#$%^&*()_+|[]\\:\";'<>?,./"
        if dataset[column_name][i][-1] in special_characters:
            dataset[column_name][i] = dataset[column_name][i][:-1].strip()
        
        dataset[column_name][i] = str(dataset[column_name][i])

    return dataset


def preprocess_data(dataframe, columns_to_process):
    """
    Preprocess the specified columns in a dataframe.
    Args:
        dataframe: pd.DataFrame
        columns_to_process: list of str, names of columns to preprocess
    Returns:
        dataframe: pd.DataFrame
    """
    for col in columns_to_process:
        if dataframe[col].dtype == 'object':
            dataframe[col] = dataframe[col].str.replace(',', '.')
    return dataframe


def change_col_datatypes(dataframe, mapping_dict):
    """
    Change the datatype of columns in a dataframe based on the mapping dictionary provided.
    Args:
        dataframe: pd.DataFrame
        mapping_dict: dict, mapping of column names to their respective datatypes
    Returns:
        dataframe: pd.DataFrame
    """
    cols_not_converted = []
    
    for col in dataframe.columns:
        if col in mapping_dict:
            
            try:
                dataframe[col] = dataframe[col].astype(mapping_dict[col])
            except Exception as e:
                print(f"Error converting column '{col}' to {mapping_dict[col]}:", e)
                # Replace "-" with NaN values
                dataframe[col] = dataframe[col].replace("-", np.nan)

               # Try converting to numeric with errors='coerce' 
                try:
                    dataframe[col] = dataframe[col].astype(mapping_dict[col])
                except Exception as e:
                    #print(f"Error converting column '{col}' to {mapping_dict[col]} after replacing '-':", e)
                    dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
                    
                    try:
                        dataframe[col] = dataframe[col].astype(mapping_dict[col])
                    except Exception as e:
                        #print(f"Error converting column '{col}' to {mapping_dict[col]} after replacing '-' with NaN and converting to numeric:", e)
                        cols_not_converted.append(col)
        else:
            print(f"No datatype specified for column '{col}', skipping conversion")
            cols_not_converted.append(col)

    print("Columns not converted:", cols_not_converted)
    return dataframe


def plot_loss_metrics(model_results):
    """
    Plots the loss, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) scores of a TensorFlow model.
    Args:
        model_results (object): The history object of the trained model.
    
    Returns:
        None
    """
    train_loss = model_results.history['loss']
    val_loss = model_results.history['val_loss']
    train_mae = model_results.history['mean_absolute_error']
    val_mae = model_results.history['val_mean_absolute_error']
    train_rmse = model_results.history['root_mean_squared_error']
    val_rmse = model_results.history['val_root_mean_squared_error']

    epochs = range(1, len(train_loss) + 1)

    # Create subplots with 1 row and 3 columns
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot loss scores
    axs[0].plot(epochs, train_loss, 'g', label='Training Loss')
    axs[0].plot(epochs, val_loss, 'b', label='Validation Loss')
    axs[0].set_title('Loss Scores')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot MAE scores
    axs[1].plot(epochs, train_mae, 'c', label='Training MAE')
    axs[1].plot(epochs, val_mae, 'm', label='Validation MAE')
    axs[1].set_title('MAE Scores')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Mean Absolute Error')
    axs[1].legend()

    # Plot RMSE scores
    axs[2].plot(epochs, train_rmse, 'y', label='Training RMSE')
    axs[2].plot(epochs, val_rmse, 'r', label='Validation RMSE')
    axs[2].set_title('RMSE Scores')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Root Mean Squared Error')
    axs[2].legend()

    # Show the plots
    plt.show()





def prep_model_data(dataset, features, target, test_size, random_state, model_type='cnn'):
    """
    Prepare data to train the model.
    
    Args:
        dataset (pd.DataFrame): Dataset to train the model
        features (list): List of feature columns to train the model
        target (str): Target variable column
        test_size (float): Size of the test set
        random_state (int): Random state for reproducibility
        model_type (str): Type of model to train ('cnn' for CNN, 'ml' for traditional ML)
    
    Returns:
        tuple: Processed training and test data, and scalers
    """
    X = dataset[features]
    y = dataset[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Input training size: {X_train.shape}\n Input test size: {X_test.shape}\n Labels for training size: {y_train.shape}\n Labels for testing size: {y_test.shape}")

    
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy() 
    y_train = y_train.to_numpy().reshape(-1, 1) # Output: (191, 1)
    y_test = y_test.to_numpy().reshape(-1, 1)

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    if model_type != 'cnn':
        X_train_reshaped = X_train_scaled
        X_test_reshaped = X_test_scaled
    else:
        # Reshape the data for CNN
        X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
        X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
        print(f"Reshaped training size: {X_train_reshaped.shape}\nReshaped test size: {X_test_reshaped.shape}")

    return X_train_reshaped, X_test_reshaped, y_train_scaled, y_test_scaled, x_scaler, y_scaler

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


def plt_feature_importance(model, feature_names):

    """
    Plot the feature importance of a model.

    Args:
        model: The trained model
        feature_names: List of feature names
        
    Returns:
        None
    """
    
    # Extract feature importance from the random forest regressor
    feature_importance = model.feature_importances_

    #Calculate the percentage of importance
    feature_importance_percentage = 100.0 * (feature_importance / feature_importance.sum())

    # Create a DataFrame with features and their importance scores
    feature_importance_df = {
        'Feature': feature_names,
        'Importance': feature_importance_percentage
    }

    feature_importance_df = pd.DataFrame(feature_importance_df)
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot the feature importances
    plt.figure(figsize=(8, 4))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, color='maroon')

    for index, value in enumerate(feature_importance_df['Importance']):
        plt.text(value, index, f'{value:.2f}%', color='black', ha="left", va="center")

    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.show()    


class CNNFeatureImportance:
    def __init__(self, model, X_train, X_test, features, index=0):
        """
        Initialize the CNNFeatureImportance class.

        Args:
            model: The trained CNN model.
            X_train: The training data.
            X_test: The testing data.
            features: The feature names.
            index: The index of the instance to explain. Default is 0.
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.features = features
        self.index = index

        # Reshape the data to match the input shape of the CNN model
        self.X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        self.X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Initialize the JS visualization code for Jupyter Notebook
        shap.initjs()

        # Create a SHAP DeepExplainer for the CNN model
        self.explainer = shap.DeepExplainer(self.model, self.X_train_reshaped)

        # Calculate SHAP values
        self.shap_values = self.explainer.shap_values(self.X_test_reshaped)

        # Reshape the SHAP values for the force plot
        self.shap_values_reshaped = self.shap_values[self.index].reshape(-1, X_test.shape[1])
        self.instance_shap_values = self.shap_values[self.index].reshape(-1)

        # Convert tensor value to float
        self.expected_value = self.explainer.expected_value[0].numpy() if hasattr(self.explainer.expected_value[0], 'numpy') else float(self.explainer.expected_value[0])

    def plot_summary(self):
        """
        Plot the summary plot for feature importance.
        """

        shap.summary_plot(self.shap_values_reshaped, self.X_test, feature_names=self.features, plot_type="bar", color='maroon')

    def plot_force(self):
        """
        Plot the force plot for a specific instance.
        """

        force_plot = shap.force_plot(base_value=self.expected_value, shap_values=self.shap_values_reshaped, features=self.X_test[self.index], feature_names=self.features)
        display(force_plot)
    
    def plot_waterfall(self):
        """
        Plot the waterfall plot for a specific instance.

        Args:
            None
        
        Returns:
            None
        """
        shap.waterfall_plot(
        shap.Explanation(values=self.instance_shap_values,  base_values=self.expected_value,  data=self.X_test[self.index], feature_names=self.features))

        


if __name__ == '__main__':
    dataset = pd.read_csv('Datasets/Aircraft Performance (Aircraft Bluebook)/Airplane_Cleaned.csv')
    #print(get_actual_company_names(dataset, 'Company'))
    #print(aircraft_name(dataset, 'Aircraft Name'))



