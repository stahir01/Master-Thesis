import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



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

    x_scaler = MinMaxScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    y_scaler = MinMaxScaler()
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



def prep_prediction_data(dataset, features, model_type='cnn'):
    """
    Prepare dataset for model prediction

    Args:
        data (pd.DataFrame): Data to predict
        features (list): List of feature columns to predict
        model_type (str): Type of model to train ('cnn' for CNN, 'ml' for traditional ML)
    
    Returns:
        X_scaled (np.array): Processed data for prediction
        x_scaler (StandardScaler): Scaler used to scale the data
    """

    X = dataset[features]

    X = X.to_numpy()

    x_scaler = MinMaxScaler()
    X_scaled = x_scaler.fit_transform(X)

    if model_type == 'cnn':
        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    return X_scaled, x_scaler
