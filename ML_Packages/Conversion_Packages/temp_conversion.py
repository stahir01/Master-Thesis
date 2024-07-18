import pandas as pd


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