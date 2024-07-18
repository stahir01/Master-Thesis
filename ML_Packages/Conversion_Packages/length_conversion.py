import pandas as pd


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