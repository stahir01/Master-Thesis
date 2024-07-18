import pandas as pd


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