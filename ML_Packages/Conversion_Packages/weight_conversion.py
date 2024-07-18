import pandas as pd


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