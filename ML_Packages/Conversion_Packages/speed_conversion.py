import pandas as pd


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