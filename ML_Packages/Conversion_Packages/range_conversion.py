import pandas as pd




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