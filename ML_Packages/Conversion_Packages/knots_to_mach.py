import pandas as pd


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