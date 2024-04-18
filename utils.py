import pandas as pd
import numpy as np  

def convert_ft_to_meters(dataset, *parameters):
    for param in parameters:
        dataset[param] = dataset[param].apply(pd.to_numeric, errors='coerce')  
        dataset[param] = dataset[param].apply(lambda x: x * 0.3048)  # Convert feet to meters
    return dataset



