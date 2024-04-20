import pandas as pd
import numpy as np 
from googlesearch import search


def convert_ft_to_meters(dataset, *parameters):
    for param in parameters:
        dataset[param] = dataset[param].apply(pd.to_numeric, errors='coerce')  
        dataset[param] = dataset[param].apply(lambda x: x * 0.3048)  # Convert feet to meters
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


if __name__ == '__main__':
    dataset = pd.read_csv('Datasets/Aircraft Performance (Aircraft Bluebook)/Airplane_Cleaned.csv')
    #print(get_actual_company_names(dataset, 'Company'))
    #print(aircraft_name(dataset, 'Aircraft Name'))



