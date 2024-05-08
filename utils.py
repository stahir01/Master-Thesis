import pandas as pd
import numpy as np 
#from googlesearch import search


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



def convert_length_to_meters(dataset, *parameters, ft=False, cm=False):
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
        # Convert feet to meters
        if ft:
            dataset[param] = dataset[param].apply(lambda x: x * 0.3048)
        
        # Convert cm to meters
        if cm:
            dataset[param] = dataset[param].apply(lambda x: x * 0.01)
    
    return dataset



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


def change_col_datatypes(dataframe, mapping_dict):
    """
    Change the datatype of columns in a dataframe based on the mapping dictionary provided.
    Args:
        dataframe: pd.DataFrame
        mapping_dict: dict, mapping of column names to their respective datatypes
    Returns:
        dataframe: pd.DataFrame
    """
    cols_not_converted = []
    
    for col in dataframe.columns:
        if col in mapping_dict:
            
            try:
                dataframe[col] = dataframe[col].astype(mapping_dict[col])
            except Exception as e:
                print(f"Error converting column '{col}' to {mapping_dict[col]}:", e)
                # Replace "-" with NaN values
                dataframe[col] = dataframe[col].replace("-", np.nan)

               # Try converting to numeric with errors='coerce' 
                try:
                    dataframe[col] = dataframe[col].astype(mapping_dict[col])
                except Exception as e:
                    #print(f"Error converting column '{col}' to {mapping_dict[col]} after replacing '-':", e)
                    dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
                    
                    try:
                        dataframe[col] = dataframe[col].astype(mapping_dict[col])
                    except Exception as e:
                        #print(f"Error converting column '{col}' to {mapping_dict[col]} after replacing '-' with NaN and converting to numeric:", e)
                        cols_not_converted.append(col)
        else:
            print(f"No datatype specified for column '{col}', skipping conversion")
            cols_not_converted.append(col)

    print("Columns not converted:", cols_not_converted)
    return dataframe



if __name__ == '__main__':
    dataset = pd.read_csv('Datasets/Aircraft Performance (Aircraft Bluebook)/Airplane_Cleaned.csv')
    #print(get_actual_company_names(dataset, 'Company'))
    #print(aircraft_name(dataset, 'Aircraft Name'))



