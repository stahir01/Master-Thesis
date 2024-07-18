import pandas as pd
import numpy as np




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


def preprocess_data(dataframe, columns_to_process):
    """
    Preprocess the specified columns in a dataframe.
    Args:
        dataframe: pd.DataFrame
        columns_to_process: list of str, names of columns to preprocess
    Returns:
        dataframe: pd.DataFrame
    """
    for col in columns_to_process:
        if dataframe[col].dtype == 'object':
            dataframe[col] = dataframe[col].str.replace(',', '.')
    return dataframe


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



def test():
    print('Hello World!')