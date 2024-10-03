import pandas as pd
from ctgan import CTGAN
from sdv.sampling import Condition


class GANModelEvaluation: 
    def __init__(self, prediction_data, parameters, model):
        """
        Initialize the GAN model trained for prediction based on certain parmaeters

        Args:
            prediction_data (DataFrame): The dataset required for model prediction
            parameters (list): The list of parameters provided for prediction (Conditional Sampling)
            model (CTGANSynthesizer): The trained GAN model
        """
        self.prediction_data = prediction_data
        self.parameters = parameters
        self.model = model

    def parameter_for_prediction(self, total_rows):
        """
        Extracts the input values from specified rows for future predictions.

        This method selects a subset of rows from the dataset and collects the specified 
        parameters for each of those rows. These values will be used as conditions for generating 
        synthetic data in later predictions. 

        Args:
            total_rows (list): A list of row indices representing the rows from which parameter 
                            values are extracted for prediction.
        
        Returns:
            prediction_input_values (dict): A dictionary where each key corresponds to a row, 
                                            and its value is a dictionary of parameter values from 
                                            the selected rows, which will be used for prediction.
        """
        prediction_input_values = {}

        for row in total_rows:
            row_values = {}
            for index, rows in self.prediction_data[self.parameters].iterrows():
                if index == row:
                    for param in self.parameters:
                        row_values[param] = rows[param]
            prediction_input_values[f'Row_{row}'] = row_values

        return prediction_input_values
    
    def predict_values(self, prediction_input_values, rows_to_produce=2, max_tries_per_batch=500, batch_size=10):
        """
        Generates synthetic data based on the input values using the GAN model.

        This method takes the input values (parameters) from selected rows and uses them as 
        conditions to generate synthetic data using the trained GAN model. The generated predictions 
        for each row are stored in a dictionary, where each key corresponds to a specific row.

        Args:
            prediction_input_values (dict): A dictionary containing input values for each row, 
                                            where the key is the row and the value is a dictionary 
                                            of parameters.
            rows_to_produce (int): The number of synthetic rows to generate for each input condition 
                                (default: 2).
            max_tries_per_batch (int): The maximum number of attempts to generate valid synthetic 
                                    rows for each batch (default: 500).
            batch_size (int): The number of rows to generate per batch (default: 10).
        
        Returns:
            prediction_output (dict): A dictionary where each key corresponds to a row, and its 
                                    value is the generated synthetic data based on the input conditions 
        """

        prediction_output = {}

        for key, value in prediction_input_values.items():
            # Condition values for the given parameters
            condition_values = Condition(
                num_rows=rows_to_produce,
                column_values=value
            )

            # Generate synthetic data based on the condition
            synthetic_data_new = self.model.sample_from_conditions(
                conditions=[condition_values],
                max_tries_per_batch=max_tries_per_batch,
                batch_size=batch_size
            )

            # Store the prediction output for the current row
            prediction_output[key] = synthetic_data_new
        
        return prediction_output

    def prediction_analysis(self, prediction_output, total_rows):
        """
        Analyze the synthetic data generated with real data for comparison.

        Args:
            prediction_output (dict): A dictionary containing the generated synthetic data for each row.
            total_rows (list): A list of row indices representing the rows for which synthetic data was generated.
        
        Returns:
            None
        """
        predition_output_df = pd.DataFrame.from_dict(prediction_output, orient='index')

        print(predition_output_df)

        print(f"Orginal Dataframe: {self.prediction_data}")
        
