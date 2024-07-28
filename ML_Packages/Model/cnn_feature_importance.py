import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from IPython.display import display



class CNNFeatureImportance:
    def __init__(self, model, X_train, X_test, features, index=0):
        """
        Initialize the CNNFeatureImportance class.

        Args:
            model: The trained CNN model.
            X_train: The training data.
            X_test: The testing data.
            features: The feature names.
            index: The index of the instance to explain. Default is 0.
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.features = features
        self.index = index

        # Reshape the data to match the input shape of the CNN model
        self.X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        self.X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Initialize the JS visualization code for Jupyter Notebook
        shap.initjs()

        # Create a SHAP DeepExplainer for the CNN model
        self.explainer = shap.DeepExplainer(self.model, self.X_train_reshaped)

        # Calculate SHAP values
        self.shap_values = self.explainer.shap_values(self.X_test_reshaped)

        # Reshape the SHAP values for the force plot
        self.shap_values_reshaped = self.shap_values[self.index].reshape(-1, X_test.shape[1])
        self.instance_shap_values = self.shap_values[self.index].reshape(-1)

        # Convert tensor value to float
        self.expected_value = self.explainer.expected_value[0].numpy() if hasattr(self.explainer.expected_value[0], 'numpy') else float(self.explainer.expected_value[0])

    def plot_summary(self):
        """
        Plot the summary plot for feature importance.
        """

        shap.summary_plot(self.shap_values_reshaped, self.X_test, feature_names=self.features, plot_type="bar", color='maroon')

    def plot_summary_for_all(self):
        """
        Plot the summary plot for feature importance for all instances.
        """
        # Reshape shap_values to be compatible with the summary_plot
        shap_values_reshaped = [sv.reshape(-1, self.X_test.shape[1]) for sv in self.shap_values]

        # Combine all SHAP values into a single array
        combined_shap_values = np.concatenate(shap_values_reshaped, axis=0)

        shap.summary_plot(combined_shap_values, self.X_test, feature_names=self.features, plot_type="bar", color='maroon')

    def plot_force(self):
        """
        Plot the force plot for a specific instance.
        """

        force_plot = shap.force_plot(base_value=self.expected_value, shap_values=self.shap_values_reshaped, features=self.X_test[self.index], feature_names=self.features)
        display(force_plot)
    
    def plot_waterfall(self):
        """
        Plot the waterfall plot for a specific instance.

        Args:
            None
        
        Returns:
            None
        """
        shap.waterfall_plot(
        shap.Explanation(values=self.instance_shap_values,  base_values=self.expected_value,  data=self.X_test[self.index], feature_names=self.features))

    def waterfall_plot_all(self):
        """
        Plot the waterfall plot for all test instances individually.

        Args:
            None
        
        Returns:
            None
        """
        for i in range(len(self.X_test)):
            instance_shap_values = self.shap_values[i].reshape(-1)
            shap.waterfall_plot(
                shap.Explanation(values=instance_shap_values, base_values=self.expected_value, data=self.X_test[i], feature_names=self.features)
            )

    def decision_plot(self):
        """
        Plot the decision plot for a specific instance.

        Args:
            None
        
        Returns:
            None
        """
        shap.decision_plot(self.expected_value, self.instance_shap_values, feature_names=self.features, highlight=0)

    def decision_plot_all(self):
        """
        Plot the decision plot for all test instances in a grid.

        Args:
            plots_per_row (int): Number of plots per row in the grid.
        
        Returns:
            None
        """
        for i in range(len(self.X_test)):
            instance_shap_values = self.shap_values[i].reshape(-1)
            shap.decision_plot(
                self.expected_value, instance_shap_values, feature_names=self.features,
                highlight=0, show=False
            )