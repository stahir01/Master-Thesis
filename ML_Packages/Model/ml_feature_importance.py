import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.inspection import permutation_importance


class MLFeatureImportance:
    def __init__(self, model, X_train, y_train, feature_names):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names

    def plot_feature_importance(self):
        """
        Plot the feature importance of a model.
        This method works for models that have the `feature_importances_` attribute (e.g., RandomForest).
        """
        if hasattr(self.model, 'feature_importances_'):
            # Extract feature importance from the model
            feature_importance = self.model.feature_importances_

            # Calculate the percentage of importance
            feature_importance_percentage = 100.0 * (feature_importance / feature_importance.sum())

            # Create a DataFrame with features and their importance scores
            feature_importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': feature_importance_percentage
            }).sort_values(by='Importance', ascending=False)

            # Plot the feature importances
            plt.figure(figsize=(8, 4))
            sns.barplot(x='Importance', y='Feature', data=feature_importance_df, color='maroon')

            for index, value in enumerate(feature_importance_df['Importance']):
                plt.text(value, index, f'{value:.2f}%', color='black', ha="left", va="center")

            plt.title('Feature Importance')
            plt.xlabel('Importance Score')
            plt.ylabel('Feature')
            plt.show()
        else:
            print("The model does not have an attribute feature_importances_. Using permutation importance instead.")
            self.plot_permutation_importance_as_bar()

    def plot_permutation_importance(self, n_repeats=30, random_state=42):
        """
        Plot permutation importance of a model using a box plot.
        This method works for models that do not have the `feature_importances_` attribute (e.g., SVR).
        Args:
            n_repeats: Number of times to permute a feature
            random_state: Random state for reproducibility
        """
        # Calculate permutation importance
        perm_importance = permutation_importance(self.model, self.X_train, self.y_train, n_repeats=n_repeats, random_state=random_state)

        # Extract the importance scores
        perm_importances = perm_importance.importances

        # Create a DataFrame for box plot
        perm_importance_df = pd.DataFrame(perm_importances, index=self.feature_names).T

        # Plot the permutation importances using box plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=perm_importance_df, orient='h', color='maroon')
        plt.xlabel('Decrease in accuracy score')
        plt.title('Permutation Importances on selected subset of features (train set)')
        plt.show()

    def plot_permutation_importance_as_bar(self, n_repeats=30, random_state=42):
        """
        Plot permutation importance of a model using a bar plot.
        Args:
            n_repeats: Number of times to permute a feature
            random_state: Random state for reproducibility
        """
        # Calculate permutation importance
        perm_importance = permutation_importance(self.model, self.X_train, self.y_train, n_repeats=n_repeats, random_state=random_state)

        # Create a DataFrame with features and their importance scores
        feature_importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': perm_importance.importances_mean
        }).sort_values(by='Importance', ascending=False)

        # Plot the feature importances
        plt.figure(figsize=(8, 4))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, color='maroon')

        for index, value in enumerate(feature_importance_df['Importance']):
            plt.text(value, index, f'{value:.2f}', color='black', ha="left", va="center")

        plt.title('Permutation Feature Importance')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.show()
