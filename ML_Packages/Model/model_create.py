import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAAnalysis:
    def __init__(self, dataset, parameters=[], n_components=2):
        self.dataset = dataset[parameters]  # Corrected this line
        self.parameters = parameters
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
        self.scaler = StandardScaler()
        self.explained_variance_ = None
        self.components_ = None

    def fit_transform(self):
        """
        Fits PCA to the data and returns the transformed data.
        Parameters: 
            None
        Returns:
            pd.DataFrame: DataFrame with the principal components.
        """
        standardized_data = self.scaler.fit_transform(self.dataset)
        principal_components = self.pca.fit_transform(standardized_data)
        self.explained_variance_ = self.pca.explained_variance_ratio_
        self.components_ = self.pca.components_

        pc_df = pd.DataFrame(data=principal_components, 
                             columns=[f'Principal Component {i+1}' for i in range(self.n_components)])
        return pc_df
    
    def plot_variance_explained(self):
        """
        Plots the explained variance by each principal component.
        """
        plt.figure(figsize=(8, 6))
        plt.bar(range(1, len(self.explained_variance_)+1), self.explained_variance_)
        plt.xlabel('PCA Component')
        plt.ylabel('Explained Variance')
        plt.title('Explained Variance by Principal Component')
        plt.xticks(range(1, len(self.explained_variance_)+1), 
                   [f'PC{i+1}' for i in range(self.n_components)], rotation=45, ha='right')
        
        # Add variance values on the bars
        for i, v in enumerate(self.explained_variance_):
            plt.text(i + 1, v + 0.01, f"{v:.4f}", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

    def most_important_features(self, top_n=5):
        """
            Identifies the most important features in each principal component and plots a bar chart for each component.
            
            Parameters:
            top_n (int): Number of top features to return for each component.
            
            Returns:
            dict: Dictionary containing the top features for each principal component.
        """
        most_important = {}
        for i in range(self.n_components):
            component = self.components_[i]
            feature_indices = np.argsort(np.abs(component))[-top_n:]
            
            # Sort the features by their importance
            sorted_indices = feature_indices[np.argsort(np.abs(component[feature_indices]))][::-1]
            most_important[f'Principal Component {i+1}'] = [(self.parameters[j], component[j]) 
                                                            for j in sorted_indices]
            
            # Extract feature names and their corresponding importances
            feature_names = [self.parameters[j] for j in sorted_indices]
            importances = [component[j] for j in sorted_indices]
            
            # Plotting the bar chart for the current principal component
            plt.figure(figsize=(8, 2))
            plt.barh(feature_names, importances, color='maroon')
            plt.xlabel('Feature Importance')
            plt.ylabel('Features')
            plt.title(f'Feature Importances for Principal Component {i+1}')
            plt.gca().invert_yaxis()  # To display the highest importance at the top
            plt.show()
        
        # Print the most important features for each component
        for component, features in most_important.items():
            print(f"{component}:")
            for feature, importance in features:
                print(f"  {feature}: {importance:.4f}")
        
        return most_important


