import pandas as pd
import matplotlib.pyplot as plt
from typing import List

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor, plot_importance
from ML_Packages.Model.model_prep import prep_model_data
from ML_Packages.Model.model_evaluation import evaluate_model, evaluate_model_confidence

class XGboostModel():
    def __init__(self, dataset: pd.DataFrame, features: List[str], target: List[str]):
        """
        Initialize the XGboostModel class with the dataset, features, and target
        Args:
            dataset (pd.DataFrame): Dataset to train the model
            features (list): List of feature columns to train the model
            target (str): Target variable column
        
        """
        self.dataset = dataset
        self.features = features
        self.target = target

    def prep_XGboost_data(self, test_split=0.2, random_state=42, model_type='XGboost'):
        """
        Split the dataset into training and testing sets, and normalize the data

        returns:
            X_train: Training dataset
            X_test: Testing dataset
            y_train: Training labels
            y_test: Testing labels
            x_scaler: Normalized Scalar used for the features
            y_scaler: Normalized Scalar used for the target
        """
        self.X_train, self.X_test, self.y_train, self.y_test, self.x_scaler, self.y_scaler = prep_model_data(self.dataset, self.features, self.target, test_size=test_split, random_state=random_state, model_type=model_type)
    
    def model_train(self, n_estimators=100, learning_rate=0.1, max_depth=5,  subsample=0.8, colsample_bytree=0.8, random_state=42):
        """
        Model Preperation and Training

        Args:
            n_estimators (int): Number of trees in the forest
            learning_rate (float): Step size shrinkage used to prevent overfitting
            max_depth (int): Maximum depth of the tree
            subsample (float): Subsample ratio of the training instances
            colsample_bytree (float): Subsample ratio of columns when constructing each tree
            random_state (int): Random state for repro
        """

        self.model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, subsample=subsample, colsample_bytree=colsample_bytree, random_state=random_state)
        self.model.fit(self.X_train, self.y_train, verbose=True)

        return self.model

    def model_predict(self):
        """
        Model predictions on the test set
        
        returns:
            y_pred: Predicted normalized values
        """
        self.y_pred = self.model.predict(self.X_test).reshape(-1, 1)
        return self.y_pred
    
    def evaluate_score(self):
        """
        Calculate the mean absolute error, root mean squared error, and R^2 score of the model

        returns:
            mae_error: Mean Absolute Error
            rmse_error: Root Mean Squared Error
            r2_error: R^2 Score
        """
        self.mae_error, self.rmse_error, self.r2_error, self.y_pred_orignal, self.y_test_original = evaluate_model(self.y_test, self.y_pred, self.y_scaler)

        print(f'Mean Absolute Error: {self.mae_error:.4f}\nRoot Mean Squared Error: {self.rmse_error:.4f}\nR^2 Score: {self.r2_error:.4f}')

        print(f'--------------------------------- \n')
        print(f'Mean Absoulte Error Normalized: {mean_absolute_error(self.y_test, self.y_pred)}\n Root Mean Squared Error Normalized: {mean_squared_error(self.y_test, self.y_pred, squared=False)}\n')

        return self.mae_error, self.rmse_error, self.r2_error
    
    def model_confidence(self):
        """
        Calculate overall model confidence and confidence for each prediction
        """
        #Normalized Confidence Score
        confidence_dataframe_normalized, avg_confidence_normalized = evaluate_model_confidence(self.y_test, self.y_pred)
        
        #Original Confidence Score
        confidence_dataframe_orignal, avg_confidence_orignal = evaluate_model_confidence(self.y_test_original, self.y_pred_orignal)

        return confidence_dataframe_normalized, avg_confidence_normalized, confidence_dataframe_orignal, avg_confidence_orignal

    
    def feature_importance(self, importance_type='total_gain'):
        """
        Plot the feature importance of the model

        Args: 
            importance_type (str): Importance type of the features
        """
        feature_names = self.dataset[self.features].columns

        # Get feature importance from the model
        feature_importance = self.model.get_booster().get_score(importance_type=importance_type)

        # Create a DataFrame to hold feature names and their importance scores
        importance_df = pd.DataFrame({
            'Feature': [feature_names[int(i[1:])] for i in feature_importance.keys()],
            'Importance': list(feature_importance.values())
        })

        # Normalize the importance to get percentages
        importance_df['Importance (%)'] = (importance_df['Importance'] / importance_df['Importance'].sum()) * 100

        # Sort the DataFrame by 'Importance' in descending order
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        # Plot the feature importance as a horizontal bar plot with the highest value at the top
        ax = importance_df.plot(kind="barh", x="Feature", y="Importance (%)", legend=False, figsize=(8, 6), color='maroon')

        # Invert the y-axis to have the highest importance at the top
        plt.gca().invert_yaxis()

        # Label each bar with the percentage value
        for index, value in enumerate(importance_df['Importance (%)']):
            plt.text(value, index, f'{value:.2f}%', va='center')

        plt.xlabel("Importance (%)")
        plt.ylabel("Feature")
        plt.title("Feature Importance")
        plt.show()










    
