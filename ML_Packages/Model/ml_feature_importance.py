import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




def plt_feature_importance(model, feature_names):

    """
    Plot the feature importance of a model.

    Args:
        model: The trained model
        feature_names: List of feature names
        
    Returns:
        None
    """
    
    # Extract feature importance from the random forest regressor
    feature_importance = model.feature_importances_

    #Calculate the percentage of importance
    feature_importance_percentage = 100.0 * (feature_importance / feature_importance.sum())

    # Create a DataFrame with features and their importance scores
    feature_importance_df = {
        'Feature': feature_names,
        'Importance': feature_importance_percentage
    }

    feature_importance_df = pd.DataFrame(feature_importance_df)
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot the feature importances
    plt.figure(figsize=(8, 4))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, color='maroon')

    for index, value in enumerate(feature_importance_df['Importance']):
        plt.text(value, index, f'{value:.2f}%', color='black', ha="left", va="center")

    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.show()   