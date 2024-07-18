import matplotlib.pyplot as plt




def plot_loss_metrics(model_results):
    """
    Plots the loss, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) scores of a TensorFlow model.
    Args:
        model_results (object): The history object of the trained model.
    
    Returns:
        None
    """
    train_loss = model_results.history['loss']
    val_loss = model_results.history['val_loss']
    train_mae = model_results.history['mean_absolute_error']
    val_mae = model_results.history['val_mean_absolute_error']
    train_rmse = model_results.history['root_mean_squared_error']
    val_rmse = model_results.history['val_root_mean_squared_error']

    epochs = range(1, len(train_loss) + 1)

    # Create subplots with 1 row and 3 columns
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot loss scores
    axs[0].plot(epochs, train_loss, 'g', label='Training Loss')
    axs[0].plot(epochs, val_loss, 'b', label='Validation Loss')
    axs[0].set_title('Loss Scores')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot MAE scores
    axs[1].plot(epochs, train_mae, 'c', label='Training MAE')
    axs[1].plot(epochs, val_mae, 'm', label='Validation MAE')
    axs[1].set_title('MAE Scores')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Mean Absolute Error')
    axs[1].legend()

    # Plot RMSE scores
    axs[2].plot(epochs, train_rmse, 'y', label='Training RMSE')
    axs[2].plot(epochs, val_rmse, 'r', label='Validation RMSE')
    axs[2].set_title('RMSE Scores')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Root Mean Squared Error')
    axs[2].legend()

    # Show the plots
    plt.show()