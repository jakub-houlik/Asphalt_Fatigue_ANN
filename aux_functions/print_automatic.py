import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score



def rmsle_calculation(y_true, y_pred):
    """
    Calculate the Root Mean Squared Logarithmic Error.

    Args:
    - y_true (array-like): True values.
    - y_pred (array-like): Predicted values.

    Returns:
    - float: The RMSLE value.
    """
    log_true = np.log1p(y_true)  # log1p is used for better numerical stability with small values
    log_pred = np.log1p(y_pred)
    squared_log_error = np.square(log_pred - log_true)
    mean_squared_log_error = np.mean(squared_log_error)
    return np.sqrt(mean_squared_log_error)

def plot_ann_performance_dec(y_true, y_pred, dataset_name, color, marker, s=20):
    """
    Plot the performance of an ANN model on given data.

    Args:
    - y_true (array-like): True output values.
    - y_pred (array-like): Predicted output values.
    - dataset_name (str): Name of the dataset (e.g., 'Train', 'Test').
    - color (str): Color for the scatter plot.
    - marker (str): Marker style for the scatter plot.
    - s (int): Size of the markers.
    """

    y_true, y_pred = np.squeeze(np.array(y_true)), np.squeeze(np.array(y_pred))

    min_value, max_value = np.min([y_true, y_pred]), np.max([y_true, y_pred])

    plt.scatter(y_true, y_pred, alpha=0.5, color=color, label=dataset_name, marker=marker, s=s)
    plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
    plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter())
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc='upper left')
    plt.xlabel(r'$N_{\mathrm{f}}^{\mathrm{true}}$')
    plt.ylabel(r'$N_{\mathrm{f}}^{\mathrm{predicted}}$')
    plt.plot([min_value, max_value], [min_value, max_value], color='black')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis([min_value, max_value, min_value, max_value])
    plt.grid(True, linestyle='--', alpha=0.7)

def ann_performance_report_separatly_dec(dir_name, train_outputs, train_predictions, test_outputs, test_predictions,
                                     validate_outputs, validate_predictions):
    """
    Generate and save plots for the performance of an ANN model on train, test, and validate datasets.

    Args:
    - train_outputs, train_predictions, test_outputs, test_predictions, validate_outputs, validate_predictions
    (array-like): True outputs and predictions.
    """

    datasets = {
        "Test": (test_outputs, test_predictions, 'blue', 'o'),
        "Train": (train_outputs, train_predictions, 'green', 'x'),
        "Validate": (validate_outputs, validate_predictions, 'red', '*')
    }

    for name, (outputs, predictions, color, marker) in datasets.items():
        plot_ann_performance_dec(outputs, predictions, name, color, marker)
        rmse = np.sqrt(mean_squared_error(outputs, predictions))
        r2 = r2_score(outputs, predictions)
        rmsle = rmsle_calculation(outputs, predictions)

        #plt.title(f'True vs predicted - {name} data')
        plt.text(0.05, 0.90, f'$R^2 = {r2:.2f}$\n$RMSE = {rmse:.0f}$\n$RMSLE = {rmsle:.4f}$',
                 transform=plt.gca().transAxes, verticalalignment='top')
        plt.savefig(f"{dir_name}/{name.lower()}.pdf", format='pdf')
        plt.savefig(f"{dir_name}/{name.lower()}.jpg", format='jpg')
        plt.close()

def ann_performance_report_all_dec(dir_name, train_outputs, train_predictions, test_outputs, test_predictions, validate_outputs,
                                    validate_predictions):
    """
    Generate and save plots for the performance of an ANN model on train, test, and validate datasets.

    Args:
    - train_outputs, train_predictions, test_outputs, test_predictions, validate_outputs, validate_predictions
    (array-like): True outputs and predictions.
    """

    datasets = {
        "Test": (test_outputs, test_predictions, 'blue', 'o'),
        "Train": (train_outputs, train_predictions, 'green', 'x'),
        "Validate": (validate_outputs, validate_predictions, 'red', '*')
    }

    # Combine all data together
    all_outputs = np.concatenate([train_outputs, test_outputs, validate_outputs])
    all_predictions = np.concatenate([train_predictions, test_predictions, validate_predictions])

    # Plot all data separately
    plot_ann_performance_dec(train_outputs, train_predictions, 'Train', 'green', 'x', 20)
    plot_ann_performance_dec(test_outputs, test_predictions, 'Test', 'blue', 'o', 20)
    plot_ann_performance_dec(validate_outputs, validate_predictions, 'Validate', 'red', '*', 20)

    # Calculate accuracy metrics
    rmse_all = np.sqrt(mean_squared_error(all_outputs, all_predictions))
    r2_all = r2_score(all_outputs, all_predictions)
    rmsle = rmsle_calculation(all_outputs, all_predictions)

    plt.text(0.05, 0.78, f'$R^2 = {r2_all:.2f}$\n$RMSE = {rmse_all:.0f}$\n$RMSLE = {rmsle:.4f}$',
             transform=plt.gca().transAxes, verticalalignment='top')
    plt.savefig(f"{dir_name}/all.pdf", format='pdf')
    plt.savefig(f"{dir_name}/all.jpg", format='jpg')
    plt.close()