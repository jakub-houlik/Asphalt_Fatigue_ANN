import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, Nadam, RMSprop
from keras import initializers
from SALib.sample import saltelli
from SALib.analyze import sobol
from keras.callbacks import Callback
import random
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

get_custom_objects().update({'mish': Activation(mish)})


def preprocess_data(data_df, columns):
    scaler = MinMaxScaler()
    inputs = scaler.fit_transform(data_df[columns])
    outputs = data_df['Number of cycles (times)'].values
    return inputs, outputs

class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, start_epoch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        # Check if the current epoch is greater than or equal to the start_epoch
        if epoch >= self.start_epoch:
            super().on_epoch_end(epoch, logs)

class EpochLogger(Callback):
    def __init__(self, log_file='training_log.xlsx'):
        super().__init__()
        self.log_file = log_file
        self.logs = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.logs.append({'epoch': epoch + 1, 'train_loss': logs.get('loss'), 'val_loss': logs.get('val_loss')})

    def on_train_end(self, logs=None):
        log_df = pd.DataFrame(self.logs)
        log_df.to_excel(self.log_file, index=False)

def train_ann_dec_cv(kf_index, actual_opt, actual_temp, train_inputs, test_inputs,
                     validate_inputs, train_outputs, test_outputs, validate_outputs,
                     num_layers, n_epochs, dense, activation_function, optimizer,
                     plot_iterations=True):
    """
    Train a sequential ANN model and return predictions for train, test, and validate datasets.

    Args:
    - sample_name (str): Name of the sample used for directory naming.
    - train_df, test_df, validate_df (pd.DataFrame): Datasets for training, testing, and validation.
    - n_epochs (int): Number of epochs for training.
    - dense (int): Number of dense neurons in ANN layers.
    - plot_iterations (bool): If True, plots the training progress.

    Returns:
    - Tuple containing true outputs and predictions for train, test, and validate datasets.
    """

    input_columns = ['Binder content (%)', 'Air Voids (%)', 'Initial strain (µɛ)', 'Penetrace', 'PMB', 'SIL']
    # Extract and normalize data
    train_inputs = train_inputs
    test_inputs = test_inputs
    validate_inputs = validate_inputs
    train_outputs = train_outputs
    test_outputs = test_outputs
    validate_outputs = validate_outputs

    # Define the model
    model = Sequential()

    # Add input layer
    model.add(Dense(dense, activation=activation_function, input_shape=(train_inputs.shape[1],)))

    # Add hidden layers based on num_layers
    for _ in range(num_layers - 1):
        model.add(Dense(dense, activation=activation_function))

        # Add output layer
    model.add(Dense(1))

    # Choose the optimizer
    if optimizer == 'adam':
        opt = Adam()
    elif optimizer == 'adamw':
        opt = Nadam()
    elif optimizer == 'rmsprop':
        opt = RMSprop()
    elif optimizer == 'sgd':
        opt = RMSprop()
    else:
        raise ValueError("Unsupported optimizer")

    model.compile(optimizer=opt, loss='mean_squared_error')

    # Save the model and plot
    dir_name = f'output/{actual_opt}/{actual_temp}/{kf_index}kf_{len(model.layers) - 1}layers_{dense}dense_{n_epochs}epochs_{optimizer}_optimizer_{activation_function}_activation'
    os.makedirs(dir_name, exist_ok=True)

    # Define the EpochLogger callback
    epoch_logger = EpochLogger(log_file=f'{dir_name}/epoch_logs.xlsx')

    # Define ModelCheckpoint callback
    start_epoch = int(n_epochs/10)
    model_checkpoint = CustomModelCheckpoint(start_epoch, filepath=f'{dir_name}/best_model.h5', save_best_only=True, monitor='val_loss',
                                       mode='min', verbose=0)
    # Train the model
    history = model.fit(train_inputs, train_outputs, epochs=n_epochs, verbose=0,
                        validation_data=(validate_inputs, validate_outputs),
                        callbacks=[model_checkpoint, epoch_logger], shuffle=True)

    train_predictions = model.predict(train_inputs).flatten()
    test_predictions = model.predict(test_inputs).flatten()
    validate_predictions = model.predict(validate_inputs).flatten()

    # Replace negative values with 5000 and limit for 2 mil
    train_predictions = np.clip(train_predictions, a_min=0, a_max=4000000)
    test_predictions = np.clip(test_predictions, a_min=0, a_max=4000000)
    validate_predictions = np.clip(validate_predictions, a_min=0, a_max=4000000)

    # Find the epoch number with the minimum validation loss
    val_loss_after_start = history.history['val_loss'][start_epoch:]
    min_val_loss = min(val_loss_after_start)
    best_val_epoch = val_loss_after_start.index(min_val_loss) + 1 + start_epoch

    train_loss_after_start = history.history['loss'][start_epoch:]
    min_train_loss = min(train_loss_after_start)
    best_train_epoch = train_loss_after_start.index(min_train_loss) + 1 + start_epoch

    if plot_iterations:
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.plot(history.history['loss'], label="Train")
        plt.plot(history.history['val_loss'], label="Validation")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        # Setting yx-axis limits
        epoch_range = int(n_epochs * 0.1)
        max_loss = max(max(history.history['loss'][epoch_range:]), max(history.history['val_loss'][epoch_range:]))
        min_loss = min(min(history.history['loss'][epoch_range:]), min(history.history['val_loss'][epoch_range:]))
        plt.ylim(min_loss - 0.1 * (max_loss - min_loss), max_loss + 0.1 * (max_loss - min_loss))
        plt.xlim((n_epochs * 0.1), n_epochs)

        # Adding entries to the legend
        train_legend = mlines.Line2D([], [], color='C0', marker='None',
                                     markersize=10,
                                     label=f'Min Train Loss: {min_train_loss:.2e}\nEpoch: {best_train_epoch}')
        val_legend = mlines.Line2D([], [], color='C1', marker='None',
                                   markersize=10, label=f'Min Val Loss: {min_val_loss:.2e}\nEpoch: {best_val_epoch}')
        plt.legend(handles=[train_legend, val_legend])
        plt.savefig(f"{dir_name}/iteration_progress.pdf")
        plt.savefig(f"{dir_name}/iteration_progress.jpg")
        plt.close()
    # Load the best model weights
    model.load_weights(f'{dir_name}/best_model.h5')

    # Results dictionary
    '''    results = {
        "train_predictions": int(train_predictions),
        "test_predictions": int(test_predictions),
        "validate_predictions": int(validate_predictions),
        "train_outputs": int(train_outputs),
        "test_outputs": int(test_outputs),
        "validate_outputs": int(validate_outputs),

        "min_train_loss": int(min_train_loss),
        "best_train_epoch": int(best_train_epoch),
        "min_val_loss": int(min_val_loss),
        "best_val_epoch": int(best_val_epoch)
    }'''

    return dir_name, train_predictions, test_predictions, validate_predictions, train_outputs, test_outputs, \
        validate_outputs, min_train_loss, best_train_epoch, min_val_loss, best_val_epoch, model

def train_ann_log_cv(kf_index, actual_opt, actual_temp, train_inputs, test_inputs,
                     validate_inputs, train_outputs, test_outputs, validate_outputs,
                     num_layers, n_epochs, dense, activation_function, optimizer,
                     plot_iterations=True):
    """
    Train a sequential ANN model and return predictions for train, test, and validate datasets.

    Args:
    - sample_name (str): Name of the sample used for directory naming.
    - train_df, test_df, validate_df (pd.DataFrame): Datasets for training, testing, and validation.
    - n_epochs (int): Number of epochs for training.
    - dense (int): Number of dense neurons in ANN layers.
    - plot_iterations (bool): If True, plots the training progress.

    Returns:
    - Tuple containing true outputs and predictions for train, test, and validate datasets.
    """

    input_columns = ['Binder content (%)', 'Air Voids (%)', 'Initial strain (µɛ)']
    # Extract and normalize data
    train_inputs = train_inputs
    test_inputs = test_inputs
    validate_inputs = validate_inputs
    train_outputs = train_outputs
    test_outputs = test_outputs
    validate_outputs = validate_outputs

    # Define the model
    model = Sequential()

    # Add input layer
    model.add(Dense(dense, activation=activation_function, kernel_initializer='he_normal', input_shape=(train_inputs.shape[1],)))

    # Add hidden layers based on num_layers
    for _ in range(num_layers - 1):
        model.add(Dense(dense, activation=activation_function))

        # Add output layer
    model.add(Dense(1))

    # Choose the optimizer
    if optimizer == 'adam':
        opt = Adam()
    elif optimizer == 'adamw':
        opt = Nadam()
    elif optimizer == 'rmsprop':
        opt = RMSprop()
    elif optimizer == 'sgd':
        opt = RMSprop()
    else:
        raise ValueError("Unsupported optimizer")

    model.compile(optimizer=opt, loss='mean_squared_logarithmic_error')

    # Save the model and plot
    dir_name = f'output/{actual_opt}/{actual_temp}/{kf_index}kf_{len(model.layers) - 1}layers_{dense}dense_{n_epochs}epochs_{optimizer}_optimizer_{activation_function}_activation'
    os.makedirs(dir_name, exist_ok=True)

    # Define the EpochLogger callback
    epoch_logger = EpochLogger(log_file=f'{dir_name}/epoch_logs.xlsx')

    # Define ModelCheckpoint callback
    start_epoch = int(n_epochs/10)
    model_checkpoint = CustomModelCheckpoint(start_epoch, filepath=f'{dir_name}/best_model.h5', save_best_only=True, monitor='val_loss',
                                       mode='min', verbose=0)
    # Train the model
    history = model.fit(train_inputs, train_outputs, epochs=n_epochs, verbose=0,
                        validation_data=(validate_inputs, validate_outputs),
                        callbacks=[model_checkpoint, epoch_logger], shuffle=True)

    train_predictions = model.predict(train_inputs).flatten()
    test_predictions = model.predict(test_inputs).flatten()
    validate_predictions = model.predict(validate_inputs).flatten()

    # Replace negative values with 5000 and limit for 2 mil
    train_predictions = np.clip(train_predictions, a_min=0, a_max=4000000)
    test_predictions = np.clip(test_predictions, a_min=0, a_max=4000000)
    validate_predictions = np.clip(validate_predictions, a_min=0, a_max=4000000)

    # Find the epoch number with the minimum validation loss
    val_loss_after_start = history.history['val_loss'][start_epoch:]
    min_val_loss = min(val_loss_after_start)
    best_val_epoch = val_loss_after_start.index(min_val_loss) + 1 + start_epoch

    train_loss_after_start = history.history['loss'][start_epoch:]
    min_train_loss = min(train_loss_after_start)
    best_train_epoch = train_loss_after_start.index(min_train_loss) + 1 + start_epoch

    if plot_iterations:
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.plot(history.history['loss'], label="Train")
        plt.plot(history.history['val_loss'], label="Validation")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        # Setting yx-axis limits
        epoch_range = int(n_epochs * 0.1)
        max_loss = max(max(history.history['loss'][epoch_range:]), max(history.history['val_loss'][epoch_range:]))
        min_loss = min(min(history.history['loss'][epoch_range:]), min(history.history['val_loss'][epoch_range:]))
        plt.ylim(min_loss - 0.1 * (max_loss - min_loss), max_loss + 0.1 * (max_loss - min_loss))
        plt.xlim((n_epochs * 0.1), n_epochs)

        # Adding entries to the legend
        train_legend = mlines.Line2D([], [], color='C0', marker='None',
                                     markersize=10,
                                     label=f'Min Train Loss: {min_train_loss:.2e}\nEpoch: {best_train_epoch}')
        val_legend = mlines.Line2D([], [], color='C1', marker='None',
                                   markersize=10, label=f'Min Val Loss: {min_val_loss:.2e}\nEpoch: {best_val_epoch}')
        plt.legend(handles=[train_legend, val_legend])
        plt.savefig(f"{dir_name}/iteration_progress.pdf")
        plt.savefig(f"{dir_name}/iteration_progress.jpg")
        plt.close()
    # Load the best model weights
    model.load_weights(f'{dir_name}/best_model.h5')

    # Results dictionary
    '''    results = {
        "train_predictions": int(train_predictions),
        "test_predictions": int(test_predictions),
        "validate_predictions": int(validate_predictions),
        "train_outputs": int(train_outputs),
        "test_outputs": int(test_outputs),
        "validate_outputs": int(validate_outputs),

        "min_train_loss": int(min_train_loss),
        "best_train_epoch": int(best_train_epoch),
        "min_val_loss": int(min_val_loss),
        "best_val_epoch": int(best_val_epoch)
    }'''

    return dir_name, train_predictions, test_predictions, validate_predictions, train_outputs, test_outputs, \
        validate_outputs, min_train_loss, best_train_epoch, min_val_loss, best_val_epoch, model

def reset_random_seeds(seed_value=42):
   os.environ['PYTHONHASHSEED']=str(seed_value)
   tf.random.set_seed(seed_value)
   np.random.seed(seed_value)
   random.seed(seed_value)

def train_ann_log(actual_opt, actual_temp, train_inputs, test_inputs,
                     validate_inputs, train_outputs, test_outputs, validate_outputs,
                     num_layers, n_epochs, dense, activation_function, optimizer,
                     plot_iterations=True):
    """
    Train a sequential ANN model and return predictions for train, test, and validate datasets.

    Args:
    - sample_name (str): Name of the sample used for directory naming.
    - train_df, test_df, validate_df (pd.DataFrame): Datasets for training, testing, and validation.
    - n_epochs (int): Number of epochs for training.
    - dense (int): Number of dense neurons in ANN layers.
    - plot_iterations (bool): If True, plots the training progress.

    Returns:
    - Tuple containing true outputs and predictions for train, test, and validate datasets.
    """

    input_columns = ['Binder content (%)', 'Air Voids (%)', 'Initial strain (µɛ)']
    # Extract and normalize data
    train_inputs = train_inputs
    test_inputs = test_inputs
    validate_inputs = validate_inputs
    train_outputs = train_outputs
    test_outputs = test_outputs
    validate_outputs = validate_outputs

    # Define the model
    model = Sequential()

    # Add input layer
    model.add(Dense(dense, activation=activation_function, kernel_initializer='he_normal', input_shape=(train_inputs.shape[1],)))

    # Add hidden layers based on num_layers
    for _ in range(num_layers - 1):
        model.add(Dense(dense, activation=activation_function))

        # Add output layer
    model.add(Dense(1))

    # Choose the optimizer
    if optimizer == 'adam':
        opt = Adam()
    elif optimizer == 'adamw':
        opt = Nadam()
    elif optimizer == 'rmsprop':
        opt = RMSprop()
    elif optimizer == 'sgd':
        opt = RMSprop()
    else:
        raise ValueError("Unsupported optimizer")

    model.compile(optimizer=opt, loss='mean_squared_logarithmic_error')

    # Save the model and plot
    dir_name = f'output/{actual_opt}/{actual_temp}/{len(model.layers) - 1}layers_{dense}dense_{n_epochs}epochs_{optimizer}_optimizer_{activation_function}_activation'
    os.makedirs(dir_name, exist_ok=True)

    # Define the EpochLogger callback
    epoch_logger = EpochLogger(log_file=f'{dir_name}/epoch_logs.xlsx')

    # Define ModelCheckpoint callback
    start_epoch = int(n_epochs/100)
    model_checkpoint = CustomModelCheckpoint(start_epoch, filepath=f'{dir_name}/best_model.h5', save_best_only=True, monitor='val_loss',
                                       mode='min', verbose=0)
    # Train the model
    history = model.fit(train_inputs, train_outputs, epochs=n_epochs, verbose=0,
                        validation_data=(validate_inputs, validate_outputs),
                        callbacks=[model_checkpoint, epoch_logger], shuffle=True)

    train_predictions = model.predict(train_inputs).flatten()
    test_predictions = model.predict(test_inputs).flatten()
    validate_predictions = model.predict(validate_inputs).flatten()

    # Replace negative values with 5000 and limit for 2 mil
    train_predictions = np.clip(train_predictions, a_min=0, a_max=4000000)
    test_predictions = np.clip(test_predictions, a_min=0, a_max=4000000)
    validate_predictions = np.clip(validate_predictions, a_min=0, a_max=4000000)

    # Find the epoch number with the minimum validation loss
    val_loss_after_start = history.history['val_loss'][start_epoch:]
    min_val_loss = min(val_loss_after_start)
    best_val_epoch = val_loss_after_start.index(min_val_loss) + 1 + start_epoch

    train_loss_after_start = history.history['loss'][start_epoch:]
    min_train_loss = min(train_loss_after_start)
    best_train_epoch = train_loss_after_start.index(min_train_loss) + 1 + start_epoch

    # Plot the model architecture
    #plot_model(model, to_file=f'{dir_name}/model_architecture.pdf', show_shapes=True, show_layer_names=True)
    #plot_model(model, to_file=f'{dir_name}/model_architecture.jpg', show_shapes=True, show_layer_names=True)

    if plot_iterations:
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.plot(history.history['loss'], label="Train")
        plt.plot(history.history['val_loss'], label="Validation")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        # Setting yx-axis limits
        epoch_range = int(n_epochs * 0.1)
        max_loss = max(max(history.history['loss'][epoch_range:]), max(history.history['val_loss'][epoch_range:]))
        min_loss = min(min(history.history['loss'][epoch_range:]), min(history.history['val_loss'][epoch_range:]))
        plt.ylim(min_loss - 0.1 * (max_loss - min_loss), max_loss + 0.1 * (max_loss - min_loss))
        plt.xlim((n_epochs * 0.1), n_epochs)

        # Adding entries to the legend
        train_legend = mlines.Line2D([], [], color='C0', marker='None',
                                     markersize=10,
                                     label=f'Min Train Loss: {min_train_loss:.2e}\nEpoch: {best_train_epoch}')
        val_legend = mlines.Line2D([], [], color='C1', marker='None',
                                   markersize=10, label=f'Min Val Loss: {min_val_loss:.2e}\nEpoch: {best_val_epoch}')
        plt.legend(handles=[train_legend, val_legend])
        plt.savefig(f"{dir_name}/iteration_progress.pdf")
        plt.savefig(f"{dir_name}/iteration_progress.jpg")
        plt.close()
    # Load the best model weights
    model.load_weights(f'{dir_name}/best_model.h5')

    # Results dictionary
    '''    results = {
        "train_predictions": int(train_predictions),
        "test_predictions": int(test_predictions),
        "validate_predictions": int(validate_predictions),
        "train_outputs": int(train_outputs),
        "test_outputs": int(test_outputs),
        "validate_outputs": int(validate_outputs),

        "min_train_loss": int(min_train_loss),
        "best_train_epoch": int(best_train_epoch),
        "min_val_loss": int(min_val_loss),
        "best_val_epoch": int(best_val_epoch)
    }'''

    return dir_name, train_predictions, test_predictions, validate_predictions, train_outputs, test_outputs, \
        validate_outputs, min_train_loss, best_train_epoch, min_val_loss, best_val_epoch, model

def train_ann_lin(actual_opt, actual_temp, train_inputs, test_inputs,
                     validate_inputs, train_outputs, test_outputs, validate_outputs,
                     num_layers, n_epochs, dense, activation_function, optimizer,
                     plot_iterations=True):
    """
    Train a sequential ANN model and return predictions for train, test, and validate datasets.

    Args:
    - sample_name (str): Name of the sample used for directory naming.
    - train_df, test_df, validate_df (pd.DataFrame): Datasets for training, testing, and validation.
    - n_epochs (int): Number of epochs for training.
    - dense (int): Number of dense neurons in ANN layers.
    - plot_iterations (bool): If True, plots the training progress.

    Returns:
    - Tuple containing true outputs and predictions for train, test, and validate datasets.
    """

    input_columns = ['Binder content (%)', 'Air Voids (%)', 'Initial strain (µɛ)']
    # Extract and normalize data
    train_inputs = train_inputs
    test_inputs = test_inputs
    validate_inputs = validate_inputs
    train_outputs = train_outputs
    test_outputs = test_outputs
    validate_outputs = validate_outputs

    # Define the model
    model = Sequential()

    # Add input layer
    model.add(Dense(dense, activation=activation_function, kernel_initializer='he_normal', input_shape=(train_inputs.shape[1],)))

    # Add hidden layers based on num_layers
    for _ in range(num_layers - 1):
        model.add(Dense(dense, activation=activation_function))

        # Add output layer
    model.add(Dense(1))

    # Choose the optimizer
    if optimizer == 'adam':
        opt = Adam()
    elif optimizer == 'adamw':
        opt = Nadam()
    elif optimizer == 'rmsprop':
        opt = RMSprop()
    elif optimizer == 'sgd':
        opt = RMSprop()
    else:
        raise ValueError("Unsupported optimizer")

    model.compile(optimizer=opt, loss='mean_squared_error')

    # Save the model and plot
    dir_name = f'output/{actual_opt}/{actual_temp}/{len(model.layers) - 1}layers_{dense}dense_{n_epochs}epochs_{optimizer}_optimizer_{activation_function}_activation'
    os.makedirs(dir_name, exist_ok=True)

    # Define the EpochLogger callback
    epoch_logger = EpochLogger(log_file=f'{dir_name}/epoch_logs.xlsx')

    # Define ModelCheckpoint callback
    start_epoch = int(n_epochs/100)
    model_checkpoint = CustomModelCheckpoint(start_epoch, filepath=f'{dir_name}/best_model.h5', save_best_only=True, monitor='val_loss',
                                       mode='min', verbose=0)
    # Train the model
    history = model.fit(train_inputs, train_outputs, epochs=n_epochs, verbose=0,
                        validation_data=(validate_inputs, validate_outputs),
                        callbacks=[model_checkpoint, epoch_logger], shuffle=True)

    train_predictions = model.predict(train_inputs).flatten()
    test_predictions = model.predict(test_inputs).flatten()
    validate_predictions = model.predict(validate_inputs).flatten()

    # Replace negative values with 5000 and limit for 2 mil
    train_predictions = np.clip(train_predictions, a_min=0, a_max=4000000)
    test_predictions = np.clip(test_predictions, a_min=0, a_max=4000000)
    validate_predictions = np.clip(validate_predictions, a_min=0, a_max=4000000)

    # Find the epoch number with the minimum validation loss
    val_loss_after_start = history.history['val_loss'][start_epoch:]
    min_val_loss = min(val_loss_after_start)
    best_val_epoch = val_loss_after_start.index(min_val_loss) + 1 + start_epoch

    train_loss_after_start = history.history['loss'][start_epoch:]
    min_train_loss = min(train_loss_after_start)
    best_train_epoch = train_loss_after_start.index(min_train_loss) + 1 + start_epoch

    # Plot the model architecture
    #plot_model(model, to_file=f'{dir_name}/model_architecture.pdf', show_shapes=True, show_layer_names=True)
    #plot_model(model, to_file=f'{dir_name}/model_architecture.jpg', show_shapes=True, show_layer_names=True)

    if plot_iterations:
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.plot(history.history['loss'], label="Train")
        plt.plot(history.history['val_loss'], label="Validation")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        # Setting yx-axis limits
        epoch_range = int(n_epochs * 0.1)
        max_loss = max(max(history.history['loss'][epoch_range:]), max(history.history['val_loss'][epoch_range:]))
        min_loss = min(min(history.history['loss'][epoch_range:]), min(history.history['val_loss'][epoch_range:]))
        plt.ylim(min_loss - 0.1 * (max_loss - min_loss), max_loss + 0.1 * (max_loss - min_loss))
        plt.xlim((n_epochs * 0.1), n_epochs)

        # Adding entries to the legend
        train_legend = mlines.Line2D([], [], color='C0', marker='None',
                                     markersize=10,
                                     label=f'Min Train Loss: {min_train_loss:.2e}\nEpoch: {best_train_epoch}')
        val_legend = mlines.Line2D([], [], color='C1', marker='None',
                                   markersize=10, label=f'Min Val Loss: {min_val_loss:.2e}\nEpoch: {best_val_epoch}')
        plt.legend(handles=[train_legend, val_legend])
        plt.savefig(f"{dir_name}/iteration_progress.pdf")
        plt.savefig(f"{dir_name}/iteration_progress.jpg")
        plt.close()
    # Load the best model weights
    model.load_weights(f'{dir_name}/best_model.h5')

    # Results dictionary
    '''    results = {
        "train_predictions": int(train_predictions),
        "test_predictions": int(test_predictions),
        "validate_predictions": int(validate_predictions),
        "train_outputs": int(train_outputs),
        "test_outputs": int(test_outputs),
        "validate_outputs": int(validate_outputs),

        "min_train_loss": int(min_train_loss),
        "best_train_epoch": int(best_train_epoch),
        "min_val_loss": int(min_val_loss),
        "best_val_epoch": int(best_val_epoch)
    }'''

    return dir_name, train_predictions, test_predictions, validate_predictions, train_outputs, test_outputs, \
        validate_outputs, min_train_loss, best_train_epoch, min_val_loss, best_val_epoch, model