import os
import time
import shutil
import joblib
import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from aux_functions.excel_modification import *
from aux_functions.imputation import *
from aux_functions.train_ml_model import *
from aux_functions.print_automatic import *

# Constants for temperature
ten_degrees= '10_deg'

# Constants for optimization type
dec = 'lin'
log = 'log'

# Configuration: current temperature and optimization type
actual_temp = ten_degrees
actual_opt = dec

# File paths and naming conventions
base_path = 'excel_spreadsheets/'
sample_name = 'asphalt'
file_name = base_path + sample_name + '_' + actual_temp

# File names for each processing step
xls_separated = file_name + '_separated.xlsx'
xls_imputed = file_name + '_imputed.xlsx'
xls_reduced = file_name + '_reduced.xlsx'
xls_rounded = file_name + '_rounded.xlsx'
xls_filtered = file_name + '_filtered.xlsx'
xls_divided = file_name + '_divided.xlsx'
xls_predicted = file_name + '_predicted.xlsx'

# Step 1: Data Preparation and Preprocessing
separate_excel_data(file_name + '.xlsx', xls_separated)
process_asfalt_binder(xls_separated, xls_imputed)
drop_rows_with_missing_values(xls_imputed, xls_reduced)
filter_outliers_by_zscore(xls_reduced, xls_filtered, threshold=3)
split_data_into_train_test_validate(xls_filtered, xls_divided)

# Step 2: Load Training, Test, and Validation Data
df_train_imputed = pd.read_excel(xls_divided, sheet_name='fatigue data - train')
df_test_imputed = pd.read_excel(xls_divided, sheet_name='fatigue data - test')
df_validate_imputed = pd.read_excel(xls_divided, sheet_name='fatigue data - validate')

# Step 3: Load Hyperparameters
hy = pd.read_excel('excel_spreadsheets/hyperparameters.xlsx', sheet_name='log')

# Prepare hyperparameter grid
hyperparameter_grid = []
for index, row in hy.iterrows():
    hyperparams = {'num_layers': row['num_layers'],
                   'dense': row['dense'],
                   'optimizer': row['optimizer'],
                   'activation_function': row['activation_function'],
                   'n_epochs': row['n_epochs']}
    hyperparameter_grid.append(hyperparams)

results_df = pd.DataFrame(
    columns=['num_layers', 'dense', 'optimizer', 'activation_function', 'n_epochs', 'min_train_loss', 'best_train_epoch', 'min_val_loss',
             'best_val_epoch', 'rmse_test', 'rmse_all', 'r2_test', 'r2_all'])
results_list = []
df = pd.read_excel(xls_filtered)
predictions = []

# Step 4: Data Preprocessing for Model Training
input_columns = ['Binder content (%)',
                 'Initial strain (µɛ)',
                 'Air Voids (%)',
                 'Penetrace',
                 'PMB', 'SIL',
                 'Initial stiffness (Mpa)'
                 ]

output_column = 'Number of cycles (times)'

# Step 5: Cross Validation
n_splits = 3
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

results = []
all_predictions = []
kf_index = 0

for train_index, test_index in kf.split(df):
    kf_index += 1
    df_train, df_test = df.iloc[train_index], df.iloc[test_index]

    # Save training and testing data for the current fold
    excel_train_test = file_name + f'_{kf_index}_train_test.xlsx'
    with pd.ExcelWriter(excel_train_test) as writer:
        df_train.to_excel(writer, sheet_name='Train', index=False)
        df_test.to_excel(writer, sheet_name='Test', index=False)

    # Scale input features
    scaler = MinMaxScaler()
    train_inputs = scaler.fit_transform(df_train[input_columns])
    test_inputs = scaler.transform(df_test[input_columns])

    # Extract target values
    train_outputs = df_train[output_column].values
    test_outputs = df_test[output_column].values

    # Split training data further into training and validation sets
    train_inputs, validate_inputs = train_test_split(train_inputs, test_size=0.3, random_state=42)
    train_outputs, validate_outputs = train_test_split(train_outputs, test_size=0.3, random_state=42)

    iter_num = 1

    # Step 6: Iterate Over Each Hyperparameter Set and Train Model
    for params in hyperparameter_grid:
        num_layers = params['num_layers']
        dense = params['dense']
        optimizer = params['optimizer']
        activation_function = params['activation_function']
        n_epochs = int(params['n_epochs'])

        # Log the iteration and hyperparameter set being processed
        print(f"Iteration: {iter_num}, K-Fold: {kf_index}, Layers: {num_layers}, Neurons: {dense}, Optimizer: "
              f"{optimizer}, Activation: {activation_function}, Epochs: {n_epochs}")
        iter_num += 1

        # Start the training process and measure time
        start_time = time.time()
        print(time.strftime("%H:%M:%S"))

        # Train the ANN model with the current set of hyperparameters and get the performance results
        dir_name, train_predictions, test_predictions, validate_predictions, train_outputs, test_outputs, \
            validate_outputs, min_train_loss, best_train_epoch, min_val_loss, best_val_epoch, model = \
            train_ann_dec_cv(
                kf_index=kf_index,
                actual_opt=actual_opt,
                actual_temp=actual_temp,
                train_inputs=train_inputs,
                test_inputs=test_inputs,
                validate_inputs=validate_inputs,
                train_outputs=train_outputs,
                test_outputs=test_outputs,
                validate_outputs=validate_outputs,
                num_layers=num_layers,
                n_epochs=n_epochs,
                dense=dense,
                optimizer=optimizer,
                activation_function=activation_function
            )

        # Step 7: Add Predictions to Excel
        if 'Predicted Cycles' not in df_test.columns:
            df_test['Predicted Cycles'] = test_predictions
        with pd.ExcelWriter(excel_train_test, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            df_test.to_excel(writer, sheet_name='Test', index=False)
        shutil.copy(excel_train_test,os.path.join(dir_name, 'asphalt_samples_both_temp_predicted.xlsx'))

        # Step 8: Generate Performance Reports
        ann_performance_report_separatly_dec(dir_name, train_outputs, train_predictions, test_outputs, test_predictions,
                                             validate_outputs, validate_predictions)

        # Step 9: Calculate Metrics
        rmse_test = np.sqrt(mean_squared_error(test_outputs, test_predictions))
        r2_test = r2_score(test_outputs, test_predictions)
        rmsle_test = rmsle_calculation(test_outputs, test_predictions)

        # Step 10: Generate Performance Reports for All Data
        ann_performance_report_all_dec(dir_name, train_outputs, train_predictions, test_outputs, test_predictions,
                                       validate_outputs, validate_predictions)
        all_outputs = np.concatenate([train_outputs, test_outputs, validate_outputs])
        all_predictions = np.concatenate([train_predictions, test_predictions, validate_predictions])

        # Step 11: Calculate Metrics
        rmse_all = np.sqrt(mean_squared_error(all_outputs, all_predictions))
        r2_all = r2_score(all_outputs, all_predictions)
        rmsle_all = rmsle_calculation(all_outputs, all_predictions)


        new_file_path = os.path.join(dir_name, os.path.basename(xls_separated))
        shutil.copy(xls_separated, new_file_path)
        new_file_path = os.path.join(dir_name, os.path.basename(xls_filtered))
        shutil.copy(xls_filtered, new_file_path)
        new_file_path = os.path.join(dir_name, os.path.basename(xls_divided))
        shutil.copy(xls_divided, new_file_path)
        new_file_path = os.path.join(dir_name, os.path.basename(xls_reduced))
        shutil.copy(xls_reduced, new_file_path)

        # Step 12: Store Results for Current Hyperparameter Set
        results_dict = {
            'fold_num': kf_index,
            'loss_optimization': actual_opt,
            'dataset_temp': actual_temp,
            'num_layers': num_layers,
            'dense': dense,
            'optimizer': optimizer,
            'activation_function': activation_function,
            'n_epochs': n_epochs,
            'min_train_loss': min_train_loss,
            'best_train_epoch': best_train_epoch,
            'min_val_loss': min_val_loss,
            'best_val_epoch': best_val_epoch,
            'rmse_test': rmse_test,
            'rmsle_test': rmsle_test,
            'r2_test': r2_test,
            'rmse_all': rmse_all,
            'rmsle_all': rmsle_all,
            'r2_all': r2_all
        }

        results_list.append(results_dict)

        # Save Aggregated Results to an Excel Spreadsheet
        temp_df = pd.DataFrame(results_list)
        temp_df.to_excel('excel_spreadsheets/' + actual_opt + actual_temp + '_hyperparameter_tuning_results.xlsx',
                         index=False)

        # Log the iteration's training duration
        end_time = time.time()
        training_duration = end_time - start_time
        print(f"Training duration for set {iter_num}: {training_duration:.0f} seconds")

# Step 13: Final Export of All Results
results_df = pd.DataFrame(results_list)
