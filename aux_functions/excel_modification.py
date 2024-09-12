import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
import re


def categorize_binder(df, column_name):
    """
    Categorizes the binder column into two categories: 'pmb' and 'ostatni'.

    This function performs the following steps:
    1. Creates a new column 'pmb' with a value of 1 if 'PMB' is in the cell, otherwise 0.
    2. Creates a new column 'ostatni' with a value of 1 if 'PMB' is not in the cell, otherwise 0.

    Args:
    - df (pd.DataFrame): The DataFrame containing the binder data.
    - column_name (str): The name of the column to categorize.

    Returns:
    - pd.DataFrame: The DataFrame with the new categorical columns added.
    """
    df['pmb'] = df[column_name].apply(lambda x: 1 if 'PMB' in str(x) else 0)
    df['sil'] = df[column_name].apply(lambda x: 0 if 'PMB' in str(x) else 1)
    return df


def process_and_save(input_file: str, sheet_name, output_file: str) -> list:
    """
    Processes an Excel file to categorize the 'Asfaltové pojivo' column and saves the result to a new file.

    This function performs the following steps:
    1. Reads the specified Excel file into a pandas DataFrame.
    2. Applies the categorize_binder function to categorize the binder column.
    3. Saves the processed DataFrame to a new Excel file.

    Args:
    - input_file (str): The path to the original Excel file.
    - output_file (str): The path to save the processed Excel file.
    - sheet_name (str, optional): The name of the sheet to read. If None, the first sheet is read.

    Returns:
    - list: The list of columns in the processed DataFrame.
    """

    # Read the Excel file
    df = pd.read_excel(input_file, sheet_name=sheet_name)

    # Apply the function to categorize the binder column
    df = categorize_binder(df, 'Binder type')

    # Write the processed data to a new Excel file
    df.to_excel(output_file, index=False)

    return df.columns.tolist()

def process_asfalt_binder(input_file: str, output_file: str):
    # Load the Excel file into a pandas DataFrame
    df = pd.read_excel(input_file)

    # Function to determine 'PMB or SIL'
    def determine_pmb_or_neat(value):
        value = str(value)
        if 'PMB' in value:
            return 'PMB'
        else:
            return 'SIL'

    # Function to calculate penetration average
    def calculate_penetration(value):
        value = str(value)
        match = re.search(r'(\d+)/(\d+)', value)
        if match:
            num1 = int(match.group(1))
            num2 = int(match.group(2))
            return (num1 + num2) / 2
        return None

    def create_binary_columns(df):
        df['PMB'] = df['PMB or SIL'].apply(lambda x: 1 if x == 'PMB' else 0)
        df['SIL'] = df['PMB or SIL'].apply(lambda x: 1 if x == 'SIL' else 0)
        return df

    # Create 'PMB or SIL' column
    df['PMB or SIL'] = df['Binder type'].apply(determine_pmb_or_neat)

    # Create 'Penetrace' column
    df['Penetrace'] = df['Binder type'].apply(calculate_penetration)

    # Apply the function to create binary columns
    df = create_binary_columns(df)

    # Write the modified DataFrame to a new Excel file
    df.to_excel(output_file, index=False)


def separate_excel_data(input_file: str, output_file: str) -> list:
    """
    Processes an Excel file to filter, clean, and separate relevant data.

    This function performs the following steps:
    1. Reads the specified Excel file into a pandas DataFrame.
    2. Cleans column headers by removing additional spaces.
    3. Filters rows containing specific keywords ('4PB') in any column, ignoring case sensitivity.
    4. Specifies and retains only desired columns related to academic or scientific data (e.g., 'Author', 'DOI').
    5. Writes the processed DataFrame to a new Excel file without row indices.
    6. Returns a list of the retained column names.

    Args:
    - input_file (str): The path to the original Excel file.
    - output_file (str): The path to save the separated Excel file.

    Returns:
    - list: The list of columns retained in the separated Excel file.
    """

    # Read the Excel file
    df = pd.read_excel(input_file)

    # Remove additional spaces from column headers
    df.columns = df.columns.str.strip()

    # Filter rows based on specific keywords in any of the columns
    keywords = ['4PB']
    df = df[df.apply(lambda row: row.astype(str).str.contains('|'.join(keywords), case=False).any(), axis=1)]

    # List of columns to retain
    columns_to_keep = [
        'Author', 'DOI', 'Binder type', 'Binder content (%)', 'Air Voids (%)', 'Initial strain (µɛ)',
        'Initial stiffness (Mpa)', 'Number of cycles (times)'
    ]

    # Retain only the desired columns
    df = df[columns_to_keep]

    # Write the cleaned data to a new Excel file
    df.to_excel(output_file, index=False)

    return columns_to_keep


def filter_outliers_by_zscore(input_file: str, output_file: str, threshold: float = 3.0) -> None:
    """
    Removes rows from an Excel dataset based on Z-scores.

    The function operates as follows:
    1. Loads the dataset from the specified Excel file into a pandas DataFrame.
    2. Calculates the Z-scores for all numeric columns in the DataFrame. The Z-score represents how many standard deviations away a value is from the mean of its column.
    3. Filters out any rows where the absolute value of the Z-score in any numeric column exceeds a specified threshold. The default threshold is set at 3.0, typically used to identify outliers in a dataset.
    4. Saves the DataFrame, now excluding the identified outliers, to a new Excel file specified by the output_file parameter, without including row indices.

    Args:
    - input_file (str): Path to the input Excel file.
    - output_file (str): Path where the filtered data will be saved.
    - threshold (float, optional): Z-score threshold to identify outliers. Default is 3.0.

    Returns:
    - None
    """

    # Load data
    df = pd.read_excel(input_file)

    # Calculate Z-scores for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number])
    z_scores = stats.zscore(numeric_cols)

    # Handle NaN values
    z_scores = np.nan_to_num(z_scores)

    # Filter rows based on Z-score threshold
    mask = (np.abs(z_scores) < threshold).all(axis=1)
    df_filtered = df[mask]

    # Save the filtered data
    df_filtered.to_excel(output_file, index=False)

def add_predictions_to_excel(input_file: str, output_file: str, train_preds, test_preds, validate_preds,
                             reshape_method=None) -> None:
    """
    Adds predictions to train, test, and validate datasets in an Excel file.

    The function performs the following actions:
    1. Opens the input Excel file to access its sheets without immediately loading all data into memory,
       optimizing for efficiency when dealing with large files.
    2. Maps the provided prediction values (for training, testing, and validation sets) to their
       corresponding sheet names within the Excel file. This mapping ensures that each dataset receives
       the correct predictions.
    3. Iterates over each dataset, loading the relevant sheet into a pandas DataFrame.
    4. If specified, reshapes the predictions according to the `reshape_method` parameter. The method can flatten
       nested lists or reshape arrays into a 1D format, ensuring the predictions align with the structure of the DataFrame.
    5. Adds the predictions as a new column named 'Predicted' to the DataFrame, effectively appending the predicted
       values to the original datasets.
    6. Saves the updated DataFrames, now including the predictions, back to a new Excel file, maintaining the original
       sheet structure but with the added predictions.
    7. The function does not return a value but saves the modified datasets to the specified output Excel file path.

    This process is particularly useful for appending model predictions directly to datasets, facilitating easier
    analysis and comparison of predicted versus actual values within the context of the original data.

    Args:
    - input_file (str): The path to the input Excel file that contains the datasets.
    - output_file (str): The path where the datasets with added predictions will be saved.
    - train_preds (list or array): Predicted values for the training set.
    - test_preds (list or array): Predicted values for the test set.
    - validate_preds (list or array): Predicted values for the validation set.
    - reshape_method (str, optional): Specifies how to reshape the predictions.
                                      'flatten' will flatten nested lists,
                                      'reshape' will reshape arrays to 1D.
                                      If None, predictions are used as-is.

    Returns:
    - None
    """

    # Read the Excel file without loading data into memory
    xls = pd.ExcelFile(input_file)

    # Map each sheet name to its corresponding predicted values
    datasets = {
        'fatigue data - train': train_preds,
        'fatigue data - test': test_preds,
        'fatigue data - validate': validate_preds
    }

    # Create an Excel writer object to save data
    with pd.ExcelWriter(output_file) as writer:
        for sheet_name, predictions in datasets.items():
            # Load data for the current sheet into a DataFrame
            df = pd.read_excel(xls, sheet_name)

            # Reshape predictions if needed
            if reshape_method == 'flatten':
                predictions = [item for sublist in predictions for item in sublist]
            elif reshape_method == 'reshape':
                predictions = np.reshape(predictions, -1)

            # Add the reshaped predictions to the DataFrame
            df['Predicted'] = predictions

            # Save the modified dataframe to a sheet in the new Excel file
            df.to_excel(writer, sheet_name=sheet_name, index=False)

def split_data_into_train_test_validate(input_file: str, output_file: str) -> str:
    """
    Splits data from an Excel file into train, test, and validate datasets.

    The procedure is as follows:
    1. Loads the dataset from the specified Excel file into a pandas DataFrame.
    2. Splits the DataFrame into a training set and a temporary set (combination of test and validate)
       with a ratio of 63% for training and 37% for the temporary set.
    3. Further splits the temporary set into test and validate sets. The split is done evenly,
       but given the proportions from the initial split, this results in approximately 10% of
       the original data for testing and 27% for validation.
    4. Saves the three datasets to separate sheets in a single Excel file specified by the output_file parameter.
       The sheets are named 'fatigue data - train', 'fatigue data - validate', and 'fatigue data - test'.
    5. Returns the path to the output Excel file where the datasets were saved.

    Args:
    - input_file (str): Path to the input Excel file.
    - output_file (str): Path where the split datasets will be saved.

    Returns:
    - str: Path where the split datasets were saved.
    """

    # Load data from Excel file
    df = pd.read_excel(input_file)

    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    train_data, validate_data = train_test_split(train_data, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    # Save each dataset to a separate sheet in the output Excel file
    with pd.ExcelWriter(output_file) as writer:
        train_data.to_excel(writer, sheet_name='fatigue data - train', index=False)
        validate_data.to_excel(writer, sheet_name='fatigue data - validate', index=False)
        test_data.to_excel(writer, sheet_name='fatigue data - test', index=False)

    return output_file