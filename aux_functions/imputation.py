import pandas as pd

def drop_rows_with_missing_values(input_file: str, output_file: str, save: bool = True) -> pd.DataFrame:
    """
    Remove rows with any missing values from a dataset in an Excel file.

    The process involves:
    1. Loading the dataset from the specified Excel file into a pandas DataFrame. This step includes reading the data, which could be from a specified sheet if the sheet name is given.
    2. Dropping all rows from the DataFrame that contain at least one missing value. This operation ensures that the dataset used in subsequent analyses or models is free of any gaps that could introduce bias or errors.
    3. If the save flag is set to True, the cleaned DataFrame, now devoid of any rows with missing values, is saved to a new Excel file specified by the output_file parameter. This step is optional and allows the user to retain a physical copy of the cleaned dataset for future use or inspection.
    4. The function then returns the cleaned DataFrame, providing immediate access to the processed data for in-memory analysis or further processing steps.

    This functionality is particularly useful in scenarios where the integrity of the dataset is paramount, and any missing data could compromise the quality of the analysis or the performance of data-driven models.

    Args:
    - input_file (str): The path to the Excel file containing the dataset.
    - output_file (str): The path where the cleaned dataset may be saved, if saving is enabled.
    - save (bool, optional): Flag indicating whether the cleaned dataset should be saved to a new Excel file. Defaults to True.

    Returns:
    - pd.DataFrame: The DataFrame with rows containing missing values removed.
    """

    # Load the data from the specified Excel sheet
    df = pd.read_excel(input_file)

    # Remove rows with any missing values
    df_cleaned = df.dropna()

    # Optionally save the cleaned dataframe to a new Excel file
    if save:
        df_cleaned.to_excel(output_file, index=False)

    return df_cleaned
