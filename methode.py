import pandas as pd


def load_data():
    """
    Load the dataset from a CSV file and return it as a pandas DataFrame.
    """
    try:
        df_test = pd.read_csv('application_test.csv')
        return df_test
    
def select_info(df, column_name):
    """
    Select a specific column from the DataFrame.
    
    :param df: The DataFrame to select from.
    :param column_name: The name of the column to select.
    :return: A pandas Series containing the selected column.
    """
    if column_name in df.columns:
        return df[column_name]
    else:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")