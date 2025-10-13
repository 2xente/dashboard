import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder



def day_birth_transformation(df):
    if 'DAYS_BIRTH' in df.columns:
        df['AGE_YEARS'] = (-df['DAYS_BIRTH']) / 365
    return df



    
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

#def preprocess_data(df):
#    le = LabelEncoder()
#    for col in df.select_dtypes(include=['object']).columns:
#        le.fit(df[col].astype(str))
#        df[col] = le.transform(df[col].astype(str))
#    imputer = SimpleImputer(strategy='median')
#    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
#    scaler = MinMaxScaler()
#    df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df.columns)
#    return df_scaled
