from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class Mapper(BaseEstimator, TransformerMixin):
    """Categorical variable mapper."""

    def __init__(self, variables: str, mappings: dict):

        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variables] = X[self.variables].map(self.mappings).astype(int)

        return X
    
class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """
    def __init__(self):
        # No initialization is needed for this imputer
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # Nothing to fit, so we just return self
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Ensure that X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input is not a pandas DataFrame")

        # Check if 'dteday' and 'weekday' columns are in X
        if 'dteday' not in X.columns or 'weekday' not in X.columns:
            raise ValueError("DataFrame must contain 'dteday' and 'weekday' columns")

        # Copy the DataFrame to avoid changing the original data
        X_transformed = X.copy()

        # Impute missing values in 'weekday'
        X_transformed['weekday'] = X_transformed.apply(
            lambda row: row['weekday'] if pd.notnull(row['weekday']) else row['dteday'].day_name()[:3],
            axis=1
        )

        return X_transformed
    
class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self):
        # No initialization is needed for this imputer
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # Nothing to fit, so we just return self
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Ensure that X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input is not a pandas DataFrame")

        # Check if 'dteday' and 'weekday' columns are in X
        if 'weathersit' not in X.columns:
            raise ValueError("DataFrame must contain 'weathersit' column")

        # Copy the DataFrame to avoid changing the original data
        X_transformed = X.copy()

        most_common_value = X_transformed['weathersit'].mode()[0]
        # Impute missing values in 'weekday'
        X_transformed['weathersit'].fillna(most_common_value, inplace=True)

        return X_transformed
    
class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, numerical_columns: List[str]):
        self.numerical_columns = numerical_columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def handle_outliers(self, dataframe, colm):
        df = dataframe.copy()
        q1 = df.describe()[colm].loc['25%']
        q3 = df.describe()[colm].loc['75%']
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        for i in df.index:
            if df.loc[i,colm] > upper_bound:
                df.loc[i,colm]= upper_bound
            if df.loc[i,colm] < lower_bound:
                df.loc[i,colm]= lower_bound
        return df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Check if input is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input is not a pandas DataFrame")

        X_transformed = X.copy()
        for cols in self.numerical_columns:
            X_transformed = self.handle_outliers(X_transformed, cols)

        return X_transformed
    
class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self, weekday_one_hot):
        self.weekday_one_hot = weekday_one_hot
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
       # Check if input is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input is not a pandas DataFrame")

        # Check if 'weekday' column is in X
        if 'weekday' not in X.columns:
            raise ValueError("DataFrame must contain 'weekday' column")

        # Perform one-hot encoding
        X_transformed = X.copy()
        weekdays = pd.get_dummies(X_transformed['weekday'].astype(pd.CategoricalDtype(categories=self.weekday_one_hot)), prefix='weekday')
        X_transformed = pd.concat([X_transformed, weekdays], axis=1).drop('weekday', axis=1)
        return X_transformed

class ColumnDropper(BaseEstimator, TransformerMixin):
    """ Custom transformer for dropping specified columns """

    def __init__(self, cols_to_drop: List[str]):
        self.cols_to_drop = cols_to_drop

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # No fitting necessary, so just return self
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Drop specified columns
        return X.drop(columns=self.cols_to_drop, errors='ignore')
    
