import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    A custom data preprocessor class.

    Parameters:
        numeric_columns (list): List of column names containing numeric data (default: None).
        categorical_columns (list): List of column names containing categorical data (default: None).
    """
    def __init__(self, numeric_columns=None, categorical_columns=None):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns

    def fit(self, X, y=None):
        # Auto-detect column types if not specified
        if self.numeric_columns is None or self.categorical_columns is None:
            self.numeric_columns, self.categorical_columns = self.detect_column_types(X)
        return self

    def transform(self, X):
        """
        Preprocesses the input data.

        Parameters:
            X (pandas.DataFrame): Input data.

        Returns:
            pandas.DataFrame: Preprocessed data.
        """

        # Preprocess numeric columns
        numeric_transformers = Pipeline([
            ('impute', SimpleImputer(strategy='mean')),
            ('scale', StandardScaler())
        ])

        # Preprocess categorical columns (one-hot encoding)
        categorical_transformers = Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder())
        ])

        # Create a column transformer for different data transformations
        data_transformer = ColumnTransformer(
            transformers=[

                ('numeric', numeric_transformers, self.numeric_columns),
                ('categorical', categorical_transformers, self.categorical_columns)
            ],
            remainder='passthrough'  # Include any columns not specified in the transformers
        )

        # Apply transformations
        X_transformed = data_transformer.fit_transform(X)

        return X_transformed

    def detect_column_types(self, X):
        """
        Automatically detect column types (numeric, categorical) in the dataset.

        Parameters:
            X (pandas.DataFrame): Input data.

        Returns:
            Tuple: (numeric_columns, categorical_columns)
        """
        numeric_columns = X.select_dtypes(include=np.number).columns
        categorical_columns = X.select_dtypes(include='object').columns

        return numeric_columns, categorical_columns



# Example usage:
if __name__ == "__main__":
    # Sample dataset with text, numeric, and categorical columns
    data = pd.DataFrame({
        'text_column': ["This is a sample text.", "Another example sentence.", "Text preprocessing is important!"],
        'numeric_column': [1.0, None, 3.0],
        'category_column': ['A', 'B', 'A']
    })

    # Define text, numeric, and categorical columns
    numeric_columns = ['numeric_column']
    categorical_columns = ['category_column']

    # Create an instance of the DataPreprocessor class
    preprocessor = DataPreprocessor(numeric_columns, categorical_columns)

    # Fit and transform the data using the preprocessing pipeline
    preprocessed_data = preprocessor.fit_transform(data)

    # The preprocessed_data variable now contains the cleaned, tokenized, imputed, scaled, and encoded data
    print(preprocessed_data)
