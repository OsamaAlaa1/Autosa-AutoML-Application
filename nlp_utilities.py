import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline



class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    A custom data preprocessor class.

    Parameters:
        text_columns (list): List of column names containing text data (default: None).
    """
    def __init__(self, text_columns=None):
        self.text_columns = text_columns

    def fit(self, X, y=None):
        # Auto-detect column types if not specified
        if self.text_columns is None:
            self.text_columns = self.detect_column_types(X)
        return self

    def transform(self, X):
        """
        Preprocesses the input data.

        Parameters:
            X (pandas.DataFrame): Input data.

        Returns:
            pandas.DataFrame: Preprocessed data.
        """
        # Preprocess text columns
        text_transformers = Pipeline([
            ('clean', TextCleaner()),
            ('tokenize', TextTokenizer()),
            ('vectorize', CountVectorizer())
        ])

        # Create a column transformer for different data transformations
        data_transformer = ColumnTransformer(
            transformers=[
                ('text', text_transformers, self.text_columns),
            ],
            remainder='passthrough'  # Include any columns not specified in the transformers
        )

        # Apply transformations
        X_transformed = data_transformer.fit_transform(X)

        return X_transformed

    def detect_column_types(self, X):
        """
        Automatically detect text columns in the dataset.

        Parameters:
            X (pandas.DataFrame): Input data.

        Returns:
            Tuple: (text_columns)
        """
        text_columns = X.select_dtypes(include='object').columns

        return text_columns

class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Custom transformer for text cleaning.
    Implement your text cleaning logic here.
    """
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Create a new DataFrame to store the transformed columns
        cleaned_X = X.copy()

        # Implement text cleaning logic here (e.g., remove special characters, lowercase)
        for col in cleaned_X.columns:
            cleaned_X[col] = cleaned_X[col].apply(self.clean_text)

        return cleaned_X

    def clean_text(self, text):
        """
        Clean and preprocess text.

        Parameters:
            text (str or list): Input text or list of texts.

        Returns:
            str or list: Cleaned text or list of cleaned texts.
        """
    
        tokens = word_tokenize(text)
        cleaned_tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in self.stop_words]
        return ' '.join(cleaned_tokens)


class TextTokenizer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for text tokenization.
    Implement your tokenization logic here.
    """
    def __init__(self, columns_to_tokenize=None):
        self.columns_to_tokenize = columns_to_tokenize

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Tokenize specified text columns
        if self.columns_to_tokenize:
            for col in self.columns_to_tokenize:
                X[col] = X[col].apply(self.tokenize_text)
        return X

    def tokenize_text(self, text):
        """
        Tokenize text.

        Parameters:
            text (str): Input text.

        Returns:
            list: List of tokens.
        """
        return text.split()


# Example usage:
if __name__ == "__main__":
    # Sample dataset with text, numeric, and categorical columns
    data = pd.DataFrame({
        'text_column': ["This is a sample text.", "Another example sentence.", "Text preprocessing is important!"]
    })

    # Define text columns
    text_columns = ['text_column']

    # Create an instance of the DataPreprocessor class
    preprocessor = DataPreprocessor(text_columns)

    # Fit and transform the data using the preprocessing pipeline
    preprocessed_data = preprocessor.fit_transform(data)

    # The preprocessed_data variable now contains the cleaned, tokenized, and vectorized data
    print(preprocessed_data)
