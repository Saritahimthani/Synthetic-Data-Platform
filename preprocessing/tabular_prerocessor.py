# preprocessing/tabular_preprocessor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class TabularPreprocessor:
    def __init__(self, categorical_columns=None, numerical_columns=None):
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.preprocessor = None
        self.column_names = None
        self.data_types = None
        
    def fit(self, data):
        # Auto-detect column types if not specified
        if self.categorical_columns is None or self.numerical_columns is None:
            self._detect_column_types(data)
            
        # Store original column names and data types for reconstruction
        self.column_names = data.columns.tolist()
        self.data_types = {col: data[col].dtype for col in data.columns}
        
        # Create preprocessing pipeline
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_columns),
                ('cat', categorical_transformer, self.categorical_columns)
            ])
        
        # Fit the preprocessor
        self.preprocessor.fit(data)
        return self
    
    def transform(self, data):
        # Transform the data
        transformed_data = self.preprocessor.transform(data)
        
        # Get feature names after transformation
        num_features = self.numerical_columns
        
        cat_features = []
        for i, col in enumerate(self.categorical_columns):
            encoder = self.preprocessor.transformers_[1][1].named_steps['onehot']
            cat_features.extend([f"{col}_{cat}" for cat in encoder.categories_[i]])
        
        # Return as DataFrame with proper column names
        return pd.DataFrame(
            transformed_data, 
            columns=num_features + cat_features
        )
    
    def inverse_transform(self, transformed_data):
        # This is a simplified version, full implementation would be more complex
        # to handle categorical variables properly
        return transformed_data
    
    def _detect_column_types(self, data):
        """Automatically detect categorical and numerical columns"""
        categorical_cols = []
        numerical_cols = []
        
        for col in data.columns:
            if data[col].dtype == 'object' or pd.api.types.is_categorical_dtype(data[col]):
                categorical_cols.append(col)
            elif pd.api.types.is_numeric_dtype(data[col]):
                numerical_cols.append(col)
        
        self.categorical_columns = categorical_cols
        self.numerical_columns = numerical_cols