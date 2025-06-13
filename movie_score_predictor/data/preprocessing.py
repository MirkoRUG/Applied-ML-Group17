"""
Data preprocessing functions for movie dataset.
Handles date preprocessing, data type conversions, and feature engineering.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Tuple, List


def preprocess_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the 'released' column to extract date-related features.
    
    Args:
        df: DataFrame with 'released' column containing date strings
        
    Returns:
        DataFrame with additional date-related features
    """
    df = df.copy()
    
    # # date preprocessing disabled
    # if 'released' in df.columns:
    #     df['release_year'] = df['released'].apply(extract_year_from_release)
    #     df['release_month'] = df['released'].apply(extract_month_from_release)
    #     df['release_day'] = df['released'].apply(extract_day_from_release)
    #     
    #     df['is_summer_release'] = df['release_month'].apply(lambda x: 1 if x in [6, 7, 8] else 0)
    #     df['is_holiday_release'] = df['release_month'].apply(lambda x: 1 if x in [11, 12] else 0)
    #     df['is_spring_release'] = df['release_month'].apply(lambda x: 1 if x in [3, 4, 5] else 0)
    #     
    #     df['days_since_epoch'] = df['released'].apply(calculate_days_since_epoch)
    
    return df


def extract_year_from_release(release_str: str) -> int:
    """Extract year from release date string."""
    if pd.isna(release_str):
        return 2000  # Default year
    
    # Look for 4-digit year pattern
    year_match = re.search(r'\b(19|20)\d{2}\b', str(release_str))
    if year_match:
        return int(year_match.group())
    return 2000  # Default year


def extract_month_from_release(release_str: str) -> int:
    """Extract month from release date string."""
    if pd.isna(release_str):
        return 6
    
    month_mapping = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    release_lower = str(release_str).lower()
    for month_name, month_num in month_mapping.items():
        if month_name in release_lower:
            return month_num
    
    # Try to extract numeric month
    month_match = re.search(r'\b(\d{1,2})\b', str(release_str))
    if month_match:
        month = int(month_match.group())
        if 1 <= month <= 12:
            return month
    
    return 6  # Default month


def extract_day_from_release(release_str: str) -> int:
    """Extract day from release date string."""
    if pd.isna(release_str):
        return 15  # Default day
    
    # Look for day pattern (1-31)
    day_match = re.search(r'\b(\d{1,2})\b', str(release_str))
    if day_match:
        day = int(day_match.group())
        if 1 <= day <= 31:
            return day
    
    return 15  # Default day


def calculate_days_since_epoch(release_str: str) -> int:
    """Calculate days since epoch (1970-01-01) for temporal ordering."""
    if pd.isna(release_str):
        return 10957  # Default: 2000-01-01
    
    try:
        year = extract_year_from_release(release_str)
        month = extract_month_from_release(release_str)
        day = extract_day_from_release(release_str)
        
        date_obj = datetime(year, month, day)
        epoch = datetime(1970, 1, 1)
        return (date_obj - epoch).days
    except:
        return 10957  # Default: 2000-01-01


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert data types for optimal model performance.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with converted data types
    """
    df = df.copy()
    
    # Define categorical columns
    categorical_columns = [
        "name", "rating", "genre", "released", "director", "writer",
        "star", "country", "company"
    ]
    
    # Convert categorical columns to string type
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('Unknown')
    
    # Convert numeric columns
    numeric_columns = ["year", "score", "votes", "budget", "gross", "runtime"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def filter_frequent_categorical(df: pd.DataFrame, min_count: int = 2) -> pd.DataFrame:
    # only keep directors/stars that appear 2+ times
    for col in ['director', 'star']:
        if col in df.columns:
            value_counts = df[col].value_counts()
            frequent_values = value_counts[value_counts >= min_count].index
            df[col] = df[col].where(df[col].isin(frequent_values), 'Other')
    return df

def clean_and_prepare_data(df: pd.DataFrame, drop_columns: List[str] = None) -> pd.DataFrame:
    """
    Clean and prepare the dataset for model training.
    
    Args:
        df: Input DataFrame
        drop_columns: List of columns to drop (default: ['votes', 'gross'])
        
    Returns:
        Cleaned and prepared DataFrame
    """
    if drop_columns is None:
        drop_columns = ['votes', 'gross']
    
    df = df.copy()
    
    # Apply data type conversions
    df = convert_data_types(df)
    
    # df = preprocess_date_features(df)  # disabled
    
    df = filter_frequent_categorical(df)
    
    # Drop specified columns
    columns_to_drop = [col for col in drop_columns if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    
    # Handle missing values for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'score':  # Don't fill target variable
            df[col] = df[col].fillna(df[col].median())
    
    return df


def get_categorical_features() -> List[str]:
    """
    Get list of categorical feature names for CatBoost.

    Returns:
        List of categorical feature column names
    """
    return [
        "name", "rating", "genre", "released", "director", "writer",
        "star", "country", "company"
    ]


def prepare_features_and_target(df: pd.DataFrame, target_col: str = 'score') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for model training.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    df_clean = clean_and_prepare_data(df)
    
    # Remove rows where target is missing
    df_clean = df_clean.dropna(subset=[target_col])
    
    # Separate features and target
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    return X, y
