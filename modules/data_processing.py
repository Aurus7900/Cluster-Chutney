#DATA_PREPROCESSING.PY

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def clean_data(df):
    """
    Cleans the user dataset:
    - Removes duplicates and fills missing values.
    - Converts Age and Income to numerical midpoints.
    - Splits 'Top Interests' if the column exists.
    """
    df_copy = df.copy()

    # Sanitize column names to remove leading/trailing whitespaces
    df_copy.columns = df_copy.columns.str.strip()

    # Drop duplicates and fill missing values
    df_copy = df_copy.drop_duplicates()
    df_copy = df_copy.fillna(method='ffill')

    # Convert Age ranges to midpoints
    if 'Age' in df_copy.columns:
        def age_to_midpoint(age):
            try:
                age = str(age).strip()
                if '+' in age:
                    return 70  # Assume age like "65+"
                elif '-' in age:
                    start, end = map(int, age.split('-'))
                    return (start + end) / 2
                return float(age) if age.isdigit() else None
            except Exception:
                return None
        df_copy['Age'] = df_copy['Age'].apply(age_to_midpoint)
    else:
        print("⚠️ Warning: 'Age' column not found.")

    # Convert Income Level ranges to estimated numeric midpoints
    if 'Income Level' in df_copy.columns:
        def income_to_midpoint(income):
            try:
                income = str(income).strip().replace('k', '')
                if '+' in income:
                    return 110000
                elif '-' in income:
                    start, end = map(int, income.split('-'))
                    return (start + end) * 500
                elif income.isdigit():
                    return int(income) * 1000
                return None
            except Exception:
                return None
        df_copy['Income Level'] = df_copy['Income Level'].apply(income_to_midpoint)
    else:
        print("⚠️ Warning: 'Income Level' column not found.")

    # Process Top Interests if column exists
    if 'Top Interests' in df_copy.columns:
        df_copy['Top Interests'] = df_copy['Top Interests'].apply(
            lambda x: [i.strip() for i in str(x).split(',')] if pd.notnull(x) else []
        )
    else:
        print("⚠️ Warning: 'Top Interests' column not found. Proceeding without it.")

    return df_copy


def encode_features(df):
    df = df.copy()

    le = LabelEncoder()
    if 'Gender' in df.columns:
        df['Gender'] = le.fit_transform(df['Gender'])

    categorical_cols = ['Location', 'Language', 'Device Usage', 'Education Level']
    existing_cols = [col for col in categorical_cols if col in df.columns]
    df = pd.get_dummies(df, columns=existing_cols)

    # One-hot encode Top Interests if it still exists
    if 'Top Interests' in df.columns:
        all_interests = set(interest for sublist in df['Top Interests'] for interest in sublist)
        for interest in all_interests:
            df[f'Interest_{interest}'] = df['Top Interests'].apply(lambda x: int(interest in x))
        df.drop(columns=['Top Interests'], inplace=True)

    # Scale numerical columns if they exist
    numeric_cols = [
        'Likes and Reactions', 'Followed Accounts',
        'Time Spent Online (hrs/weekday)', 'Time Spent Online (hrs/weekend)',
        'Click-Through Rates (CTR)', 'Conversion Rates', 'Ad Interaction Time (sec)'
    ]
    existing_numeric = [col for col in numeric_cols if col in df.columns]
    if existing_numeric:
        scaler = MinMaxScaler()
        df[existing_numeric] = scaler.fit_transform(df[existing_numeric])

    return df
