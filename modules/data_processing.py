# data_processing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

df = pd.read_csv(r"C:\Users\HP\Documents\Projects\user_segmentation\data\user_profiles_for_ads.csv")

def clean_data(df):
    df_copy = df.copy()
    df_copy = df_copy.drop_duplicates()
    df_copy = df_copy.fillna(method='ffill')

    def age_to_midpoint(age):
        if '+' in age:
            return 70
        else:
            start, end = map(int, age.split('-'))
            return (start + end) / 2

    df_copy['Age'] = df_copy['Age'].apply(age_to_midpoint)

    def income_to_midpoint(income):
        if '+' in income:
            return 110000
        else:
            start, end = income.replace('k', '').split('-')
            return (int(start) + int(end)) * 500

    df_copy['Income Level'] = df_copy['Income Level'].apply(income_to_midpoint)

    df_copy['Top Interests'] = df_copy['Top Interests'].apply(
        lambda x: [i.strip() for i in str(x).split(',')]
    )

    return df_copy   # <- must be indented inside the function

def encode_features(df):
    df = df.copy()

    # Label Encode Gender
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])

    # One-hot encode multi-category columns (if present)
    categorical_cols = ['Location', 'Language', 'Device Usage', 'Education Level']
    existing_cols = [col for col in categorical_cols if col in df.columns]
    df = pd.get_dummies(df, columns=existing_cols)

    # One-hot encode interests
    all_interests = set(interest for sublist in df['Top Interests'] for interest in sublist)
    for interest in all_interests:
        df[f'Interest_{interest}'] = df['Top Interests'].apply(lambda x: int(interest in x))
    df.drop(columns=['Top Interests'], inplace=True)

    # Normalize numeric columns
    numeric_cols = [
        'Likes and Reactions', 'Followed Accounts',
        'Time Spent Online (hrs/weekday)', 'Time Spent Online (hrs/weekend)',
        'Click-Through Rates (CTR)', 'Conversion Rates', 'Ad Interaction Time (sec)'
    ]
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df

df_cleaned = clean_data(df)
df_encoded = encode_features(df_cleaned)

df_encoded.to_csv(r"C:\Users\HP\Documents\Projects\user_segmentation\data\cleaned_users.csv", index=False)