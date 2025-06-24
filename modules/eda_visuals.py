# eda_visuals.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap
import math

def plot_age_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Age'].dropna(), bins=15, kde=True, color='#457B9D')
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("User Count")
    plt.savefig("assets/visuals/age_distribution.png")
    plt.show()

def plot_gender_vs_interests(df, gender_col='Gender'):
    # List of interest columns
    interest_cols = [col for col in df.columns if col.startswith('Interest_')]

    # Melt the dataframe: wide -> long format
    melted = df.melt(id_vars=[gender_col], value_vars=interest_cols,
                     var_name='Interest', value_name='HasInterest')

    # Filter rows where interest is present
    melted = melted[melted['HasInterest'] == 1]

    # Simplify interest names (remove prefix)
    melted['Interest'] = melted['Interest'].str.replace('Interest_', '', regex=False)

    # Get top 6 most common interests overall
    top_interests = melted['Interest'].value_counts().nlargest(6).index

    # Filter to just those interests
    melted_top = melted[melted['Interest'].isin(top_interests)]

    # Plot
    plt.figure(figsize=(10, 6))
    sns.countplot(data=melted_top, x='Interest', hue=gender_col, palette='Set2')
    plt.title("Gender vs Top Interests")
    plt.xlabel("Top Interests")
    plt.ylabel("Count")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("assets/visuals/gender_vs_interests.png")
    plt.show()

def plot_income_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Income Level', order=df['Income Level'].value_counts().index)
    plt.title("Income Level Distribution")
    plt.xticks(rotation=30)
    plt.ylabel("User Count")
    plt.tight_layout()
    plt.savefig("assets/visuals/income_distribution.png")
    plt.show()

def plot_time_spent(df):
    plt.figure(figsize=(8, 5))
    sns.kdeplot(df['Time Spent Online (hrs/weekday)'], label='Weekday', fill=True, color='#1D3557')
    sns.kdeplot(df['Time Spent Online (hrs/weekend)'], label='Weekend', fill=True, color='#E23A46')
    plt.title("Time Spent Online: Weekday vs Weekend")
    plt.xlabel("Hours")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("assets/visuals/time_spent_online.png")
    plt.show()

def show_heatmap(df, chunk_size=13, save_base_path="/content/heatmap"):
    """
    Displays correlation heatmaps of numeric features in chunks for readability.

    Parameters:
    - df: pandas DataFrame
    - chunk_size: Number of features per heatmap
    - save_base_path: Base path for saving heatmap images
    """
    numeric = df.select_dtypes(include=['float64', 'int64'])

    all_columns = numeric.columns
    n_features = len(all_columns)
    n_chunks = math.ceil(n_features / chunk_size)

    custom_cmap = LinearSegmentedColormap.from_list("custom_div", ['#F1FAEE', '#457B9D'])

    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_features)
        chunk_cols = all_columns[start:end]
        chunk_df = numeric[chunk_cols]
        corr = chunk_df.corr()

        fig_width = max(10, 0.5 * corr.shape[1])
        fig_height = max(8, 0.5 * corr.shape[0])
        plt.figure(figsize=(fig_width, fig_height))

        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap=custom_cmap,
            linewidths=0.5,
            linecolor='gray',
            center=0,
            cbar_kws={"shrink": 0.75},
            annot_kws={"size": 8}
        )

        plt.title(f"Correlation Heatmap ({start + 1}–{end} of {n_features})", fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        plt.savefig("assets/visuals/heatmap.png")
        plt.show()