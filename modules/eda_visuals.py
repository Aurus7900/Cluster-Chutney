# eda_visuals.py

import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap
import math

# Custom color palette
colors = ['#1b112e', '#6c304c', '#bb3f48', '#e39372', '#3a1e6f']

def plot_age_distribution(df):
    plt.figure(figsize=(8, 5))
    plt.hist(df['Age'].dropna(), bins=15, color=colors[2], alpha=0.85)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("User Count")
    plt.tight_layout()
    plt.savefig("assets/visuals/age_distribution.png")
    plt.show()

def plot_gender_vs_interests(df, gender_col='Gender'):
    interest_cols = [col for col in df.columns if col.startswith('Interest_')]
    melted = df.melt(id_vars=[gender_col], value_vars=interest_cols,
                     var_name='Interest', value_name='HasInterest')
    melted = melted[melted['HasInterest'] == 1]
    melted['Interest'] = melted['Interest'].str.replace('Interest_', '', regex=False)

    top_interests = melted['Interest'].value_counts().nlargest(6).index
    melted_top = melted[melted['Interest'].isin(top_interests)]
    grouped = melted_top.groupby(['Interest', gender_col]).size().unstack(fill_value=0)

    interests = grouped.index.tolist()
    genders = grouped.columns.tolist()
    bar_width = 0.35
    x = range(len(interests))

    plt.figure(figsize=(10, 6))
    for i, gender in enumerate(genders):
        counts = grouped[gender].values
        plt.bar([pos + i * bar_width for pos in x], counts, width=bar_width,
                label=str(gender), color=colors[i % len(colors)])

    plt.title("Gender vs Top Interests")
    plt.xlabel("Top Interests")
    plt.ylabel("Count")
    plt.xticks([pos + bar_width / 2 for pos in x], interests, rotation=30)
    plt.legend(title='Gender')
    plt.tight_layout()
    plt.savefig("assets/visuals/gender_vs_interests.png")
    plt.show()

def plot_income_distribution(df):
    income_counts = df['Income Level'].value_counts()
    plt.figure(figsize=(8, 5))
    bars = plt.bar(income_counts.index.astype(str), income_counts.values,
                   color=colors[2], alpha=0.85)
    plt.title("Income Level Distribution")
    plt.xlabel("Income Level")
    plt.ylabel("User Count")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("assets/visuals/income_distribution.png")
    plt.show()

def plot_time_spent(df):
    plt.figure(figsize=(8, 5))
    df = df.copy()
    weekday = df['Time Spent Online (hrs/weekday)'].dropna()
    weekend = df['Time Spent Online (hrs/weekend)'].dropna()

    plt.hist(weekday, bins=30, density=True, alpha=0.5, label='Weekday', color=colors[0])
    plt.hist(weekend, bins=30, density=True, alpha=0.5, label='Weekend', color=colors[2])
    plt.title("Time Spent Online: Weekday vs Weekend")
    plt.xlabel("Hours")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("assets/visuals/time_spent_online.png")
    plt.show()

def show_heatmap(df, chunk_size=14, save_base_path="assets/visuals/heatmap"):
    numeric = df.select_dtypes(include=['float64', 'int64'])
    all_columns = numeric.columns
    n_features = len(all_columns)
    n_chunks = math.ceil(n_features / chunk_size)

    custom_cmap = LinearSegmentedColormap.from_list("custom_div", ["#521a34", "#ffc4e6"])

    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_features)
        chunk_cols = all_columns[start:end]
        chunk_df = numeric[chunk_cols]
        corr = chunk_df.corr()

        fig_width = max(10, 0.5 * corr.shape[1])
        fig_height = max(8, 0.5 * corr.shape[0])
        plt.figure(figsize=(fig_width, fig_height))

        plt.imshow(corr, cmap=custom_cmap, interpolation='nearest')
        plt.colorbar(shrink=0.75)
        plt.xticks(ticks=range(len(chunk_cols)), labels=chunk_cols, rotation=45, ha='right', fontsize=9)
        plt.yticks(ticks=range(len(chunk_cols)), labels=chunk_cols, fontsize=9)
        plt.title(f"Correlation Heatmap ({start + 1}–{end} of {n_features})", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{save_base_path}_{i+1}.png")
        plt.show()
