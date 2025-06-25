#APP.PY

import streamlit as st
import pandas as pd
import datetime
from PIL import Image
import os

from modules.eda_visuals import (
    plot_age_distribution,
    plot_gender_vs_interests,
    plot_income_distribution,
    plot_time_spent,
    show_heatmap,
)

from modules.after_visuals import (
    plot_pca_clusters,
    cluster_summary_table,
    plot_cluster_centroids_heatmap
)

from modules.clustering_module import (
    run_kmeans,
    elbow_method,
    find_optimal_k
)

from modules.data_processing import clean_data, encode_features

def generate_cluster_summary(df, cluster_id):
    # Step 1: Filter the cluster first (if not already filtered)
    cluster_df = df[df['Cluster'] == cluster_id].copy()

    # Step 2: Compute true stats from the raw data
    age = cluster_df["Age"].mean() if "Age" in cluster_df.columns else None
    income = cluster_df["Income Level"].mean() if "Income Level" in cluster_df.columns else None

    # Get most common gender
    if "Gender" in cluster_df.columns and not cluster_df["Gender"].isnull().all():
        most_common_gender = cluster_df["Gender"].mode()[0]
        gender_str = "Male" if most_common_gender == 1 else "Female"
    else:
        gender_str = "N/A"

    # Online time
    time_online = cluster_df.get("Weekend_Online_Hours", pd.Series([0])).mean()

    # Step 3: Cleaned data only for interests
    cleaned_df = clean_data(cluster_df.copy())

    # Extract top interest
    interest_columns = [col for col in cleaned_df.columns if col.startswith("Interest_")]
    top_interest = (
        max(interest_columns, key=lambda col: cleaned_df[col].mean(), default="N/A")
        .replace("Interest_", "")
        .replace("_", " ")
        if interest_columns else "N/A"
    )

    # Step 4: Assemble the summary string
    summary = f""" *Cluster {cluster_id} Summary*"""

    summary += f"\n- Average age: *{round(age, 1)}*" if age is not None else "\n- Average age: *N/A*"
    summary += f"\n- Avg income: *₹{income:,.2f}*" if income is not None else "\n- Avg income: *N/A*"
    summary += f"\n- Gender: *{gender_str}*"
    summary += f"\n- Most interested in *{top_interest}*"
    summary += f"\n- Great for *targeted campaigns* based on interest & income for *{top_interest}*!"

    return summary



# Streamlit page setup
st.set_page_config(layout="wide", page_title="User Profiling Dashboard")

# ---- Topbar Styling ----
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="stDecoration"] { display: none; }
        .topbar {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 9999;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #1b112e;
            padding: 1rem 2rem;
        }
        .topbar-logo {
            font-size: 24px;
            font-weight: bold;
            color: #ffffff;
        }
        .topbar-buttons {
            display: flex;
            gap: 1rem;
        }
        .topbar-buttons button {
            background-color: #6c304c;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            cursor: pointer;
        }
        .stApp {
            padding-top: 100px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="topbar">'
            '<div class="topbar-logo">Cluster Chutney</div>'
            '<div class="topbar-buttons">'
            '<form action="#hero"><button type="submit">Home</button></form>'
            '<form action="#about-us"><button type="submit">About Us</button></form>'
            '</div></div>', unsafe_allow_html=True)

# ---- Hero Section ----
st.markdown('<div id="hero"></div>', unsafe_allow_html=True)
st.markdown("## User Profiling & Segmentation Dashboard")
st.markdown("Explore, cluster, and understand user behavior to improve advertising and personalization.")

# ---- Upload Section ----
uploaded_file = st.file_uploader("Upload your cleaned user dataset (.csv)", type=["csv"])

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    st.success("✔ File uploaded successfully!")

    try:
        cleaned = clean_data(raw_df)
        df = encode_features(cleaned)
        st.success("✔ Data cleaned & encoded successfully!")
    except Exception as e:
        st.error(f"✖ Error during preprocessing: {e}")
        st.stop()

    if "original_df" not in st.session_state:
        st.session_state.original_df = df.copy()
    if "clustered_df" not in st.session_state:
        st.session_state.clustered_df = None
    if "kmeans_model" not in st.session_state:
        st.session_state.kmeans_model = None
    if "feature_names" not in st.session_state:
        st.session_state.feature_names = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    tab = st.radio("Navigate to:", ["EDA", "Clusters", "Profiles"])

    # -------------------- EDA TAB --------------------
    if tab == "EDA":
        st.header(" Exploratory Data Analysis")
        if st.checkbox("Show Raw Data"):
            st.write(df.head())

        chart = st.selectbox("Choose a chart to explore:", [
            "Age Distribution",
            "Gender vs Top Interests",
            "Income Level Distribution",
            "Time Spent Online (Weekday vs Weekend)",
            "Correlation Heatmap"
        ])

        if chart == "Age Distribution":
            plot_age_distribution(df)
            st.image("assets/visuals/age_distribution.png")

        elif chart == "Gender vs Top Interests":
            plot_gender_vs_interests(df)
            st.image("assets/visuals/gender_vs_interests.png")

        elif chart == "Income Level Distribution":
            plot_income_distribution(df)
            st.image("assets/visuals/income_distribution.png")

        elif chart == "Time Spent Online (Weekday vs Weekend)":
            plot_time_spent(df)
            st.image("assets/visuals/time_spent_online.png")

        elif chart == "Correlation Heatmap":
            show_heatmap(df)
            for file in sorted(os.listdir("assets/visuals")):
                if "heatmap" in file:
                    st.image(f"assets/visuals/{file}")

    # -------------------- CLUSTERS TAB --------------------
    elif tab == "Clusters":
        st.header(" User Segmentation with KMeans")

        method = st.radio("Clustering Method", ["Use Elbow Method (Auto)", "Choose K Manually"])

        if method == "Use Elbow Method (Auto)":
            st.info("Generating elbow plot & finding optimal K using silhouette score...")
            elbow_method(df.copy())
            st.image("assets/visuals/elbow.png", caption="Elbow Method")

            optimal_k = find_optimal_k(df.copy())
            st.success(f"Optimal K = {optimal_k}")
            scaled_data, kmeans_model, clustered_df = run_kmeans(df.copy(), optimal_k)

        else:
            k = st.slider("Select number of clusters (K)", min_value=2, max_value=10, value=4)
            if st.button("Run Clustering"):
                scaled_data, kmeans_model, clustered_df = run_kmeans(df.copy(), k)

        if 'clustered_df' in locals():
            st.session_state.clustered_df = clustered_df
            st.session_state.kmeans_model = kmeans_model

            st.subheader(" PCA Plot")
            plot_pca_clusters(clustered_df)
            st.image("assets/visuals/pca_clusters.png", caption="PCA Cluster Visualization")

            st.subheader(" Cluster Summary Table")
            summary = cluster_summary_table(clustered_df)
            st.dataframe(summary)

            st.subheader(" Cluster Centroid Heatmap")
            plot_cluster_centroids_heatmap(kmeans_model, st.session_state.feature_names)
            for img_file in sorted(os.listdir("assets/visuals")):
                if img_file.startswith("centroid_chunk"):
                    st.image(f"assets/visuals/{img_file}", caption=img_file)

            st.subheader(" Preview Clustered Data")
            st.dataframe(clustered_df.head())

            st.download_button("Download Clustered CSV", clustered_df.to_csv(index=False), file_name="clustered_users.csv")

    # -------------------- PROFILES TAB --------------------
    elif tab == "Profiles":
        st.header(" Cluster Profiles")

        if st.button("Check Clustering & Show Summary"):
            if st.session_state.clustered_df is None:
                st.info("Running clustering first...")
                try:
                    elbow_method(st.session_state.original_df.copy())
                    optimal_k = find_optimal_k(st.session_state.original_df.copy())
                    _, kmeans_model, clustered_df = run_kmeans(st.session_state.original_df.copy(), optimal_k)
                    st.session_state.clustered_df = clustered_df
                    st.session_state.kmeans_model = kmeans_model
                    st.success(f"✔ Clustering completed with K = {optimal_k}")
                except Exception as e:
                    st.error(f"✖ Clustering failed: {e}")
                    st.stop()

        df_clustered = st.session_state.clustered_df
        available_clusters = sorted(df_clustered['Cluster'].unique())
        selected_cluster = st.selectbox("Choose a Cluster", available_clusters)

        cluster_data = df_clustered[df_clustered["Cluster"] == selected_cluster]

        st.subheader(f" Profile Summary for Cluster {selected_cluster}")
        st.dataframe(cluster_data.describe())

        summary_text = generate_cluster_summary(cluster_data, selected_cluster)
        st.markdown(summary_text)

        st.download_button("Export Cluster Profile",
                        cluster_data.to_csv(index=False),
                        file_name=f"cluster_{selected_cluster}.csv")


# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown('<div id="about-us"></div>', unsafe_allow_html=True)
st.markdown("## About Us")
st.markdown("""
This project enables businesses to segment and understand user behavior using unsupervised learning.
We aim to support targeted ads, personalized recommendations, and behavior-based engagement strategies.
""")

st.markdown("""
*Team Members:*
- Sri Sai Lahari – [LahariRachapudi](https://github.com/LahariRachapudi)  
- Abhishek Karthik – [Abhishek18004](https://github.com/Abhishek18004)  
- Sai Sushanth Guddeti – [Aurus7900](https://github.com/Aurus7900)  
- Veenasree Krishna – [Veenbeans](https://github.com/Veenbeans)
""")