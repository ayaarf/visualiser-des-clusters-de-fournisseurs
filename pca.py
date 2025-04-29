import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Set page configuration
st.set_page_config(page_title="Supplier Clustering Dashboard", layout="wide")

# Title and description
st.title("Supplier Clustering Dashboard")
st.markdown("""
This dashboard visualizes supplier clusters based on payment behavior and dispute metrics.
The model uses KMeans clustering with PCA for dimensionality reduction.
You can also evaluate a new supplier by entering their data below.
""")

# Define colors globally
colors = ['#4daf4a', '#e41a1c', '#377eb8', '#ff7f00', '#984ea3']

# Load models and data
@st.cache_resource
def load_models_and_data():
    try:
        pca = joblib.load("pca_model.pkl")
        kmeans = joblib.load("kmeans_model.pkl")
        df = pd.read_csv("export_clust.csv")
        return pca, kmeans, df
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Please ensure 'pca_model.pkl', 'kmeans_model.pkl', and 'export_clust.csv' are in the correct directory.")
        st.stop()

try:
    pca, kmeans, df_export = load_models_and_data()
except Exception as e:
    st.error(f"Error loading models or data: {e}")
    st.stop()

# Load clustering data
try:
    df_cluster = pd.read_csv("Clustering.csv")
except FileNotFoundError:
    st.error("Error: 'Clustering.csv' not found. Please ensure the file is in the correct directory.")
    st.stop()

# Function to preprocess data
def preprocess_data(df, scaler=None):
    selected_features = [
        'Mean_Payment_Delay',
        'Paid_Ratio',
        'Unpaid_Ratio',
        'Financial_Risk_Score',
        'Total_Disputes',
        'Dispute_Intensity'
    ]
    
    missing_features = [f for f in selected_features if f not in df.columns]
    if missing_features:
        st.error(f"Missing features in DataFrame: {missing_features}")
        st.stop()
    
    X = df[selected_features]
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, selected_features, scaler
    else:
        try:
            X_scaled = scaler.transform(X)
            return X_scaled, selected_features
        except Exception as e:
            st.error(f"Error scaling data: {e}")
            st.stop()

# Function to plot clusters
def plot_clusters(X_pca, clusters, centers_pca, cluster_names, n_clusters, highlight_supplier=None, highlight_data=None, df_cluster=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i in range(n_clusters):
        cluster_data = X_pca[clusters == i]
        ax.scatter(
            cluster_data[:, 0],
            cluster_data[:, 1],
            c=colors[i],
            label=f"{cluster_names.get(i, f'Cluster {i}')}",  # No cluster count
            s=80,
            alpha=0.7
        )
    
    ax.scatter(
        centers_pca[:, 0], centers_pca[:, 1],
        c='black', marker='*', s=500,
        label='Centroids'
    )
    
    # Highlight specific supplier
    if highlight_supplier and df_cluster is not None:
        try:
            supplier_idx = df_cluster[df_cluster["Fk_Supplier"] == highlight_supplier].index[0]
            ax.scatter(
                X_pca[supplier_idx, 0],
                X_pca[supplier_idx, 1],
                c='yellow',
                edgecolors='black',
                s=200,
                label=f"{highlight_supplier}",
                marker='o'
            )
        except IndexError:
            st.warning(f"Supplier {highlight_supplier} not found in clustering data.")
    
    # Highlight new supplier
    if highlight_data is not None:
        ax.scatter(
            highlight_data[0],
            highlight_data[1],
            c='cyan',
            edgecolors='black',
            s=200,
            label="New Supplier",
            marker='^'
        )
    
    ax.set_xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f'Supplier Clustering' if not highlight_supplier and highlight_data is None else f'Supplier Clustering with Highlight')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.3)
    return fig

# Sidebar for parameters
show_metrics = st.sidebar.checkbox("Show Clustering Metrics", value=True)
selected_supplier = st.sidebar.selectbox("Select Supplier", options=["All"] + list(df_export["Fk_Supplier"].unique()))

# Preprocess existing data
try:
    X_scaled, selected_features, scaler = preprocess_data(df_cluster)
except Exception as e:
    st.error(f"Error preprocessing data: {e}")
    st.stop()

# Apply KMeans clustering with fixed number of clusters
try:
    n_clusters = 2  # Fixed number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_cluster['Cluster'] = kmeans.fit_predict(X_scaled)
except Exception as e:
    st.error(f"Error applying KMeans clustering: {e}")
    st.stop()

# Apply PCA transformation
try:
    X_pca = pca.transform(X_scaled)
    centers_pca = pca.transform(kmeans.cluster_centers_)
except ValueError as e:
    st.error(f"PCA transformation error: {e}")
    st.stop()

# Cluster names
cluster_names = {
    0: "Reliable Suppliers",
    1: "High-Risk Suppliers",
    2: "Moderate-Risk Suppliers",
    3: "Low-Volume Suppliers",
    4: "Dispute-Prone Suppliers"
}

# New Supplier Evaluation Section
st.subheader("Evaluate New Supplier")
with st.form("new_supplier_form"):
    st.write("Enter the details for the new supplier:")
    supplier_id = st.text_input("Supplier ID", value="NEW_SUPPLIER")
    mean_payment_delay = st.number_input("Mean Payment Delay (days)", min_value=0.0, value=0.0, step=0.1)
    paid_ratio = st.number_input("Paid Ratio (0 to 1)", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
    unpaid_ratio = st.number_input("Unpaid Ratio (0 to 1)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    financial_risk_score = st.number_input("Financial Risk Score (0 to 1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    total_disputes = st.number_input("Total Disputes", min_value=0, value=0, step=1)
    dispute_intensity = st.number_input("Dispute Intensity", min_value=0.0, value=0.0, step=0.01)
    
    submitted = st.form_submit_button("Evaluate Supplier")
    
    if submitted:
        # Validate inputs
        if paid_ratio + unpaid_ratio > 1.0:
            st.error("Error: Paid Ratio + Unpaid Ratio cannot exceed 1.0")
        elif not supplier_id.strip():
            st.error("Error: Supplier ID cannot be empty")
        else:
            try:
                # Create DataFrame for new supplier
                new_supplier_data = pd.DataFrame({
                    'Fk_Supplier': [supplier_id],
                    'Mean_Payment_Delay': [mean_payment_delay],
                    'Paid_Ratio': [paid_ratio],
                    'Unpaid_Ratio': [unpaid_ratio],
                    'Financial_Risk_Score': [financial_risk_score],
                    'Total_Disputes': [total_disputes],
                    'Dispute_Intensity': [dispute_intensity]
                })
                
                # Preprocess new supplier data
                X_new_scaled, _ = preprocess_data(new_supplier_data, scaler=scaler)
                
                # Predict cluster
                cluster_pred = kmeans.predict(X_new_scaled)[0]
                cluster_name = cluster_names.get(cluster_pred, f"Cluster {cluster_pred}")
                
                # Apply PCA transformation
                X_new_pca = pca.transform(X_new_scaled)[0]
                
                st.write(f"**Evaluation Results for {supplier_id}**:")
                st.write(f"- Predicted Cluster: {cluster_name}")
                st.write(f"- PCA Coordinates: (X: {X_new_pca[0]:.2f}, Y: {X_new_pca[1]:.2f})")
                
                # Visualize new supplier
                st.subheader(f"New Supplier: {supplier_id}")
                fig = plot_clusters(X_pca, df_cluster['Cluster'], centers_pca, cluster_names, n_clusters, highlight_data=X_new_pca)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error evaluating new supplier: {e}")

# Plot clusters
st.subheader("Cluster Visualization")
fig = plot_clusters(X_pca, df_cluster['Cluster'], centers_pca, cluster_names, n_clusters)
st.pyplot(fig)

# Display clustering metrics
if show_metrics:
    st.subheader("Clustering Metrics")
    try:
        st.write(f"- **Silhouette Score**: {silhouette_score(X_scaled, df_cluster['Cluster']):.3f}")
        st.write(f"- **Calinski-Harabasz Index**: {calinski_harabasz_score(X_scaled, df_cluster['Cluster']):.1f}")
        st.write(f"- **Davies-Bouldin Score**: {davies_bouldin_score(X_scaled, df_cluster['Cluster']):.3f}")
    except Exception as e:
        st.error(f"Error calculating clustering metrics: {e}")

# Display supplier details
st.subheader("Supplier Details")
if selected_supplier != "All":
    supplier_data = df_export[df_export["Fk_Supplier"] == selected_supplier]
    if not supplier_data.empty:
        st.write("**Selected Supplier Info**:")
        st.dataframe(supplier_data[["Fk_Supplier", "Cluster_Name", "PCA_X", "PCA_Y"]])
        
        # Highlight supplier in plot
        st.subheader(f"Highlighted Supplier: {selected_supplier}")
        fig = plot_clusters(X_pca, df_cluster['Cluster'], centers_pca, cluster_names, n_clusters, highlight_supplier=selected_supplier, df_cluster=df_cluster)
        st.pyplot(fig)
    else:
        st.warning(f"No data found for supplier {selected_supplier}.")
else:
    st.write("**All Suppliers**:")
    st.dataframe(df_export[["Fk_Supplier", "Cluster_Name", "PCA_X", "PCA_Y"]].head(10))

# Display cluster statistics
st.subheader("Cluster Statistics")
try:
    cluster_stats = df_cluster.groupby('Cluster').agg({
        'Mean_Payment_Delay': 'mean',
        'Paid_Ratio': 'mean',
        'Unpaid_Ratio': 'mean',
        'Financial_Risk_Score': 'mean',
        'Total_Disputes': 'mean',
        'Dispute_Intensity': 'mean',
        'Avg_Dispute_Duration': 'mean'
    }).reset_index()
    cluster_stats["Cluster_Name"] = cluster_stats["Cluster"].map(cluster_names)
    st.dataframe(cluster_stats)
except Exception as e:
    st.error(f"Error calculating cluster statistics: {e}")

# Footer
st.markdown("---")
st.markdown("Developed with Streamlit | Data Source: Supplier Invoices and Disputes")