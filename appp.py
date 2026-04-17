import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Iris Clustering", layout="centered")

st.title("🌸 Iris Dataset Clustering using K-Means")

# Load Iris dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)

st.subheader("Dataset Preview")
st.dataframe(data.head())

# Standardization
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Sidebar - cluster selection
st.sidebar.header("Model Settings")
k = st.sidebar.slider("Select Number of Clusters (k)", 2, 10, 3)

# Elbow Method
st.subheader("Elbow Method (WCSS)")
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

fig1, ax1 = plt.subplots()
ax1.plot(range(1, 11), wcss, marker='o')
ax1.set_xlabel("Number of Clusters")
ax1.set_ylabel("WCSS")
ax1.set_title("Elbow Method")
st.pyplot(fig1)

# K-Means Model
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
labels = kmeans.fit_predict(scaled_data)

# Silhouette Score
score = silhouette_score(scaled_data, labels)
st.success(f"Silhouette Score: {score:.3f}")

# Cluster Visualization
st.subheader("Cluster Visualization (Sepal Length vs Sepal Width)")

fig2, ax2 = plt.subplots()
scatter = ax2.scatter(
    data['sepal length (cm)'],
    data['sepal width (cm)'],
    c=labels,
    cmap='viridis'
)
ax2.set_xlabel("Sepal Length (cm)")
ax2.set_ylabel("Sepal Width (cm)")
ax2.set_title("Iris Clusters")
st.pyplot(fig2)

st.caption(
    "Note: Clustering is performed using all four features. "
    "The plot is a 2D projection for visualization only."
)