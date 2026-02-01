# Step E.1 --> Streamlit App Skeleton
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


st.set_page_config(layout="wide")
st.markdown("""
<style>
/* Wrap the ACTUAL Plotly canvas */
div[data-testid="stPlotlyChart"] > div > div {
    border: 3px solid #2a2f3a;      /* thicker */
    border-radius: 12px;
    padding: 8px;
    background-color: #0e1117;
    box-sizing: border-box;
}

/* Prevent canvas from overflowing and hiding bottom border */
div[data-testid="stPlotlyChart"] {
    overflow: visible !important;
}
</style>
""", unsafe_allow_html=True)


st.title(" 🧠 Unsupervised Image Ograniser")
st.markdown(
    """
    **Pipeline:**
    Raw Images --> CNN Embeddings --> PCA --> t-SNE --> Clustering
    *(No labels were used at any stage)*
    """
)

# Step E.2 --> Load the Artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    df = pd.read_csv("viz_data.csv")
    images = np.load("image_data.npy")
    return df, images

df, images = load_data()
df = df.reset_index(drop=True)

# Step E.3 --> Layout
col1, col2 = st.columns([3, 1])

# ---- Read clustering choice from session state (or default) ----
cluster_type = st.session_state.get("cluster_type", "KMeans")
color_col = "kmeans_cluster" if cluster_type == "KMeans" else "dbscan_cluster"

# Step E.4 --> 3D Plot
with col1:
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color=df[color_col].astype(str),
        hover_data=["image_id"],
        title=f"3D Image Clusters ({cluster_type})"
    )

    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white")
    )

    st.plotly_chart(fig, width="stretch")
    


# Step E.5 --> Image Inspector + Clustering Control
with col2:
    st.subheader(" 🔍 Image Inspector")

    img_id = st.number_input(
        "Enter the Image ID : Range (0 - {})".format(len(images) - 1),
        min_value=0,
        max_value=len(images) - 1,
        value=0,
        step=1
    )

    img = images[img_id].astype("uint8")
    st.image(
        img,
        caption=f"{cluster_type} Cluster : {df.loc[img_id, color_col]}",
        width=220
    )

    st.info(
        "Images within the same cluster share semantic features "
        "(shape, texture, object type) — without labels."
    )

    st.markdown("### 🧬 Clustering Method")

    # 🔥 This radio controls the plot via session_state
    st.radio(
        "",
        ["KMeans", "DBSCAN"],
        horizontal=True,
        key="cluster_type"
    )
