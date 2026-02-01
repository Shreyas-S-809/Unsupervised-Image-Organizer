# Step E.1 --> Streamlit App Skeleton
import os 
import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.express as px 


st.set_page_config(layout = "wide")
st.title(" 🧠 Unsupervised Image Ograniser")
st.markdown(
    """
    **Pipeline:**
    Raw Images --> CNN Embeddings --> PCA --> t-SNE --> Clustering
    *(No labels were used at any stage)*
    """
)

# Step E. 2 --> Load the Articrafts 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "..", "viz_data.csv"))
    images = np.load(os.path.join(BASE_DIR, "..", "image_data.npy"))
    return df, images

df, images = load_data()

# Clean indexing defensively
df = df.reset_index(drop=True)



# Step E. 3 --> Layout (Graph + Inspector) 
col1, col2 = st.columns([3,1])

# Step E.4 --> 3D Interactive plot (Core Visual)

with col1:
    fig = px.scatter_3d(
        df,
        x = "x", 
        y = "y", 
        z = "z",
        color = df["kmeans_cluster"].astype(str),
        hover_data = ["image_id"],
        title = f"3D Image Clusters (K - Means on CNN Embeddings)"
    )
    

    fig.update_traces(marker = dict(size = 4))
    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white")
    )

    # --- Bordered container ---
    st.plotly_chart(fig, width="stretch")
    

# Step E. 5 --> Image Inspector 

with col2: 
    st.subheader(" 🔍 Image Inspector")

    img_id = st.number_input(
        "Enter the Image ID : Range (1 - 999)", 
        min_value=0, 
        max_value=len(images)-1,
        value=0, 
        step=1
    )

    img = images[img_id].astype("uint8")
    st.image(
        img, 
        caption=f"KMeans Cluster : {df.loc[img_id, 'kmeans_cluster']}",
        width=220
    )

    st.info(
        "Images within the same cluster share semantic features "
        "(shape, texture, object type) — without labels."
    )

    # 🔥 CLUSTER TOGGLE RIGHT UNDER THE INFO
    # DB SCAN Toggle Edge case 
    st.markdown("### 🧬 Clustering Method")
    cluster_type = st.radio(
        "", 
        ["KMeans", "DBSCAN"], 
        horizontal=True
    )
# Modifying the plot 

color_col = "kmeans_cluster" if cluster_type == "KMeans" else "dbscan_cluster" 