"""Pagina 6 – Segmentarea restaurantelor cu KMeans."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from utils.preprocessing import load_clean

st.set_page_config(page_title="Clustering", page_icon="🧩", layout="wide")
st.title("6. Clustering – segmentarea restaurantelor")

st.markdown(
    """
    **Definirea problemei.** Vrem să identificăm **segmente naturale** de
    restaurante pentru a alinia propunerea de valoare DineDroid cu fiecare tip
    de client (fine dining, casual high-volume, delivery-first, underperformers).

    **Metodă.** `KMeans` din `scikit-learn`, aplicat pe variabile numerice
    standardizate. Alegem `k` optim pe baza:
    - **regulii cotului** (inerția vs k)
    - **scorului silhouette**
    """
)

df = load_clean()

st.header("6.1 Selecția și pregătirea variabilelor")

df_model = df.copy()
df_model["has_booking"] = (df_model["Has Table booking"] == "Yes").astype(int)
df_model["has_delivery"] = (df_model["Has Online delivery"] == "Yes").astype(int)

features = [
    "Average Cost for two",
    "Aggregate rating",
    "Votes",
    "Price range",
    "has_booking",
    "has_delivery",
]
st.write("Features folosite pentru clustering:", features)

df_model["cost_relativ_tara"] = df_model.groupby("Country")[
    "Average Cost for two"
].transform(lambda s: s / s.median())
features_final = [
    "cost_relativ_tara",
    "Aggregate rating",
    "Votes",
    "Price range",
    "has_booking",
    "has_delivery",
]

X = df_model[features_final].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.header("6.2 Alegerea numărului optim de clustere")

@st.cache_data(show_spinner=True)
def compute_kmeans_diagnostics(X_scaled: np.ndarray, k_range: tuple[int, int]) -> pd.DataFrame:
    rows = []
    rng = range(k_range[0], k_range[1] + 1)
    sample_idx = np.random.RandomState(42).choice(
        len(X_scaled), size=min(3000, len(X_scaled)), replace=False
    )
    for k in rng:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled[sample_idx], labels[sample_idx])
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)

diag = compute_kmeans_diagnostics(X_scaled, (2, 8))

c1, c2 = st.columns(2)
with c1:
    fig = px.line(diag, x="k", y="inertia", markers=True, title="Cot (inertia)")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.line(
        diag, x="k", y="silhouette", markers=True, title="Silhouette score"
    )
    st.plotly_chart(fig, use_container_width=True)

st.dataframe(diag.round(3), use_container_width=True, hide_index=True)

k = st.slider("Alege k:", min_value=2, max_value=8, value=4)

st.header(f"6.3 KMeans cu k={k}")

km = KMeans(n_clusters=k, n_init=10, random_state=42)
df_model["cluster"] = km.fit_predict(X_scaled)

profile = (
    df_model.groupby("cluster")
    .agg(
        Restaurante=("Restaurant ID", "count"),
        Cost_relativ=("cost_relativ_tara", "mean"),
        Rating=("Aggregate rating", "mean"),
        Votes=("Votes", "mean"),
        Price_range=("Price range", "mean"),
        Pct_booking=("has_booking", "mean"),
        Pct_delivery=("has_delivery", "mean"),
    )
    .round(2)
)
st.subheader("Profilul clusterelor")
st.dataframe(profile, use_container_width=True)

st.header("6.4 Vizualizare PCA 2D")
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
viz_df = pd.DataFrame(
    {
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "cluster": df_model["cluster"].astype(str),
        "Restaurant": df_model["Restaurant Name"],
        "City": df_model["City"],
    }
).sample(min(3000, len(df_model)), random_state=42)

fig = px.scatter(
    viz_df,
    x="PC1",
    y="PC2",
    color="cluster",
    hover_data=["Restaurant", "City"],
    title=f"Proiecție PCA – {k} clustere",
    opacity=0.6,
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"Variația explicată: PC1 = {pca.explained_variance_ratio_[0]*100:.1f}%, "
    f"PC2 = {pca.explained_variance_ratio_[1]*100:.1f}%"
)

st.header("6.5 Interpretarea economică a clusterelor")

st.markdown(
    """
    Citind tabelul de profil de mai sus, fiecare cluster capătă o **persona**
    de business pe care o putem lega de propunerea de valoare DineDroid:

    - **Fine dining premium** (cost relativ mare, rating mare, table booking mare) →
      vindem **NFC proof-of-presence** + **rezervări cu hartă 3D**.
    - **Casual high-volume** (cost mediu, votes mari, delivery mare) →
      vindem **meniu digital + comandă la masă** (reduce timp staff).
    - **Delivery-first** (cost mic, delivery=1, booking=0) →
      vindem **back-office + integrare cu agregatori**.
    - **Underperformers / nou intrate** (votes mici, rating lipsă) →
      pachet de **onboarding + program de loialitate** (NFC ca diferențiator).

    Concret: distribuția de mărime a clusterelor ne spune **câți potențiali
    clienți de fiecare tip** există în baza de date, iar profilul mediu ne
    spune **ce mesaj de vânzare** funcționează pentru fiecare segment.
    """
)

st.session_state["df_clustered"] = df_model
