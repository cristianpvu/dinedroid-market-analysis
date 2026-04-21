"""Pagina 3 – Codificarea variabilelor categoriale și scalarea variabilelor numerice."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import (
    MinMaxScaler,
    MultiLabelBinarizer,
    OneHotEncoder,
    StandardScaler,
)

from utils.preprocessing import load_clean

st.set_page_config(page_title="Encoding & Scaling", page_icon="🔡", layout="wide")
st.title("3. Encoding & Scaling")

st.markdown(
    """
    **Definirea problemei.** Modelele din `scikit-learn` (clustering, regresie
    logistică) lucrează numai cu valori numerice și sunt sensibile la scara
    variabilelor. Avem nevoie să:

    1. **Codificăm** variabilele categoriale (`Rating text`, `Cuisines`, `City`)
       într-o reprezentare numerică.
    2. **Scalăm** variabilele numerice (`Average Cost for two`, `Votes`,
       `Aggregate rating`) astfel încât să aibă magnitudini comparabile.

    **Informații necesare.** Setul de date deja curățat (Pagina 2).
    """
)

df = load_clean()


st.header("3.1 One-Hot Encoding – `Rating text`")
st.markdown(
    """
    `Rating text` are 6 categorii distincte (`Excellent`, `Very Good`, `Good`,
    `Average`, `Poor`, `Not rated`). Cum nu există o ordine perfectă (în
    special pentru `Not rated`), aplicăm **One-Hot Encoding** – fiecare categorie
    devine o coloană binară 0/1.
    """
)

ohe = OneHotEncoder(sparse_output=False, dtype=int)
rating_ohe = ohe.fit_transform(df[["Rating text"]])
rating_ohe_df = pd.DataFrame(
    rating_ohe,
    columns=[c.replace("Rating text_", "rt_") for c in ohe.get_feature_names_out(["Rating text"])],
    index=df.index,
)
st.dataframe(
    pd.concat([df[["Restaurant Name", "Rating text"]], rating_ohe_df], axis=1).head(10),
    use_container_width=True,
    hide_index=True,
)

st.header("3.2 Multi-Label Binarizer – `Cuisines`")
st.markdown(
    """
    `Cuisines` este o listă (un restaurant poate fi `"Italian, Pizza, Cafe"`).
    `MultiLabelBinarizer` creează câte o coloană binară pentru fiecare tip de
    bucătărie. Pentru lizibilitate păstrăm doar **top 15 cele mai frecvente**.
    """
)

cuisine_lists = df["Cuisines"].fillna("Unknown").str.split(", ")
mlb = MultiLabelBinarizer()
cuisine_matrix = mlb.fit_transform(cuisine_lists)
cuisine_df = pd.DataFrame(cuisine_matrix, columns=mlb.classes_, index=df.index)

top_cuisines = cuisine_df.sum().sort_values(ascending=False).head(15)
fig = px.bar(
    top_cuisines.iloc[::-1],
    orientation="h",
    title="Top 15 tipuri de bucătărie după număr de restaurante",
    labels={"value": "Număr restaurante", "index": "Bucătărie"},
)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    pd.concat(
        [df[["Restaurant Name", "Cuisines"]], cuisine_df[top_cuisines.index]], axis=1
    ).head(10),
    use_container_width=True,
    hide_index=True,
)
st.caption(f"Total tipuri de bucătărie unice: **{len(mlb.classes_)}**")

st.header("3.3 Frequency Encoding – `City`")
st.markdown(
    """
    `City` are sute de valori distincte – One-Hot ar exploda dimensionalitatea.
    Aplicăm **frequency encoding**: fiecare oraș este înlocuit cu numărul de
    restaurante din oraș. Așa păstrăm o singură coloană dar transmitem informația
    de "mărime a pieței".
    """
)

city_freq = df["City"].value_counts()
df_freq = df.copy()
df_freq["City_freq"] = df_freq["City"].map(city_freq)
st.dataframe(
    df_freq[["Restaurant Name", "City", "City_freq"]].head(10),
    use_container_width=True,
    hide_index=True,
)

st.header("3.4 Scalarea variabilelor numerice")
st.markdown(
    """
    Comparăm două abordări pe `Average Cost for two`, `Votes`, `Aggregate rating`:

    - **StandardScaler**: $z = (x - \\mu) / \\sigma$ – centrează în 0 cu varianță 1.
      Recomandat pentru regresie logistică (presupune distribuții ~simetrice).
    - **MinMaxScaler**: $x' = (x - \\min)/(\\max - \\min)$ – aduce totul în [0, 1].
      Recomandat când vrei să păstrezi proporțiile relative.
    """
)

num_cols = ["Average Cost for two", "Votes", "Aggregate rating"]
std_scaled = StandardScaler().fit_transform(df[num_cols])
mm_scaled = MinMaxScaler().fit_transform(df[num_cols])

std_df = pd.DataFrame(std_scaled, columns=[f"{c}_std" for c in num_cols])
mm_df = pd.DataFrame(mm_scaled, columns=[f"{c}_mm" for c in num_cols])

stats = pd.DataFrame(
    {
        "Coloană": num_cols,
        "Mean original": [df[c].mean().round(2) for c in num_cols],
        "Std original": [df[c].std().round(2) for c in num_cols],
        "Mean (StandardScaler)": std_df.mean().round(3).tolist(),
        "Std (StandardScaler)": std_df.std().round(3).tolist(),
        "Min (MinMaxScaler)": mm_df.min().round(3).tolist(),
        "Max (MinMaxScaler)": mm_df.max().round(3).tolist(),
    }
)
st.dataframe(stats, use_container_width=True, hide_index=True)

c1, c2 = st.columns(2)
with c1:
    fig = px.histogram(
        std_df.melt(var_name="variable", value_name="value"),
        x="value",
        color="variable",
        nbins=60,
        barmode="overlay",
        title="Distribuții după StandardScaler",
    )
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.histogram(
        mm_df.melt(var_name="variable", value_name="value"),
        x="value",
        color="variable",
        nbins=60,
        barmode="overlay",
        title="Distribuții după MinMaxScaler",
    )
    st.plotly_chart(fig, use_container_width=True)

st.header("3.5 Interpretarea economică")
st.markdown(
    """
    - **One-Hot pe `Rating text`**: ne permite să modelăm impactul calității
      percepute fără a impune o ordine artificială (ex. "Not rated" nu e
      neapărat "mai prost" decât "Average").
    - **Multi-label pe `Cuisines`**: surprinde realitatea că un restaurant
      „Italian + Pizza + Cafe" e diferit de unul „Italian” simplu – relevant
      pentru DineDroid, fiindcă restaurantele cu meniu mixt sunt candidați
      excelenți pentru meniul digital (ușor de structurat în categorii).
    - **Frequency encoding pe `City`**: comprimăm sute de orașe într-o singură
      coloană interpretabilă ("cât de mare e piața locală").
    - **Scalarea**: necesară fiindcă `Votes` poate ajunge la mii, iar
      `Aggregate rating` între 0 și 5 – fără scalare, modelele de clustering ar
      fi dominate de `Votes` și ar ignora rating-ul.
    """
)
