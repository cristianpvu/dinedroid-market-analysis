"""Pagina 4 – Analiză exploratorie cu pandas: groupby, agregări, funcții de grup."""
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.preprocessing import load_clean

st.set_page_config(page_title="EDA & Grupări", page_icon="📊", layout="wide")
st.title("4. EDA & Grupări (pandas)")

st.markdown(
    """
    **Definirea problemei.** Pentru a decide pe ce piețe să se extindă DineDroid
    avem nevoie să înțelegem **profilul agregat** al pieței: în ce țări/orașe
    sunt cele mai multe restaurante, cât costă o masă, ce rating mediu au, ce
    procent oferă deja livrare online sau rezervare la masă.

    Folosim funcțiile de grup din `pandas` (`groupby`, `agg`, `transform`,
    `rank`, `apply`) pentru a răspunde la aceste întrebări.
    """
)

df = load_clean()

st.header("4.1 Profilul pieței per țară")
st.markdown(
    "Folosim `groupby('Country').agg({...})` cu **mai multe agregări simultane**."
)

per_country = (
    df.groupby("Country")
    .agg(
        Restaurante=("Restaurant ID", "count"),
        Cost_mediu=("Average Cost for two", "mean"),
        Cost_median=("Average Cost for two", "median"),
        Rating_mediu=("Aggregate rating", "mean"),
        Pct_table_booking=("Has Table booking", lambda s: (s == "Yes").mean() * 100),
        Pct_online_delivery=("Has Online delivery", lambda s: (s == "Yes").mean() * 100),
    )
    .round(2)
    .sort_values("Restaurante", ascending=False)
)
st.dataframe(per_country, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    fig = px.bar(
        per_country.reset_index().head(10),
        x="Country",
        y="Restaurante",
        title="Top 10 țări după număr de restaurante",
        color="Rating_mediu",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.scatter(
        per_country.reset_index(),
        x="Pct_online_delivery",
        y="Pct_table_booking",
        size="Restaurante",
        color="Rating_mediu",
        hover_name="Country",
        title="Maturitate digitală: livrare vs rezervări",
        labels={
            "Pct_online_delivery": "% restaurante cu livrare online",
            "Pct_table_booking": "% restaurante cu rezervare",
        },
    )
    st.plotly_chart(fig, use_container_width=True)

st.header("4.2 Top orașe (apply cu funcție custom)")

def city_summary(group: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "Restaurante": len(group),
            "Rating mediu": round(group["Aggregate rating"].mean(), 2),
            "Cost median": round(group["Average Cost for two"].median(), 2),
            "% premium (price 3-4)": round((group["Price range"] >= 3).mean() * 100, 1),
        }
    )

top_cities = (
    df.groupby(["Country", "City"])
    .apply(city_summary, include_groups=False)
    .reset_index()
    .sort_values("Restaurante", ascending=False)
    .head(20)
)
st.dataframe(top_cities, use_container_width=True, hide_index=True)

st.header("4.3 `transform` și `rank` – top performers per oraș")
st.markdown(
    """
    `transform` returnează rezultatul aliniat la rândurile originale – ideal
    pentru a calcula **devieri față de media orașului** sau **percentile**
    pe care apoi le adăugăm direct ca feature pe fiecare restaurant.
    """
)

df_ranked = df.copy()
df_ranked["Rating_vs_oras"] = (
    df_ranked["Aggregate rating"]
    - df_ranked.groupby("City")["Aggregate rating"].transform("mean")
).round(2)
df_ranked["Rang_oras"] = (
    df_ranked.groupby("City")["Aggregate rating"]
    .rank(method="dense", ascending=False)
    .astype(int)
)
df_ranked["Percentila_votes"] = (
    df_ranked.groupby("City")["Votes"].rank(pct=True).round(3)
)

df_ranked["Top_10pct_oras"] = df_ranked["Percentila_votes"] >= 0.9

st.dataframe(
    df_ranked[
        [
            "Restaurant Name",
            "City",
            "Aggregate rating",
            "Rating_vs_oras",
            "Rang_oras",
            "Percentila_votes",
            "Top_10pct_oras",
        ]
    ]
    .sort_values(["City", "Rang_oras"])
    .head(30),
    use_container_width=True,
    hide_index=True,
)

n_premium = int(df_ranked["Top_10pct_oras"].sum())
st.metric(
    "Restaurante 'top 10% în orașul lor' (lead-uri premium)",
    f"{n_premium:,}",
)

st.header("4.4 Tabel pivot: distribuția calitate × preț")
pivot = pd.crosstab(
    df["Price range"],
    df["Rating text"],
    normalize="index",
).round(3) * 100
st.dataframe(pivot, use_container_width=True)
st.caption("Valorile sunt în % – fiecare rând (price range) însumează 100.")

st.header("4.5 Interpretarea economică")
st.markdown(
    """
    - **Maturitatea digitală variază enorm între țări.** India are >25% livrare
      online dar <10% rezervare la masă, în timp ce UAE/Singapore au procente
      ridicate la ambele. Pentru DineDroid, acesta e un indicator direct: în
      piețele cu adopție digitală scăzută vindem **inovație** (modul digital),
      în piețele mature vindem **diferențiere** (NFC proof-of-presence).
    - **Top 10% restaurante per oraș** este o listă naturală de lead-uri
      premium – cele mai vizibile, cu cei mai mulți reviewers, deci cele mai
      tentate să adopte un sistem care le diferențiază.
    - **Pivotul preț × calitate** confirmă o regulă de business: restaurantele
      din price range 3-4 sunt disproporționat de "Excellent" / "Very Good" –
      exact segmentul pentru care proof-of-presence NFC are valoare maximă
      (programe de loialitate scumpe, mese pe rezervare).
    """
)
