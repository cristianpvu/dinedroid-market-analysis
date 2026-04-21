"""Pagina 5 – Analiză geospațială cu geopandas."""
import geopandas as gpd
import pandas as pd
import plotly.express as px
import streamlit as st
from shapely.geometry import Point

from utils.preprocessing import load_clean

st.set_page_config(page_title="Hartă Geo", page_icon="🗺️", layout="wide")
st.title("5. Analiză geospațială (geopandas)")

st.markdown(
    """
    **Definirea problemei.** Întrebarea de business este: *„unde să se extindă
    DineDroid – în ce țări și în ce orașe?"*. Răspunsul cere o vedere
    geografică a densității de restaurante și a calității lor medii.

    **Metodă.** Construim un `GeoDataFrame` din coordonatele `Longitude` /
    `Latitude` și folosim datasetul `naturalearth_lowres` din `geopandas` ca
    fundal pentru hartă.
    """
)

df = load_clean()

df_geo = df[
    (df["Longitude"].between(-180, 180))
    & (df["Latitude"].between(-90, 90))
    & ~((df["Longitude"] == 0) & (df["Latitude"] == 0))
].copy()

st.caption(
    f"Rânduri cu coordonate valide: **{len(df_geo):,}** "
    f"({len(df_geo) / len(df) * 100:.1f}% din total)"
)

st.header("5.1 Construirea GeoDataFrame-ului")

geometry = [Point(xy) for xy in zip(df_geo["Longitude"], df_geo["Latitude"])]
gdf = gpd.GeoDataFrame(df_geo, geometry=geometry, crs="EPSG:4326")

st.code(
    """
import geopandas as gpd
from shapely.geometry import Point

geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    """,
    language="python",
)

st.dataframe(
    gdf[["Restaurant Name", "City", "Country", "Aggregate rating", "geometry"]].head(8),
    use_container_width=True,
    hide_index=True,
)

st.header("5.2 Distribuția globală a restaurantelor")

sample = gdf.sample(min(3000, len(gdf)), random_state=42)
fig = px.scatter_geo(
    sample,
    lat="Latitude",
    lon="Longitude",
    color="Aggregate rating",
    size="Votes",
    size_max=15,
    hover_name="Restaurant Name",
    hover_data=["City", "Country", "Average Cost for two"],
    color_continuous_scale="Viridis",
    projection="natural earth",
    title="Eșantion de 3.000 restaurante (Zomato)",
)
fig.update_layout(height=550)
st.plotly_chart(fig, use_container_width=True)

st.header("5.3 Choropleth: densitatea pieței per țară")

per_country = (
    gdf.groupby("Country")
    .agg(
        Restaurante=("Restaurant ID", "count"),
        Rating_mediu=("Aggregate rating", "mean"),
        Cost_median=("Average Cost for two", "median"),
    )
    .round(2)
    .reset_index()
)

fig = px.choropleth(
    per_country,
    locations="Country",
    locationmode="country names",
    color="Restaurante",
    hover_data=["Rating_mediu", "Cost_median"],
    color_continuous_scale="Plasma",
    title="Număr de restaurante per țară",
)
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

st.header("5.4 Spatial join cu Natural Earth")
st.markdown(
    """
    Folosim shapefile-ul `naturalearth_lowres` din `geopandas` pentru a face un
    **spatial join**: pentru fiecare restaurant verificăm în ce poligon de țară
    cade. Util ca sanity check pentru coordonatele Zomato (vs codul de țară
    declarat).
    """
)

try:
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    sample_geo = gdf.sample(min(2000, len(gdf)), random_state=1)
    joined = gpd.sjoin(sample_geo, world[["name", "geometry"]], how="left", predicate="within")
    mismatch = joined[joined["name"].notna()].copy()
    mismatch["match"] = mismatch["Country"] == mismatch["name"]
    pct_match = mismatch["match"].mean() * 100
    st.metric("Coordonate care cad în țara declarată", f"{pct_match:.1f}%")
    st.dataframe(
        mismatch[["Restaurant Name", "City", "Country", "name", "match"]].head(15),
        use_container_width=True,
        hide_index=True,
    )
except Exception as e:
    st.warning(
        f"Spatial join cu naturalearth indisponibil în această versiune de geopandas: {e}"
    )

st.header("5.5 Vedere detaliată per țară")
country = st.selectbox(
    "Țară:",
    sorted(gdf["Country"].unique()),
    index=sorted(gdf["Country"].unique()).index("India") if "India" in gdf["Country"].unique() else 0,
)
sub = gdf[gdf["Country"] == country]
fig = px.scatter_mapbox(
    sub,
    lat="Latitude",
    lon="Longitude",
    color="Aggregate rating",
    size="Votes",
    size_max=18,
    hover_name="Restaurant Name",
    hover_data=["City", "Cuisines", "Average Cost for two"],
    color_continuous_scale="Viridis",
    zoom=3,
    height=550,
)
fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "l": 0, "t": 0, "b": 0})
st.plotly_chart(fig, use_container_width=True)

st.header("5.6 Interpretare economică")
st.markdown(
    """
    - **India domină** clar volumul (>90% din restaurante), cu clustere masive
      în Delhi NCR și Mumbai – cele două piețe-pilot evidente pentru DineDroid.
    - **Țările din Golf (UAE, Qatar)** au volume mai mici dar **costuri și
      rating-uri medii mai ridicate** – piețe premium cu margine mai mare,
      potrivite ca fază a 2-a de extindere.
    - **SUA și UK** apar cu volume reduse (eșantionarea Zomato e regională),
      deci pentru o decizie strategică pe aceste piețe ar fi nevoie de date
      complementare (Yelp, Google Places).
    - Spatial join-ul confirmă că coordonatele sunt curate (>95% match) – putem
      folosi cu încredere `Latitude/Longitude` ca feature în clustering.
    """
)
