"""
DineDroid – Market Expansion Analysis
Aplicație Streamlit pentru analiza pieței de restaurante (dataset Zomato)
ca suport pentru deciziile de extindere a platformei DineDroid.
"""
import streamlit as st

from utils.loader import load_enriched

st.set_page_config(
    page_title="DineDroid – Market Expansion",
    page_icon="🍽️",
    layout="wide",
)

st.title("DineDroid – Market Expansion Analysis")
st.caption(
    "Analiza pieței globale de restaurante (dataset Zomato) "
    "pentru identificarea oportunităților de extindere a platformei DineDroid."
)

df = load_enriched()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Restaurante", f"{len(df):,}")
c2.metric("Țări", df["Country"].nunique())
c3.metric("Orașe", df["City"].nunique())
c4.metric("Tipuri bucătărie", df["Cuisines"].dropna().str.split(", ").explode().nunique())

st.divider()

st.markdown(
    """
    ### Cum e structurată aplicația

    Folosește meniul din stânga pentru a naviga între pagini:

    1. **Prezentare** – contextul DineDroid și prezentarea datasetului
    2. **Curățare** – tratarea valorilor lipsă și a outlier-ilor
    3. **Encoding & Scaling** – pregătirea datelor pentru modele
    4. **EDA & Grupări** – analize agregate cu pandas
    5. **Hartă Geo** – analiză geospațială cu geopandas
    6. **Clustering** – segmentarea restaurantelor cu KMeans
    7. **Logistic** – scoring lead-uri DineDroid (regresie logistică)
    8. **Regresie Liniară** – predicția ratingului (R², RMSE, MAE, rezidualuri)
    9. **Clasificare Comparativă** – Decision Tree, Random Forest, Gradient Boosting vs. Logistic Regression

    Fiecare pagină corespunde uneia dintre cerințele obligatorii ale proiectului
    și răspunde la o întrebare concretă de business pentru DineDroid.
    """
)

with st.expander("Preview dataset brut"):
    st.dataframe(df.head(50), use_container_width=True)
