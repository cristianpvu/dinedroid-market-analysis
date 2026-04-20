"""Pagina 1 – Prezentarea contextului și a datasetului."""
import streamlit as st

from utils.loader import load_enriched

st.set_page_config(page_title="Prezentare", page_icon="📘", layout="wide")
st.title("1. Prezentare")

st.header("Contextul de business: DineDroid")
st.markdown(
    """
    **DineDroid** este o platformă integrată pentru restaurante care oferă:

    - **Proof of presence** prin tag-uri NFC criptate (NTAG 424 DNA) – pentru
      programe de loialitate și validarea prezenței efective la masă
    - **Meniu digital** și gestionarea comenzilor direct de la masă
    - **Rezervări** cu hartă 3D a sălii
    - **Back-office** pentru gestiunea restaurantului (stocuri, raportări, staff)

    Următorul pas natural este **extinderea pe piețe noi**. Întrebările-cheie:

    1. În ce **orașe / țări** există cea mai mare concentrație de restaurante
       compatibile cu profilul nostru?
    2. Ce **segment** de restaurante (fine dining, casual, delivery-first)
       reprezintă cel mai bun target inițial?
    3. Care sunt **caracteristicile** unui restaurant „ideal” pentru DineDroid?
    """
)

st.header("Datasetul: Zomato Restaurants")
df = load_enriched()

st.markdown(
    f"""
    Setul de date conține **{len(df):,} restaurante** din **{df['Country'].nunique()} țări**,
    cu informații despre locație, tip bucătărie, cost mediu, rating și features
    digitale (rezervare online, livrare etc.) – exact dimensiunile relevante
    pentru a evalua potrivirea cu DineDroid.
    """
)

st.subheader("Coloane disponibile")
schema = [
    ("Restaurant Name", "Numele restaurantului"),
    ("Country / City / Locality", "Locație geografică (folosit la geopandas)"),
    ("Longitude / Latitude", "Coordonate (hartă)"),
    ("Cuisines", "Listă de tipuri de bucătărie (multi-label)"),
    ("Average Cost for two", "Cost mediu pentru două persoane (în moneda locală)"),
    ("Currency", "Moneda locală – necesită normalizare"),
    ("Has Table booking", "Rezervare la masă (proxy pt. fine dining / DineDroid fit)"),
    ("Has Online delivery", "Livrare online (proxy pt. digital maturity)"),
    ("Price range", "1–4, range relativ de preț"),
    ("Aggregate rating / Votes", "Calitate percepută și volum recenzii"),
    ("Rating text", "Etichetă calitativă (Excellent, Very Good, ...)"),
]
st.table({"Coloană": [c for c, _ in schema], "Descriere": [d for _, d in schema]})

st.subheader("Eșantion de date")
st.dataframe(df.sample(20, random_state=42), use_container_width=True)
