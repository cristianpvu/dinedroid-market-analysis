"""Încărcarea datasetului Zomato și mapări auxiliare."""
from pathlib import Path
import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "zomato.csv"

# Mapare cod țară -> nume țară (conform documentației Zomato Kaggle)
COUNTRY_CODES = {
    1: "India",
    14: "Australia",
    30: "Brazil",
    37: "Canada",
    94: "Indonesia",
    148: "New Zealand",
    162: "Philippines",
    166: "Qatar",
    184: "Singapore",
    189: "South Africa",
    191: "Sri Lanka",
    208: "Turkey",
    214: "UAE",
    215: "United Kingdom",
    216: "United States",
}


@st.cache_data(show_spinner=False)
def load_raw() -> pd.DataFrame:
    """Citește CSV-ul brut, fără modificări."""
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    return df


@st.cache_data(show_spinner=False)
def load_enriched() -> pd.DataFrame:
    """CSV brut + coloana Country derivată din Country Code."""
    df = load_raw().copy()
    df["Country"] = df["Country Code"].map(COUNTRY_CODES).fillna("Other")
    return df
