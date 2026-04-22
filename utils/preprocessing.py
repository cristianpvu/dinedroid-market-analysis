"""Funcții de curățare reutilizabile între pagini."""
import numpy as np
import pandas as pd
import streamlit as st

from utils.loader import load_enriched


def iqr_bounds(s: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Returnează limitele IQR (low, high) pentru o serie."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def winsorize(s: pd.Series, k: float = 1.5) -> pd.Series:
    """Plafonează valorile la limitele IQR."""
    low, high = iqr_bounds(s, k)
    return s.clip(lower=low, upper=high)


@st.cache_data(show_spinner=False)
def load_clean() -> pd.DataFrame:
    """Dataset cu valori lipsă tratate și outlieri winsorizați."""
    df = load_enriched().copy()

    df["Cuisines"] = df["Cuisines"].fillna("Unknown")
    df.loc[df["Aggregate rating"] == 0, "Aggregate rating"] = np.nan
    df.loc[df["Average Cost for two"] == 0, "Average Cost for two"] = np.nan

    for col in ["Aggregate rating", "Average Cost for two"]:
        df[col] = df.groupby("Country")[col].transform(lambda s: s.fillna(s.median()))
    for col in ["Aggregate rating", "Average Cost for two"]:
        df[col] = df[col].fillna(df[col].median())

    df["Average Cost for two"] = df.groupby("Country")["Average Cost for two"].transform(
        winsorize
    )
    df["Votes"] = winsorize(df["Votes"])

    return df


@st.cache_data(show_spinner=False)
def load_featured() -> pd.DataFrame:
    """load_clean() plus derived ML features used across modelling pages."""
    df = load_clean().copy()
    df["has_booking"] = (df["Has Table booking"] == "Yes").astype(int)
    df["has_delivery"] = (df["Has Online delivery"] == "Yes").astype(int)
    df["city_size"] = df["City"].map(df["City"].value_counts())
    df["cost_relativ"] = df.groupby("Country")["Average Cost for two"].transform(
        lambda s: s / s.median()
    )
    return df


@st.cache_data(show_spinner=False)
def load_model_df() -> pd.DataFrame:
    """load_featured() plus the binary premium-lead target used in pages 7 & 9."""
    df = load_featured().copy()
    votes_median = df["Votes"].median()
    df["target"] = (
        (df["Aggregate rating"] >= 4.0)
        & (df["Votes"] >= votes_median)
        & (df["Price range"] >= 2)
    ).astype(int)
    return df
