"""Pagina 2 – Tratarea valorilor lipsă și a valorilor extreme."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.loader import load_enriched

st.set_page_config(page_title="Curățare date", page_icon="🧹", layout="wide")
st.title("2. Curățarea datelor")

st.markdown(
    """
    **Definirea problemei.** Înainte de orice analiză sau model, datele trebuie
    aduse într-o formă consistentă. La datasetul Zomato avem două categorii de
    probleme: (1) **valori lipsă** (în special pe `Cuisines`) și (2) **valori
    extreme / aberante** pe variabile numerice precum `Average Cost for two`
    (cost mediu pentru două persoane) și `Votes` – acolo unde câteva restaurante
    "outlier" pot distorsiona statisticile descriptive și antrenarea modelelor.

    **Informații necesare.** Setul de date complet, plus cunoașterea unităților
    de măsură (costul e exprimat în moneda locală – deci outlier-ii se tratează
    *per țară*, nu global).
    """
)

df = load_enriched()
st.caption(f"Dimensiune set de date inițial: **{df.shape[0]:,} rânduri × {df.shape[1]} coloane**")

# ──────────────────────────────────────────────────────────────────────────────
# 1. Valori lipsă
# ──────────────────────────────────────────────────────────────────────────────
st.header("2.1 Valori lipsă")

missing = (
    df.isna()
    .sum()
    .to_frame("Valori lipsă")
    .assign(**{"Procent (%)": lambda x: (x["Valori lipsă"] / len(df) * 100).round(2)})
    .sort_values("Valori lipsă", ascending=False)
)
missing = missing[missing["Valori lipsă"] > 0]

col_a, col_b = st.columns([1, 2])
with col_a:
    if missing.empty:
        st.success("Nu există valori lipsă marcate ca NaN.")
    else:
        st.dataframe(missing, use_container_width=True)
with col_b:
    st.markdown(
        """
        **Observații:**
        - `Cuisines` are câteva valori lipsă reale → înlocuim cu `"Unknown"`,
          deoarece nu are sens să imputăm un tip de bucătărie inventat.
        - Atenție la **valori "marcate" ca prezente, dar fără sens economic**:
          - `Aggregate rating == 0.0` înseamnă, de fapt, "neevaluat" (nu un
            rating de zero) – îl tratăm ca lipsă.
          - `Average Cost for two == 0` este imposibil pentru un restaurant
            real – îl tratăm ca lipsă.
        """
    )

# Detectare "missing ascuns"
hidden_missing = pd.DataFrame(
    {
        "Coloană": ["Aggregate rating", "Average Cost for two"],
        "Regulă": ["valoare = 0.0", "valoare = 0"],
        "Număr cazuri": [
            int((df["Aggregate rating"] == 0).sum()),
            int((df["Average Cost for two"] == 0).sum()),
        ],
    }
)
st.markdown("**Valori lipsă ascunse:**")
st.dataframe(hidden_missing, use_container_width=True, hide_index=True)

st.markdown("**Strategia de tratare:**")
st.code(
    """
df_clean = df.copy()
df_clean["Cuisines"] = df_clean["Cuisines"].fillna("Unknown")
df_clean.loc[df_clean["Aggregate rating"] == 0, "Aggregate rating"] = np.nan
df_clean.loc[df_clean["Average Cost for two"] == 0, "Average Cost for two"] = np.nan

# Imputare: rating și cost mediu, *per țară* (mediana e robustă la outlieri)
for col in ["Aggregate rating", "Average Cost for two"]:
    df_clean[col] = df_clean.groupby("Country")[col].transform(
        lambda s: s.fillna(s.median())
    )
    """,
    language="python",
)

df_clean = df.copy()
df_clean["Cuisines"] = df_clean["Cuisines"].fillna("Unknown")
df_clean.loc[df_clean["Aggregate rating"] == 0, "Aggregate rating"] = np.nan
df_clean.loc[df_clean["Average Cost for two"] == 0, "Average Cost for two"] = np.nan
for col in ["Aggregate rating", "Average Cost for two"]:
    df_clean[col] = df_clean.groupby("Country")[col].transform(
        lambda s: s.fillna(s.median())
    )

st.success(
    f"După imputare: **{df_clean['Aggregate rating'].isna().sum()}** rating-uri "
    f"și **{df_clean['Average Cost for two'].isna().sum()}** costuri rămase NaN."
)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Valori extreme
# ──────────────────────────────────────────────────────────────────────────────
st.header("2.2 Valori extreme (outlieri)")

st.markdown(
    """
    **Metoda IQR (interquartile range).** O observație este considerată outlier
    dacă se află în afara intervalului:

    $$[\,Q_1 - 1.5 \\cdot IQR,\\; Q_3 + 1.5 \\cdot IQR\\,]$$

    unde $IQR = Q_3 - Q_1$. Aplicăm regula **per țară** pentru `Average Cost for two`
    (deoarece valutele diferă) și **global** pentru `Votes`.
    """
)

target_col = st.selectbox(
    "Variabila analizată:",
    ["Average Cost for two", "Votes", "Aggregate rating"],
    index=0,
)

def iqr_bounds(s: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

if target_col == "Average Cost for two":
    bounds = df_clean.groupby("Country")[target_col].apply(iqr_bounds)
    flags = []
    for country, (low, high) in bounds.items():
        mask = (
            (df_clean["Country"] == country)
            & ((df_clean[target_col] < low) | (df_clean[target_col] > high))
        )
        flags.append(mask)
    is_outlier = np.logical_or.reduce(flags)
else:
    low, high = iqr_bounds(df_clean[target_col])
    is_outlier = (df_clean[target_col] < low) | (df_clean[target_col] > high)

n_out = int(is_outlier.sum())
pct_out = n_out / len(df_clean) * 100

c1, c2, c3 = st.columns(3)
c1.metric("Outlieri detectați", f"{n_out:,}")
c2.metric("Procent din total", f"{pct_out:.2f}%")
c3.metric("Restaurante valide", f"{len(df_clean) - n_out:,}")

fig = px.box(
    df_clean,
    x="Country" if target_col == "Average Cost for two" else None,
    y=target_col,
    points="outliers",
    title=f"Distribuția pentru `{target_col}`",
)
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
    **Strategia de tratare a outlier-ilor.** În loc să eliminăm rândurile (am
    pierde restaurante de fine dining legitime), aplicăm **winsorizare** –
    plafonăm valorile extreme la limitele IQR. Astfel păstrăm dimensiunea
    setului, dar limităm influența disproporționată a câtorva observații.
    """
)

df_winsor = df_clean.copy()
def winsorize(s: pd.Series, k: float = 1.5) -> pd.Series:
    low, high = iqr_bounds(s, k)
    return s.clip(lower=low, upper=high)

df_winsor["Average Cost for two"] = df_winsor.groupby("Country")[
    "Average Cost for two"
].transform(winsorize)
df_winsor["Votes"] = winsorize(df_winsor["Votes"])

st.subheader("Comparație înainte / după winsorizare")
compare = pd.DataFrame(
    {
        "Metrică": ["Mean cost (USD-equivalent N/A)", "Std cost", "Max cost", "Mean votes", "Max votes"],
        "Înainte": [
            df_clean["Average Cost for two"].mean(),
            df_clean["Average Cost for two"].std(),
            df_clean["Average Cost for two"].max(),
            df_clean["Votes"].mean(),
            df_clean["Votes"].max(),
        ],
        "După": [
            df_winsor["Average Cost for two"].mean(),
            df_winsor["Average Cost for two"].std(),
            df_winsor["Average Cost for two"].max(),
            df_winsor["Votes"].mean(),
            df_winsor["Votes"].max(),
        ],
    }
)
compare["Înainte"] = compare["Înainte"].round(2)
compare["După"] = compare["După"].round(2)
st.dataframe(compare, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Interpretare economică
# ──────────────────────────────────────────────────────────────────────────────
st.header("2.3 Interpretarea economică a rezultatelor")
st.markdown(
    """
    Pentru DineDroid, calitatea curățării contează direct la pasul de **lead
    scoring**: dacă lăsăm rating-uri de `0.0` în model, restaurantele neevaluate
    apar artificial ca "slabe" și sunt eliminate din lista de prospecți – când
    de fapt, lipsa rating-ului poate însemna pur și simplu un restaurant nou,
    care e exact tipul de client interesat să adopte o platformă digitală.

    Tratarea outlier-ilor pe `Average Cost for two` *per țară* este esențială:
    un cost de 5.000 INR în India e premium dar plauzibil, în timp ce 5.000 USD
    în SUA ar fi clar o eroare de date. Winsorizarea pe grupuri de țări previne
    "dispariția" segmentului fine-dining, care pentru DineDroid e un target
    important (proof-of-presence prin NFC are valoare maximă acolo unde fiecare
    masă contează).
    """
)

# salvăm DataFrame-ul curățat în session state pentru paginile următoare
st.session_state["df_clean"] = df_winsor
st.caption("✅ Setul curățat este disponibil în `st.session_state['df_clean']` pentru paginile următoare.")
