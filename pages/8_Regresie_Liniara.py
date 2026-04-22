"""Pagina 8 – Regresie liniară: predicția ratingului agregat."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.preprocessing import load_featured

st.set_page_config(page_title="Regresie Liniară", page_icon="📈", layout="wide")
st.title("8. Regresie liniară – predicția ratingului")

st.markdown(
    """
    **Definirea problemei.** Vrem să prezicem **`Aggregate rating`-ul** unui restaurant
    pornind de la caracteristici obiective: cost, volum de voturi, interval de preț,
    prezență digitală (livrare/rezervare) și dimensiunea orașului.

    **Metodă.** `LinearRegression` din `scikit-learn`. Evaluăm cu **R²**, **RMSE** și **MAE**.
    Inspectăm graficul rezidualurilor pentru a verifica ipoteza de liniaritate.
    """
)

FEATURES = {
    "cost_relativ": "Cost relativ față de țară",
    "Votes": "Număr voturi",
    "Price range": "Price range (1–4)",
    "has_booking": "Are rezervare online (0/1)",
    "has_delivery": "Are livrare online (0/1)",
    "city_size": "Dimensiune oraș (nr. restaurante)",
}

st.header("8.1 Selecția variabilelor")
selected = st.multiselect(
    "Alege features pentru model:",
    options=list(FEATURES.keys()),
    default=list(FEATURES.keys()),
    format_func=lambda k: FEATURES[k],
)
if not selected:
    st.warning("Selectează cel puțin o variabilă.")
    st.stop()

df_m = load_featured().dropna(subset=selected + ["Aggregate rating"])
X = df_m[selected].values
y = df_m["Aggregate rating"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

st.header("8.2 Metrici de performanță")
c1, c2, c3, c4 = st.columns(4)
c1.metric("R²", f"{r2:.3f}", help="1.0 = perfect, 0 = la fel ca media")
c2.metric("RMSE", f"{rmse:.3f}", help="Eroarea medie în unități de rating")
c3.metric("MAE", f"{mae:.3f}", help="Eroare absolută medie")
c4.metric("Observații test", f"{len(y_test):,}")

st.header("8.3 Coeficienții modelului")
coef_df = pd.DataFrame(
    {"Feature": [FEATURES[f] for f in selected], "Coeficient (standardizat)": model.coef_}
).sort_values("Coeficient (standardizat)", key=abs, ascending=False)

c1, c2 = st.columns([2, 3])
with c1:
    st.dataframe(coef_df.round(4), use_container_width=True, hide_index=True)
with c2:
    fig = px.bar(
        coef_df,
        x="Coeficient (standardizat)",
        y="Feature",
        orientation="h",
        color="Coeficient (standardizat)",
        color_continuous_scale="RdBu",
        title="Importanța variabilelor (coeficienți standardizați)",
    )
    st.plotly_chart(fig, use_container_width=True)

st.header("8.4 Valori reale vs. prezise")
scatter_df = pd.DataFrame({"Real": y_test, "Prezis": y_pred})
fig = px.scatter(
    scatter_df,
    x="Real",
    y="Prezis",
    opacity=0.4,
    title="Rating real vs. rating prezis (set de test)",
    labels={"Real": "Rating real", "Prezis": "Rating prezis"},
)
fig.add_shape(
    type="line",
    x0=scatter_df["Real"].min(),
    x1=scatter_df["Real"].max(),
    y0=scatter_df["Real"].min(),
    y1=scatter_df["Real"].max(),
    line=dict(color="red", dash="dash"),
)
st.plotly_chart(fig, use_container_width=True)

st.header("8.5 Analiza rezidualurilor")
residuals = y_test - y_pred
res_df = pd.DataFrame({"Valori prezise": y_pred, "Rezidualuri": residuals})
fig = px.scatter(
    res_df,
    x="Valori prezise",
    y="Rezidualuri",
    opacity=0.4,
    title="Rezidualuri vs. valori prezise",
)
fig.add_hline(y=0, line_dash="dash", line_color="red")
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
    **Interpretare:**
    - Dacă punctele sunt distribuite aleator în jurul lui 0 → ipoteza de liniaritate este validă.
    - Un pattern în formă de pâlnie → heteroschedzaticitate (varianta erorilor nu e constantă).
    - O curbă sistematică → relație neliniară → ar trebui considerate transformări (log, sqrt).
    """
)

st.header("8.6 Distribuția rezidualurilor")
fig = px.histogram(
    res_df,
    x="Rezidualuri",
    nbins=50,
    title="Histograma rezidualurilor (ar trebui să fie aproape normală)",
)
st.plotly_chart(fig, use_container_width=True)

st.header("8.7 Interpretarea economică")
st.markdown(
    f"""
    - **R² = {r2:.3f}** — modelul explică **{r2*100:.1f}%** din variația ratingurilor.
      Valoarea relativ modestă confirmă că ratingul depinde mult de calitatea mâncării/serviciului,
      variabile pe care nu le avem în dataset.
    - **RMSE = {rmse:.3f}** puncte de rating — la o scară 0–5, aceasta este eroarea tipică de predicție.
    - **Cel mai important predictor** (după magnitudinea coeficientului) este
      **{coef_df.iloc[0]['Feature']}**, ceea ce sugerează că DineDroid ar trebui să prioritizeze
      restaurantele cu acest profil în strategia de onboarding.
    - **Limitare:** regresia liniară presupune relații liniare. Graficul rezidualurilor poate revela
      nevoia unui model mai complex (Random Forest, Gradient Boosting).
    """
)
