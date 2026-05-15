"""Pagina 8 – Regresie liniară: predicția ratingului agregat."""
import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

st.header("8.7 Regresie multiplă cu `statsmodels` (OLS)")
st.markdown(
    """
    `scikit-learn` e bun pentru predicție, dar pentru **inferență statistică**
    (semnificația coeficienților, intervale de încredere, multicoliniaritate)
    folosim `statsmodels.api.OLS`. Spre deosebire de `LinearRegression`, `OLS`
    cere adăugarea explicită a coloanei de intercept prin `sm.add_constant`.
    """
)

X_train_sm = sm.add_constant(X_train_s)
ols_model = sm.OLS(y_train, X_train_sm).fit()

feature_names = ["const"] + [FEATURES[f] for f in selected]
sm_coefs = pd.DataFrame(
    {
        "Variabilă": feature_names,
        "Coef": ols_model.params,
        "Std err": ols_model.bse,
        "t": ols_model.tvalues,
        "p-value": ols_model.pvalues,
        "CI 2.5%": ols_model.conf_int()[:, 0],
        "CI 97.5%": ols_model.conf_int()[:, 1],
    }
).round(4)

c1, c2, c3, c4 = st.columns(4)
c1.metric("R²", f"{ols_model.rsquared:.3f}")
c2.metric("R² ajustat", f"{ols_model.rsquared_adj:.3f}")
c3.metric("F-statistic", f"{ols_model.fvalue:.2f}")
c4.metric("Prob (F-stat)", f"{ols_model.f_pvalue:.2e}")

st.subheader("Tabel coeficienți + semnificație statistică")
st.dataframe(sm_coefs, use_container_width=True, hide_index=True)

st.markdown(
    """
    **Cum se citește tabelul:**
    - **p-value < 0.05** → coeficientul este semnificativ statistic (variabila
      chiar contribuie la explicarea ratingului).
    - **CI (interval de încredere)** care **nu include 0** confirmă semnificația.
    - **R² ajustat** penalizează adăugarea de variabile inutile — îl preferăm
      lui R² brut când comparăm modele cu număr diferit de predictori.
    - **F-statistic** testează dacă modelul în ansamblu este mai bun decât
      simpla predicție a mediei. Prob(F) < 0.05 → modelul are valoare predictivă.
    """
)

with st.expander("Summary complet statsmodels (output text)"):
    st.text(str(ols_model.summary()))

st.subheader("Verificarea multicoliniarității (VIF)")
st.markdown(
    """
    **VIF (Variance Inflation Factor)** măsoară cât de mult este "umflată"
    varianta unui coeficient din cauza corelației cu celelalte variabile.
    Regulă uzuală: **VIF > 5** semnalează multicoliniaritate problematică,
    **VIF > 10** este gravă.
    """
)
vif_data = pd.DataFrame(
    {
        "Variabilă": [FEATURES[f] for f in selected],
        "VIF": [
            variance_inflation_factor(X_train_s, i) for i in range(X_train_s.shape[1])
        ],
    }
).round(3)
st.dataframe(vif_data, use_container_width=True, hide_index=True)

st.header("8.8 Interpretarea economică")
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
