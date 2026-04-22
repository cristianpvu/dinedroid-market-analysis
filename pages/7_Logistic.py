"""Pagina 7 – Regresie logistică pentru lead scoring DineDroid."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.preprocessing import load_clean

st.set_page_config(page_title="Logistic – Lead Scoring", page_icon="🎯", layout="wide")
st.title("7. Regresie logistică – Lead scoring DineDroid")

st.markdown(
    """
    **Definirea problemei.** Vrem un model care, dat fiind profilul unui
    restaurant, să prezică **probabilitatea de a fi un lead bun pentru DineDroid**.
    Construim variabila țintă pe baza unor reguli de business:

    > Un restaurant este **lead premium (1)** dacă îndeplinește **toate**:
    > - `Aggregate rating >= 4.0`
    > - `Votes >= median(Votes)`
    > - `Price range >= 2`

    Restul sunt **lead-uri standard (0)**. Aplicăm `LogisticRegression` din
    `scikit-learn` și evaluăm pe accuracy, precision, recall, ROC-AUC.
    """
)

df = load_clean()

st.header("7.1 Variabila țintă")

votes_median = df["Votes"].median()
df_m = df.copy()
df_m["has_booking"] = (df_m["Has Table booking"] == "Yes").astype(int)
df_m["has_delivery"] = (df_m["Has Online delivery"] == "Yes").astype(int)
df_m["target"] = (
    (df_m["Aggregate rating"] >= 4.0)
    & (df_m["Votes"] >= votes_median)
    & (df_m["Price range"] >= 2)
).astype(int)

c1, c2 = st.columns(2)
with c1:
    counts = df_m["target"].value_counts().rename({0: "Standard", 1: "Premium"})
    fig = px.pie(values=counts.values, names=counts.index, title="Distribuția target")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.metric("Lead-uri premium", f"{int(counts.get('Premium', 0)):,}")
    st.metric("Lead-uri standard", f"{int(counts.get('Standard', 0)):,}")
    st.metric("Mediana Votes (prag)", f"{votes_median:.0f}")

st.header("7.2 Features")

city_freq = df_m["City"].value_counts()
df_m["city_size"] = df_m["City"].map(city_freq)

df_m["cost_relativ"] = df_m.groupby("Country")["Average Cost for two"].transform(
    lambda s: s / s.median()
)

features = [
    "cost_relativ",
    "city_size",
    "has_booking",
    "has_delivery",
]

st.write("Features (exclus rating/votes/price ca să evităm data leakage):", features)

X = df_m[features].values
y = df_m["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

st.header("7.3 Antrenarea modelului")

model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)[:, 1]

acc = (y_pred == y_test).mean()
auc = roc_auc_score(y_test, y_proba)

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{acc*100:.1f}%")
c2.metric("ROC-AUC", f"{auc:.3f}")
c3.metric("Lead-uri test", f"{int(y_test.sum()):,} / {len(y_test):,}")

st.header("7.4 Importanța features (coeficienți)")

coefs = pd.DataFrame(
    {
        "feature": features,
        "coef": model.coef_[0],
        "odds_ratio": np.exp(model.coef_[0]),
    }
).sort_values("coef", ascending=False)

c1, c2 = st.columns([2, 3])
with c1:
    st.dataframe(coefs.round(3), use_container_width=True, hide_index=True)
with c2:
    fig = px.bar(
        coefs,
        x="coef",
        y="feature",
        orientation="h",
        title="Coeficienții regresiei logistice (după standardizare)",
        color="coef",
        color_continuous_scale="RdBu",
    )
    st.plotly_chart(fig, use_container_width=True)

st.header("7.5 Evaluare detaliată")

c1, c2 = st.columns(2)
with c1:
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Real: Standard", "Real: Premium"],
        columns=["Pred: Standard", "Pred: Premium"],
    )
    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
    )
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig = px.area(
        x=fpr,
        y=tpr,
        title=f"ROC curve (AUC = {auc:.3f})",
        labels={"x": "False Positive Rate", "y": "True Positive Rate"},
    )
    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Classification report")
report = classification_report(
    y_test, y_pred, target_names=["Standard", "Premium"], output_dict=True
)
st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

st.header("7.6 Top 20 lead-uri prezise (din întreg setul)")

X_all_s = scaler.transform(df_m[features].values)
df_m["lead_score"] = model.predict_proba(X_all_s)[:, 1]

top_leads = df_m.nlargest(20, "lead_score")[
    [
        "Restaurant Name",
        "City",
        "Country",
        "Cuisines",
        "Average Cost for two",
        "Aggregate rating",
        "Votes",
        "lead_score",
    ]
].round(3)
st.dataframe(top_leads, use_container_width=True, hide_index=True)

st.header("7.7 Interpretarea economică")
st.markdown(
    f"""
    - **AUC = {auc:.3f}** arată că, doar pe baza features-lor "necontaminate"
      (mărime oraș, cost relativ, prezență digitală), modelul reușește deja
      să separe lead-urile premium de cele standard. Asta confirmă că DineDroid
      poate face **prospectare direcționată** chiar și fără să cunoască rating-ul
      sau volumul de recenzii al unui restaurant.
    - **Coeficienții** arată ce contează pentru a fi un lead premium:
      restaurantele din **orașe mari** și cu **adopție digitală deja existentă**
      (livrare/rezervare online) au probabilitate semnificativ mai mare de
      a fi clienți de calitate. Asta sugerează că DineDroid ar trebui să
      prioritizeze **vânzarea în piețele unde concurența digitală a făcut deja
      educația pieței**.
    - **Top 20 lead-uri prezise** sunt o listă concretă pe care echipa de
      vânzări o poate prelua imediat. În producție, modelul ar fi rerulat
      săptămânal cu date noi din scraping/API.
    - **Limitări**: target-ul e definit prin reguli fixe – într-o fază
      ulterioară l-am înlocui cu **răspunsul real** al lead-urilor (au răspuns
      pozitiv la outreach? Da/Nu), antrenând modelul pe istoricul CRM.
    """
)
