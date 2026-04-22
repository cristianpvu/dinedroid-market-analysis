"""Pagina 9 – Comparație clasificatori: DT, RF, GBM vs. Logistic Regression."""
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from utils.preprocessing import load_model_df

st.set_page_config(page_title="Clasificare Comparativă", page_icon="🏆", layout="wide")
st.title("9. Clasificare comparativă – modele multiple")

st.markdown(
    """
    **Definirea problemei.** Folosim aceeași variabilă țintă ca la pagina 7
    (lead premium = rating ≥ 4.0 AND votes ≥ median AND price range ≥ 2) și
    comparăm patru clasificatori:

    | Model | Idee principală |
    |---|---|
    | Logistic Regression | Limită de decizie liniară, probabilități calibrate |
    | Decision Tree | Întrebări DA/NU în cascadă, explicabil complet |
    | Random Forest | Sute de arbori independenți votează împreună (bagging) |
    | Gradient Boosting | Arbori construiți secvențial, fiecare corectează erorile anterioare |

    Evaluăm fiecare model cu **Accuracy**, **F1-Score (weighted)** și **ROC-AUC**.
    """
)

df_m = load_model_df()
features = ["cost_relativ", "city_size", "has_booking", "has_delivery"]
X = df_m[features].values
y = df_m["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

st.header("9.1 Antrenarea modelelor")

@st.cache_data(show_spinner=True)
def train_all(X_train_s, y_train, X_test_s, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42),
    }
    results = []
    trained = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1]
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred, average="weighted"),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        })
        trained[name] = (model, y_pred, y_proba)
    return pd.DataFrame(results), trained

results_df, trained_models = train_all(X_train_s, y_train, X_test_s, y_test)

st.header("9.2 Tabel comparativ")

display_df = results_df.copy()
display_df["Accuracy"] = display_df["Accuracy"].apply(lambda x: f"{x*100:.2f}%")
display_df["F1-Score"] = display_df["F1-Score"].apply(lambda x: f"{x:.4f}")
display_df["ROC-AUC"] = display_df["ROC-AUC"].apply(lambda x: f"{x:.4f}")
st.dataframe(display_df, use_container_width=True, hide_index=True)

fig = px.bar(
    results_df.melt(id_vars="Model", var_name="Metrică", value_name="Valoare"),
    x="Model",
    y="Valoare",
    color="Metrică",
    barmode="group",
    title="Comparație metrici — toate modelele",
    range_y=[0, 1],
)
st.plotly_chart(fig, use_container_width=True)

st.header("9.3 Curbe ROC")

auc_lookup = results_df.set_index("Model")["ROC-AUC"]
fig = px.line(title="Curbe ROC – toate modelele")
for name, (model, y_pred, y_proba) in trained_models.items():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc_lookup[name]:.3f})")
fig.add_shape(type="line", line=dict(dash="dash", color="gray"), x0=0, x1=1, y0=0, y1=1)
fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
st.plotly_chart(fig, use_container_width=True)

st.header("9.4 Matrici de confuzie")

cols = st.columns(2)
for i, (name, (model, y_pred, y_proba)) in enumerate(trained_models.items()):
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Real: Standard", "Real: Premium"],
        columns=["Pred: Standard", "Pred: Premium"],
    )
    fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues", title=name)
    cols[i % 2].plotly_chart(fig, use_container_width=True)

st.header("9.5 Raport de clasificare detaliat")

selected_model_name = st.selectbox("Alege modelul:", list(trained_models.keys()), index=2)
_, y_pred_sel, _ = trained_models[selected_model_name]
report = classification_report(
    y_test, y_pred_sel, target_names=["Standard", "Premium"], output_dict=True
)
st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

st.header("9.6 Importanța variabilelor")

fi_model_name = st.selectbox(
    "Alege modelul pentru feature importance:",
    ["Decision Tree", "Random Forest", "Gradient Boosting"],
    index=1,
)
fi_model, _, _ = trained_models[fi_model_name]
fi_df = pd.DataFrame(
    {"Feature": features, "Importanță": fi_model.feature_importances_}
).sort_values("Importanță", ascending=False)

fig = px.bar(
    fi_df,
    x="Importanță",
    y="Feature",
    orientation="h",
    title=f"Importanța variabilelor – {fi_model_name}",
    color="Importanță",
    color_continuous_scale="Blues",
)
st.plotly_chart(fig, use_container_width=True)

st.header("9.7 Interpretarea economică")

best_row = results_df.loc[results_df["ROC-AUC"].idxmax()]
st.markdown(
    f"""
    - **Cel mai bun model după ROC-AUC** este **{best_row['Model']}**
      (AUC = {best_row['ROC-AUC']:.3f}), ceea ce înseamnă că are cea mai bună capacitate
      de a separa lead-urile premium de cele standard.
    - **Decision Tree** este cel mai explicabil — fiecare decizie poate fi urmărită pas cu pas,
      util pentru a justifica prioritizarea unui lead în fața echipei de vânzări.
    - **Random Forest** și **Gradient Boosting** oferă precizie mai mare prin combinarea
      mai multor arbori, dar sunt mai greu de interpretat ("cutii negre" parțiale).
    - **Gradient Boosting** este mai sensibil la hyperparametri (`learning_rate`, `n_estimators`, `max_depth`)
      — necesită tuning mai atent, dar poate depăși Random Forest pe date cu pattern-uri subtile.
    - **Recomandare DineDroid:** folosiți **{best_row['Model']}** pentru scoring automat în batch,
      și **Decision Tree** pentru explicații la nivel de restaurant individual în interfața de vânzări.
    """
)
