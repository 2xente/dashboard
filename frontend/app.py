import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import math
from datetime import datetime


st.set_page_config(page_title="CréditScope — prédiction de crédit",
    page_icon="💳",
    layout="wide")

st.title("💳 CréditScope")
st.caption("Prédiction d’éligibilité & explications locales")

@st.cache_data(show_spinner=False, ttl=60)
def fetch_global_importance(method: str = "model", k: int = 20, grouped: bool = True, shap_sample: int = 1000):
    r = requests.get(
        "http://localhost:8000/global_importance",
        params={"method": method, "k": k, "grouped": grouped, "shap_sample": shap_sample},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()


# --- layout à 2 colonnes ---
left, right = st.columns([3.5, 6.5])

# Titre de l'application
#st.title("Dashboard explication risque crédit")

# Bouton test chargement des données
#if st.button("Charger les données"):
#    try:
#        response = requests.get("http://localhost:8000/load_data")
#        df = response.json()["data"]
#        df = pd.DataFrame(df)
#        data_test = df.drop("SK_ID_CURR", axis=1)
#        st.session_state.data_test = data_test  
#        st.session_state.columns = data_test.columns.tolist()  
#        st.write(df.head())  
#    except Exception as e:
#        st.error(f"Erreur lors du chargement des données : {e}") 

# Bouton test chargement du model
#if st.button("Charger le model"):
#    try:
#        response = requests.get("http://localhost:8000/load_model")
#        print(response)
#        modele = response.json()["model"]
#        selector = modele["selector"]
#        feature_order = modele.get("feature_order", None)
#        model = modele["model"]
#        print(len(feature_order))
#      
#    except Exception as e:
#        st.error(f"Erreur lors du chargement des données : {e}") 

def risk_band(proba, threshold):
    if proba is None: return ("Indéterminé", "⚪️")
    if proba < 0.33: return ("Faible", "🟢")
    if proba < 0.66: return ("Modéré", "🟠")
    return ("Élevé", "🔴")

with left:

    st.subheader("Prédiction d’éligibilité")

    SK_ID_CURR = st.text_input("Entrez l'identifiant client", placeholder="ex: 100005")
    THRESHOLD = st.slider("Seuil de refus (%)", 0, 100, 50) / 100  # 0.50 par défaut

    def normalize_proba(p):
        try:
            p = float(p)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(p):
            return None
        # si jamais l’API renvoie 0–100, re-normalise
        if p > 1 and p <= 100:
            p /= 100
        return max(0.0, min(1.0, p))

    if st.button("Faire une prédiction", use_container_width=True):
        if not SK_ID_CURR.isdigit():
            st.warning("Veuillez entrer un identifiant **numérique**.")
        else:
            with st.spinner("Calcul en cours…"):
                try:
                    r = requests.get(f"http://localhost:8000/prediction", params={"sk_id_curr": int(SK_ID_CURR)}, timeout=10)
                    if r.status_code == 404:
                        st.error("Client introuvable.")
                    else:
                        r.raise_for_status()
                        data = r.json()
                        pred = data.get("prediction")
                        proba = normalize_proba(data.get("probability"))

                        # Décision (prend le label si présent, sinon seuil)
                        if isinstance(pred, (int, float)) and int(pred) in (0, 1):
                            decision = "ACCEPTÉ" if int(pred) == 0 else "REFUSÉ"
                        elif proba is not None:
                            decision = "ACCEPTÉ" if proba < THRESHOLD else "REFUSÉ"
                        else:
                            decision = "—"

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Décision", decision, help=f"Seuil: {THRESHOLD:.2f}")
                        c2.metric("Risque", f"{proba*100:.1f} %" if proba is not None else "—")
                        c3.metric("Client", SK_ID_CURR)

                        band, emoji = risk_band(proba, THRESHOLD)
                        st.markdown(f"**Niveau de risque : {emoji} {band}**")

                        if proba is not None and abs(proba - THRESHOLD) < 0.02:
                            st.info("Cas **borderline** : proche du seuil — un document complémentaire peut faire la différence.")


                        #if proba is not None:
                        #    st.progress(int(proba * 100))
                        #with st.expander("Détails (réponse API)"):
                        #    st.json(data)

                except requests.RequestException as e:
                    st.error(f"Erreur API : {e}")




# Zone de texte
#SK_ID_CURR = st.text_input("Entrez l'identifiant client")
#if st.button("Faire une prédiction"): 
#    try:
#        #loan prediciton
#        response = requests.get("http://localhost:8000/prediction", params={"sk_id_curr": SK_ID_CURR})
#        predictions = response.json()["prediction"]
#        proba = response.json()["probability"]
#        st.write(predictions)
#        st.write(proba)

        #client info
#customer_info = requests.get("http://localhost:8000/customer_info", params={"sk_id_curr": SK_ID_CURR})
#customer_info = customer_info.json()["customer_info"]
#st.write(customer_info)



#    except Exception as e:
#        st.error(f"Erreur lors du chargement des données : {e}") 

## Columns information
    columns_names_response = requests.get("http://localhost:8000/columns_names")
    columns_names = columns_names_response.json()["columns_names"]

    #col1, col2 = st.columns([3, 1])
    choix = st.selectbox("Choisissez une colonne", columns_names, index=0)
    lancer = st.button("Générer", use_container_width=True)

    st.session_state.setdefault("history", [])  # liste d'items: {when, column, sk_id, fig_dict, percentile, mean}

    def _now(): 
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _push_history(column, sk_id, fig, percentile=None, mean_value=None):
        st.session_state["history"].append({
            "when": _now(),
            "column": column,
            "sk_id": sk_id,
            "percentile": percentile,
            "mean": mean_value,
            "fig_dict": fig.to_dict(),  # stocker dict pour robustesse
        })

    if choix:
        description_response = requests.get("http://localhost:8000/columns_description", params={"column_name": choix})
        description = description_response.json().get("columns_description", "Description non disponible")
    else :
        description = "Aucune colonne sélectionnée."

    st.write(f"📌 **{choix}** : {description}")

    if lancer:
        try:
            specific_info_response = requests.get("http://localhost:8000/customer_specific_info", params={"sk_id_curr": SK_ID_CURR, "column_name": choix})
            specific_info = specific_info_response.json()
            customer_value = specific_info.get("customer_value", "Valeur non disponible")
            all_customers_values = specific_info.get("all_customers_values", "Valeur non disponible")
            mean_value = specific_info.get("mean_value", "Valeur moyenne non disponible")
            #st.write(f"Valeur pour le client {SK_ID_CURR} : {customer_value}")
            #st.write(f"Valeur moyenne pour cette colonne : {mean_value}")
        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {e}")

    # Affichage
    #st.write(f"📌 **{choix}** : {description}")
    #
    ##Graphique de comparaison client vs population
    if lancer:
        try:
            values=pd.Series(all_customers_values)
            n = len(values)
            lt = (values < customer_value).sum()
            eq = (values == customer_value).sum()
            percentile = (lt + 0.5 * eq) / n * 100
            vc = pd.to_numeric(pd.Series([customer_value]), errors="coerce").iloc[0]
            mv = pd.to_numeric(pd.Series([mean_value]), errors="coerce").iloc[0]
            delta = (vc - mv) if (pd.notna(vc) and pd.notna(mv)) else None

            c1, c2, c3 = st.columns(3)
            c1.metric("Valeur client", f"{vc:,.2f}".replace(",", " ") if pd.notna(vc) else "—")
            c2.metric(
                "Moyenne",
                f"{mv:,.2f}".replace(",", " ") if pd.notna(mv) else "—",
                delta=(f"{delta:+.2f}".replace(",", " ") if delta is not None else None),
                delta_color="off"  # neutre (pas de vert/rouge auto)
            )
            c3.metric("Percentile", f"{percentile:.1f}ᵉ")

            st.progress(int(percentile))  # jauge 0–100

            # Conversion en série numérique propre
            ser = pd.Series(all_customers_values)
            ser = pd.to_numeric(ser, errors="coerce").dropna()

            # On fabrique un DataFrame avec une seule colonne nommée comme la variable choisie
            df = pd.DataFrame({choix: ser})

            fig = px.histogram(
                df,
                x=choix,   # <- ici c’est le nom de la colonne du DataFrame
                nbins=50,
                title=f"Distribution de la colonne {choix} pour tous les clients",
                labels={"x": choix, "y": "Nombre de clients"}
            )

            try:
                customer_value_num = float(customer_value)
            except Exception:
                customer_value_num = None

            if customer_value_num is not None:
                fig.add_vline(
                    x=customer_value_num,
                    line_width=3,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Client {SK_ID_CURR}",
                    annotation_position="top right"
                )

            #st.plotly_chart(fig, use_container_width=True, key=f"current_chart_{choix}_{len(st.session_state['history'])}")
            _push_history(choix, SK_ID_CURR, fig, percentile=percentile, mean_value=mean_value)

        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {e}")
#


st.markdown("---")

with right:

    st.subheader("📚 Visualisation des variables")

    history = st.session_state["history"]
    if not history:
        st.caption("Aucune visualisation en mémoire pour l’instant.")
    else:
        # Sélection d'une visualisation passée
        items = [f"{h['column']} (client {h['sk_id']})" for i, h in enumerate(history)]
        idx = st.selectbox("Revoir une visu précédente :", options=list(range(len(items))), format_func=lambda i: items[i])

        # Afficher la figure sélectionnée
        past = history[idx]
        fig_past = go.Figure(past["fig_dict"])
        #st.plotly_chart(fig_past, use_container_width=True, key=f"history_selected_{idx}")

        # Option comparaison côte à côte avec la plus récente
        st.caption("Comparer avec la plus récente")
        if history:
            cols = st.columns(2)
            with cols[0]:
                st.write("◀️ Sélectionnée")
                st.plotly_chart(fig_past, use_container_width=True, key=f"compare_left_{idx}")
            with cols[1]:
                st.write("▶️ Plus récente")
                latest = history[-1]
                st.plotly_chart(go.Figure(latest["fig_dict"]), use_container_width=True, key=f"compare_right_{len(history)-1}")
    #
    #
    #
    #
#st.divider()
st.subheader("Importance globale des variables")

col_gi1, col_gi2, col_gi3 = st.columns([2,2,1])
with col_gi1:
    method = st.radio("Méthode", ["model"], horizontal=True, index=0, help="model=feature_importances_ LGBM (rapide); shap=mean(|SHAP|) (plus fidèle)")
#with col_gi2:
    #grouped = st.toggle("Agréger par variable brute", value=True, help="Additionne les importances des OneHot par colonne d'origine")
with col_gi3:
    top_k = st.number_input("Top K", min_value=5, max_value=50, value=20, step=1)

try:
    gi = fetch_global_importance(method=method, k=int(top_k), shap_sample=1000)
    items = gi.get("items", [])
    if items:
        df_gi = pd.DataFrame(items)
        df_gi = df_gi.sort_values("importance", ascending=True)  # pour barres horizontales
        fig_gi = go.Figure(go.Bar(
            x=df_gi["importance"], y=df_gi["name"], orientation="h",
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.5f}<extra></extra>"
        ))
        fig_gi.update_layout(
            height=max(320, 26 * len(df_gi)),
            xaxis_title="Importance",
            yaxis_title="",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_gi, use_container_width=True, key=f"global_imp_{method}_{len(df_gi)}")
    else:
        st.info("Aucune importance renvoyée.")
except requests.HTTPError as e:
    st.error(f"Erreur API: {getattr(e.response,'status_code',None)} — {e}")
except Exception as e:
    st.error(f"Erreur lors de l'affichage des importances: {e}")
