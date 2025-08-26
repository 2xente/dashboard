import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go


# Titre de l'application
st.title("Dashboard explication risque crédit")

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

# Zone de texte
SK_ID_CURR = st.text_input("Entrez l'identifiant client")
if st.button("Faire une prédiction"): 
    try:
        #loan prediciton
        response = requests.get("http://localhost:8000/prediction", params={"sk_id_curr": SK_ID_CURR})
        predictions = response.json()["predictions"]
        st.write(predictions)

        #client info
        customer_info = requests.get("http://localhost:8000/customer_info", params={"sk_id_curr": SK_ID_CURR})




    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}") 

# Columns information
columns_names_response = requests.get("http://localhost:8000/columns_names")
columns_names = columns_names_response.json()["columns_names"]

col1, col2 = st.columns([3, 1])

choix = col1.selectbox("Choisissez une colonne", columns_names)
lancer = col2.button("Generate")
if choix:
    description_response = requests.get("http://localhost:8000/columns_description", params={"column_name": choix})
    description = description_response.json().get("columns_description", "Description non disponible")

if lancer:
    try:
        specific_info_response = requests.get("http://localhost:8000/customer_specific_info", params={"sk_id_curr": SK_ID_CURR, "column_name": choix})
        specific_info = specific_info_response.json()
        customer_value = specific_info.get("customer_value", "Valeur non disponible")
        all_customers_values = specific_info.get("all_customers_values", "Valeur non disponible")
        mean_value = specific_info.get("mean_value", "Valeur moyenne non disponible")
        st.write(f"Valeur pour le client {SK_ID_CURR} : {customer_value}")
        st.write(f"Valeur moyenne pour cette colonne : {mean_value}")
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")

# Affichage
st.write(f"📌 **{choix}** : {description}")

#Graphique de comparaison client vs population
if lancer:
    try:
        values=pd;Series(all_customers_values)
        n = len(values)
        lt = (values < customer_value).sum()
        eq = (values == customer_value).sum()
        precentile = (lt + 0.5 * eq) / n * 100
        st.write(f"Le client est au {precentile:.2f}ème percentile pour la colonne {choix}")

        fig = px.histogram(
            x=all_customers_values,
            nbins=50,
            title=f"Distribution de la colonne {choix} pour tous les clients",
            labels={ "x": choix, "y": "Nombre de clients" }
        )
        fig.add_vline(
            x=customer_value,
            line_width=3,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Client {SK_ID_CURR}",
            annotation_position="top right"
        )
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")






