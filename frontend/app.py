import streamlit as st
import pandas as pd

# Titre de l'application
st.title("Dashboard explication risque crédit")

# Zone de texte
SK_ID_CURR = st.text_input("Entrez l'identifiant client")

# Bouton
if st.button("Charger les données"):
    try:
        df=pd.read_csv("https://projetbix.s3.eu-west-3.amazonaws.com/data_test_with_index.csv")
        st.write(df.head())  
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}") 